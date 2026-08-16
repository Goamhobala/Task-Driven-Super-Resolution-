"""Joint SR + UNet Lightning module — the unet baseline with an SR front-end.

Subclasses :class:`unet.model.UNetLightning` so the segmentation network, the
loss (legacy Dice + weighted-BCE, or any ``unet.losses.build_loss`` arm via
``loss_arm``), the per-crop IoU/F1 metrics and the ``val_iou`` monitoring
convention are IDENTICAL to the baseline; the only differences are:

  * ``forward`` prepends an upsampler: raw-DN input / 10000 -> SEN2SR (or
    parameter-free bicubic) -> x 10000 -> frozen per-band z-score -> UNet.
    The dataset (``JointSRDataModule``) therefore feeds raw DN, and the
    normalisation the baseline applies in the dataloader happens HERE, after
    super-resolution, as a differentiable affine (gradients flow through it
    into SEN2SR).
  * ``configure_optimizers`` uses two parameter groups: the UNet at ``lr``
    and the SR network at ``lr_sr``. Task-driven SR (Haris et al.): the ONLY
    loss is the segmentation loss; ``lr_sr / lr`` plays the role of the
    gradient-scaling alpha, so keep ``lr_sr`` well below ``lr`` (both are
    Optuna-searchable via ``sr.tune``).

Upsampler configs (mirrors the R-series):
    bicubic                      R0 deterministic baseline (no SR params)
    sen2sr + freeze_sr=true      R1 frozen SEN2SR preprocessing
    sen2sr + freeze_sr=false     R2 joint task-driven fine-tuning

SEN2SR constraints (see sr/README.md): input must be raw reflectance in
[B4, B3, B2, B8] order == V2 bands (1, 2, 3, 4) — do NOT pass the enhanced-RGB
bands 21-23 — and the shipped FFT mask pins the LR patch to 128 px (-> 512 HR).

Adaptive post-SR normalisation (docs/adaptive_norm_plan.md) is available but
OFF by default: ``adaptive_norm`` (per-batch EMA of the post-SR moments) and
``norm_recalibrate`` (exact PreciseBN-style recompute) both default to the
legacy frozen-dataset-stats behaviour, so every existing arm and benchmark row
stays bit-for-bit comparable until the flags are set.
"""
from __future__ import annotations

import torch

from sr.sen2sr_loader import (
    SEN2SR_SCALE,
    BicubicUpsampler,
    TrainableSEN2SR,
    build_hard_constraint,
    load_trainable_sen2sr,
    load_trainable_sen2sr_full,
    pad_low_pass_mask,
    resolve_sr_hc,
)
from unet.model import UNetLightning

# DN -> surface reflectance divisor expected by SEN2SR.
REFLECTANCE_SCALE = 10000.0

# SEN2SR's required input bands (V2 layout, 1-based): [B4, B3, B2, B8].
SEN2SR_BANDS = (1, 2, 3, 4)

# --- adaptive post-SR normalisation constants (docs/adaptive_norm_plan.md) ---
# Variance floor for the derived std: keeps the 1e-6 std guard the frozen
# adapter already used, expressed in variance units.
_VAR_FLOOR = 1e-12
# Momentum of the SECOND, faster EMA kept purely as a diagnostic: the gap
# between it and the slow (adaptive_norm_momentum) EMA IS the lag term the
# plan asks to monitor (§4.1). Never used in the forward path.
_ADAPT_FAST_M = 0.2
# Std band defaults, as multiples of the construction-time dataset std (§4.3).
# Warn first, then RAISE — deliberately not a clamp: a silent floor is a
# trapdoor that only engages long after the run entered the failure mode.
#
# ASYMMETRIC, because the two directions are not equally dangerous:
#   * COLLAPSE is the catastrophe this feature exists to prevent. The adapter's
#     gradient gain into the generator is reflectance_scale/band_std, so as std
#     falls the gain explodes and the drift it causes accelerates — positive
#     feedback. Tight bound, non-negotiable.
#   * GROWTH is self-stabilising under a tracking normaliser: rising std LOWERS
#     the gain, and the U-Net's input stays whitened either way. Some contrast
#     growth is arguably what task-driven SR is for. Its real risk is the SR
#     output ceasing to resemble imagery, which is `sr_psnr_vs_init`'s job, not
#     a moment's. Loose bound, and the warning is the actionable signal.
# Empirical anchor: the first tune run (2026-08-13) tripped a symmetric 2.0
# upper bound on a SEN2SR trial whose sampled lr_sr=3.1e-4 was 30x the design
# default — means DC-pinned and steady, blue-band std growing monotonically
# (1.6x @ step 1050 -> 2.1x @ 1400). Correct detection, wrong severity.
_STD_WARN_LO, _STD_WARN_HI = 0.7, 1.5
_STD_RAISE_LO, _STD_RAISE_HI = 0.5, 4.0
# scripts/sr4rs/sanity_viz.py's own "LARGE — investigate!" threshold, reused
# so the training-time monitor and the offline check agree on what is large.
_MEAN_DRIFT_WARN = 0.05
# Valid values for `norm_recalibrate` (§4.4).
NORM_RECALIBRATE_MODES = ("off", "pre", "post", "auto")
# Batches used by the before/after IoU probe around a recalibration swap.
# A bounded SUBSAMPLE, deliberately named as such wherever it is reported.
_VAL_SNAPSHOT_BATCHES = 64


class AdaptiveNormBandExit(RuntimeError):
    """The post-SR band std left its hard band (§4.3).

    A dedicated type so callers can distinguish "this hyperparameter corner is
    unusable" from a genuine bug. `sr.tune` catches it and prunes the TRIAL —
    a band exit is exactly the kind of signal the sampler should learn from,
    and letting it propagate would take down the whole study over one bad
    corner of the search space (which is what happened on 2026-08-13).

    Production fits do NOT catch it: there the loud failure is the point.
    """

    def __init__(self, message, ratios=None, band=None):
        super().__init__(message)
        self.ratios = list(ratios) if ratios is not None else []
        self.band = band


class _preserve_rng:
    """Run a block without letting it disturb the global RNG stream.

    Building a DataLoader iterator draws a ``_base_seed`` from the global
    default generator (``torch.utils.data.dataloader``), which sets the random
    crops the training loader will then hand out. So an auxiliary loader — the
    recalibration pass, the drift monitor's fixed batch, the IoU probe — would
    silently change which crops the RUN sees, i.e. break same-seed
    reproducibility of arms whose only change is a monitor being switched on.
    These helpers must be numerically inert, and this is what makes them so.
    """

    def __enter__(self):
        self._cpu = torch.get_rng_state()
        self._cuda = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._cpu)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False


class JointSRUNetLightning(UNetLightning):
    """UNet baseline + jointly-tuned SR front-end (two learning rates)."""

    def __init__(
        self,
        # --- UNet baseline args (identical to UNetLightning) ---------------
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 4,
        classes: int = 1,
        lr: float = 1e-4,
        pos_weight: float = 5.0,
        bands: tuple[int, ...] = SEN2SR_BANDS,
        image_size: int = 256,
        threshold: float = 0.5,
        normalize: bool = True,
        norm_mean: list[float] | None = None,
        norm_std: list[float] | None = None,
        # --- SR front-end ---------------------------------------------------
        upsampler: str = "sen2sr",
        sen2sr_dir: str | None = None,
        lr_sr: float = 1e-5,
        freeze_sr: bool = False,
        upscale: int = 4,
        sr_pad: int = 0,
        # --- FFT hard constraint x generator (docs/hc_2x2_plan.md) ----------
        # The constraint is a pure function of (lr, sr, mask) — it touches no
        # generator internals — so it can be taken off SEN2SR or put on SR4RS,
        # which is what separates "the constraint helps" from "that generator
        # is better". The treatment is the whole BUNDLE: positivity clamp +
        # frequency splice (see TrainableSEN2SR), with the sr_pad border
        # mitigation assigned by the arm script, not here.
        #   native  each upsampler's shipped default (sen2sr on, sr4rs and
        #           bicubic off). Checkpoints written before this hparam
        #           existed have no key, resolve to `native`, and therefore
        #           reconstruct byte-identically.
        #   on/off  forced. `bicubic + on` raises (a near-identity cell).
        sr_hc: str = "native",
        # Where to read the low-pass mask when the constraint is mounted on a
        # generator that does not ship one (sr4rs). The SEN2SR arms take theirs
        # from sen2sr_dir as they always have. Reuse the shipped file
        # byte-for-byte — it IS the deployed cutoff.
        hc_mask_path: str | None = None,
        reflectance_scale: float = 10000.0,
        # --- recipe v2 training dynamics ------------------------------------
        # lr_schedule: "cosine" = per-step cosine of BOTH LR groups to 0 over
        # the run's budget (alpha = lr_sr/lr stays constant; the SR group gets
        # its adapt-early-lock-late decay). "none" = legacy constant LRs --
        # the default, so old checkpoints restore under the recipe that
        # trained them; recipe-v2 configs set cosine explicitly.
        lr_schedule: str = "none",
        # Linear LR ramp (0 -> lr_sr) on the SR group only, in epochs
        # (fractional ok; converted to steps). LP-FT rationale: skip the
        # phase-1 window where a random decoder sends destructive noise into
        # the pretrained generator -- NOT the phase-2 co-adaptation the joint
        # arms exist to measure, so keep it short (~1 epoch). AUTO-DISABLED
        # when structurally covered: frozen/bicubic SR (no gradients to
        # protect against) or staged warm starts (stage-1 IS the warmup).
        sr_warmup_epochs: float = 1.0,
        # Optional L2-SP anchor (Li et al. 2018, arXiv:1802.01483):
        # + l2sp_lambda * sum ||theta_sr - theta_sr,0||^2. Decays toward the
        # PRETRAINED weights (unlike weight decay's pull toward 0). Default 0
        # = pure task-driven SR (Haris et al. TDSR-DET); escalate only on
        # sr_drift_rel evidence, per the pre-registered protocol.
        l2sp_lambda: float = 0.0,
        # SR-evolution snapshots (demo/insight): every N epochs during fit,
        # save the SR net's state_dict ALONE (~45 MB SR4RS / ~2 MB SEN2SR-Lite
        # vs ~500 MB for a full Lightning ckpt) so the SR output's evolution
        # under the task loss can be replayed frame by frame. An extra
        # "epoch_000_init" frame captures the pretrained net before any task
        # gradient. 0 = off. Auto-skipped when the SR net has no trainable
        # params (frozen/bicubic -- nothing evolves).
        sr_snapshot_every: int = 0,
        sr_snapshot_dir: str = "sr_snapshots",
        # Staged warm start (R6/R7): path to a stage-1 (frozen-SR, R5/R1) ckpt
        # whose UNet weights initialise THIS model's UNet; the SR net still
        # loads from sen2sr_dir. Rationale: the UNet is the task-critic whose
        # gradients sculpt the SR net — warm-starting it on the frozen-SR
        # input distribution avoids the destructive early gradients an
        # incompetent critic sends into a pretrained generator (Grigoryev et
        # al. 2022 analogue). Guarded: source ckpt must match this model's
        # reflectance_scale; encoder/head mismatches fail via strict load.
        # None/"" = ImageNet init (cold joint, R2/R4 protocol). When RESTORING
        # a stage-2 ckpt the restored weights overwrite this init anyway —
        # pass warm_start_unet=None at load_from_checkpoint if the stage-1
        # file is not on this machine (viz_grid does).
        warm_start_unet: str | None = None,
        # --- adaptive post-SR normalisation (docs/adaptive_norm_plan.md) ----
        # The post-SR z-score below uses FROZEN dataset statistics, which are
        # only valid while the SR output distribution stays put. SEN2SR's FFT
        # HardConstraint pins each band's DC component to the bicubic input, so
        # its output means cannot drift far; SR4RS has no such anchor
        # (PixelNorm divides out feature magnitude, no clamp, no DC pin), so
        # under task-only joint fine-tuning its output level is free to wander
        # and the adapter's gradient gain reflectance_scale/band_std (~15-33x)
        # compounds the drift. These flags make the z-score TRACK the SR output
        # instead of assuming it never moves.
        #
        # adaptive_norm: per-batch EMA of the post-SR per-band moments, held on
        # STOP-GRAD buffers (BatchNorm running-stats semantics, Ioffe &
        # Szegedy 2015 — NOT batch-stats-in-forward, which would route
        # gradients through the statistics and change the optimisation
        # problem). Default False = byte-identical legacy behaviour.
        adaptive_norm: bool = False,
        # EMA coefficient: new = (1-m)*old + m*batch. Horizon ~ 1/m steps.
        # PINNED, never searched (the batch-size-confound lesson): m is also
        # the only loss-side restoring force against SR output drift, because
        # a perfectly tracking normaliser makes the task loss exactly
        # invariant to any per-band affine change of the SR output (§7).
        # Slower EMA = more restoring force, more staleness. Diagnose with
        # `adapt_lag_max` rather than tuning this.
        adaptive_norm_momentum: float = 0.01,
        # Delay the EMA until this global step (e.g. past the SR LR ramp) so
        # it does not chase a frozen-SR phase. 0 = update from step 0.
        adaptive_norm_warmup_steps: int = 0,
        # Steps between std-hard-band checks. Each check costs one host sync,
        # so it is not done every step; at the default momentum (horizon 100
        # steps) 50 cannot miss an excursion by more than half a horizon.
        adaptive_norm_check_every: int = 50,
        # Std band limits, as multiples of the run's starting std. ASYMMETRIC
        # by design — see the module constants for why collapse and growth are
        # not equally dangerous. Exposed as hparams so an arm with a known
        # reason to expect contrast growth can loosen the upper side without
        # touching the lower one, which is the actual catastrophe bound.
        std_band_raise_lo: float = _STD_RAISE_LO,
        std_band_raise_hi: float = _STD_RAISE_HI,
        std_band_warn_lo: float = _STD_WARN_LO,
        std_band_warn_hi: float = _STD_WARN_HI,
        # Exact PreciseBN-style recompute of the post-SR statistics (Wu &
        # Johnson 2021), over `norm_recalibrate_batches` train batches through
        # the SR stage ONLY (no UNet, no backward):
        #   off   never (default = legacy behaviour)
        #   pre   once at on_fit_start, BEFORE training. The complete and
        #         strictly correct fix for FROZEN-SR arms (r1/r5), whose SR
        #         output distribution is stationary but whose adapter is stale
        #         from step 0 (frozen SR4RS output stats != dataset stats).
        #   post  at the end of training, after which `last.ckpt` (and the
        #         UNMONITORED `_final.ckpt`) are REWRITTEN so they carry
        #         population statistics — see on_train_end for why re-saving
        #         is necessary rather than relying on hook order. Removes the
        #         EMA's residual lag; this is what makes the test numbers
        #         clean, since bench/test run as separate processes off disk.
        #   auto  pre for frozen SR nets with parameters, post when
        #         adaptive_norm is on, off for bicubic (out of scope: R0 is
        #         the untouched deterministic baseline).
        norm_recalibrate: str = "off",
        norm_recalibrate_batches: int = 200,
        # Functional (image-space) drift monitor: PSNR/RMSE of the CURRENT SR
        # net's output against the INITIAL net's output on one fixed batch.
        # No ground truth is involved — both tensors are SR outputs on
        # identical inputs, so these are drift distances, not quality scores.
        # Answers what neither the weight-space `sr_drift_rel` (cosine-
        # confounded) nor the moments can: falling PSNR-vs-init at stable
        # mean/std means the SR is restructuring content while keeping its
        # statistics. Numerically inert: no_grad, one fixed batch, and the
        # auxiliary loader is built inside `_preserve_rng` so it cannot shift
        # the crop stream the run trains on. Auto-skipped when the SR net has
        # no trainable parameters (nothing evolves).
        sr_functional_monitor: bool = True,
        sr_monitor_samples: int = 2,
        # --- loss (unet.losses.build_loss pass-through) ---------------------
        # Same names/defaults as UNetLightning. None = legacy Dice + weighted
        # BCE. NB sr_w/sr_radius are the Skeleton-Recall loss knobs (parent's
        # naming), NOT super-resolution knobs — the SR net's are lr_sr/sr_pad.
        loss_arm: str | None = None,
        pstar: str = "bce",
        gap_r: int = 4,
        gap_k: float = 60.0,
        tl_ell: int = 5,
        tl_theta: float = 0.375,
        gap_theta: float = 0.5,
        tversky_alpha: float = 0.7,
        mix_w: float = 0.5,
        cl_alpha: float = 0.3,
        cl_iters: int = 5,
        sr_w: float = 1.0,
        sr_radius: int = 1,
        warmup_start: int = 30,
        warmup_ramp: int = 10,
        # --- read-out head: RL-series (docs/sr_linear_probe.md) -------------
        # "unet" (default) = every R-arm, unchanged. "linear" = a 5-parameter
        # 1x1 conv after the z-score adapter, which removes the decoder's
        # ability to compensate for a bad SR front-end.
        head: str = "unet",
        # logit(prior) bias init. None = measured from the training masks in
        # setup() and written into hparams, so the value that was actually used
        # is recorded in the checkpoint rather than recomputed on restore.
        head_bias_prior: float | None = None,
        # Cap on the batches setup() streams for that measurement. A base rate
        # near 0.2 is estimated to well under a percent from a few hundred
        # batches, and this runs before EVERY fit — an unbounded pass over the
        # 2.5 m masks is not worth the precision it would buy.
        head_bias_batches: int = 200,
        # LP-FT warm start for the joint arms (rl2 <- rl1, rl4 <- rl3): path to
        # the frozen twin's FINAL ckpt, whose head weights initialise this
        # model's probe. STRICTLY SEPARATE from warm_start_unet, which zeroes
        # the SR warmup ramp (correct when a staged U-Net start IS the warmup,
        # wrong here) — routing the head through that flag would silently drop
        # sr_warmup_epochs and nothing in the run would say so.
        warm_start_head: str | None = None,
        # Per-parameter-GROUP gradient clipping (§6.2). Lightning's
        # `gradient_clip_val` is a single global L2 norm over all trainable
        # params, so "1.0" means one thing in rl1 (5 params) and another in rl2
        # (5 + ~240 k SEN2SR params) — the nuisance variable would move with the
        # treatment. > 0 clips the SR group ONLY, at this norm, and leaves the
        # head unclipped; the Trainer-level clip must then be off. 0 = the old
        # global behaviour, i.e. every R-arm.
        clip_sr: float = 0.0,
    ):
        # reflectance_scale: divisor mapping the dataloader's raw values to the
        # 0-1 reflectance the SR nets expect. 10000.0 for DN-valued COGs;
        # **1.0 for the ROSA V2 datasets, whose COGs already store 0-1
        # reflectance** (verified: tile values ~0.02-0.6, norm_stats means
        # ~0.05-0.24). With the wrong 10000.0 on reflectance data the SR nets
        # receive ~1e-5 inputs: bicubic/R0 is unaffected (linear ops cancel),
        # but SEN2SR degenerates to its DC-anchored bicubic and SR4RS emits its
        # zero-input pattern. Old ckpts (no such hparam) load with 10000.0,
        # matching how they were trained.
        super().__init__(
            encoder_name=encoder_name, encoder_weights=encoder_weights,
            in_channels=in_channels, classes=classes, lr=lr,
            pos_weight=pos_weight, bands=bands, image_size=image_size,
            threshold=threshold, normalize=normalize,
            norm_mean=norm_mean, norm_std=norm_std,
            loss_arm=loss_arm, pstar=pstar, gap_r=gap_r, gap_k=gap_k,
            tl_ell=tl_ell, tl_theta=tl_theta, gap_theta=gap_theta,
            tversky_alpha=tversky_alpha, mix_w=mix_w,
            cl_alpha=cl_alpha, cl_iters=cl_iters, sr_w=sr_w,
            sr_radius=sr_radius, warmup_start=warmup_start,
            warmup_ramp=warmup_ramp,
            head=head, head_bias_prior=head_bias_prior,
        )
        if head == "linear":
            # No encoder exists. Blank the two hparams so a linear-probe run can
            # never be read back — from the checkpoint, the bench store or the
            # overlay — as though it were a resnet34 encoder ablation.
            self.hparams.encoder_name = None
            self.hparams.encoder_weights = None
        # NOTE: no second save_hyperparameters() call — the parent's call
        # already captures this subclass's full init signature (Lightning
        # walks the __init__ frames), including upsampler/lr_sr/freeze_sr.

        # The post-SR adapter needs the frozen stats — per-image fallback is
        # not possible here (the baseline's dataset-side normalisation is).
        if norm_mean is None or norm_std is None:
            raise ValueError(
                "JointSRUNetLightning needs frozen norm stats (norm_mean/"
                "norm_std) — layer norm_stats.yaml over the base config."
            )
        # Validate the adaptive-norm settings BEFORE the SR weights load: a
        # typo'd mode should not cost a full SEN2SR/SR4RS load (and, for
        # sen2sr_full, the mlstac wrapper) to discover.
        if norm_recalibrate not in NORM_RECALIBRATE_MODES:
            raise ValueError(f"norm_recalibrate={norm_recalibrate!r} "
                             f"(choose from {NORM_RECALIBRATE_MODES})")
        if not (0.0 < float(adaptive_norm_momentum) <= 1.0):
            raise ValueError("adaptive_norm_momentum must be in (0, 1], got "
                             f"{adaptive_norm_momentum}")

        # Resolve the constraint BEFORE any weights load: a bad combination
        # should not cost a full SEN2SR/SR4RS load to discover. `_sr_hc_on` is
        # the single resolved answer every branch below (and the checkpoint
        # guard) reads, so `native` can never mean two things in one process.
        self._sr_hc_on = resolve_sr_hc(upsampler, sr_hc)

        if upsampler in ("sen2sr", "sen2sr_full"):
            if sen2sr_dir is None:
                raise ValueError(f"upsampler={upsampler!r} needs sen2sr_dir "
                                 "(Lite: prefetch with sr.sen2sr_loader.download_sen2sr)")
            if tuple(bands) != SEN2SR_BANDS:
                raise ValueError(
                    f"SEN2SR requires raw reflectance bands {SEN2SR_BANDS} "
                    f"([B4,B3,B2,B8] = R,G,B,NIR), got {tuple(bands)}. The "
                    "enhanced-RGB bands (21-23) are not valid SEN2SR input."
                )
            if upscale != SEN2SR_SCALE:
                raise ValueError(f"SEN2SR is a fixed x{SEN2SR_SCALE} model; got upscale={upscale}")
            # Lite = CNNSR via the train_mode fix; full = MambaSR via mlstac's
            # own trainable_model (no collapse quirk; needs mamba_ssm).
            loader = (load_trainable_sen2sr if upsampler == "sen2sr"
                      else load_trainable_sen2sr_full)
            self.sr = loader(sen2sr_dir, hard_constraint=self._sr_hc_on)
            if self._sr_hc_on:
                # The shipped FFT low-pass mask fixes the HR size -> LR patches
                # are pinned to mask_size / scale (512 / 4 = 128). Checked in
                # forward. (Computed BEFORE any pad-resize: it constrains the
                # MODEL-facing input; sr_pad grows the mask so the padded input
                # still fits.)
                self._required_lr = (self.sr.hard_constraint.low_pass_mask
                                     .shape[-1] // SEN2SR_SCALE)
                if sr_pad > 0:
                    # Border-artifact mitigation: the FFT constraint assumes a
                    # periodic patch, so edge discontinuities ring at the
                    # borders. Reflect-padding the input and cropping the
                    # output moves the ring into discarded context (see
                    # forward).
                    pad_low_pass_mask(self.sr, sr_pad)
            else:
                # r2b: no mask, so nothing pins the input size — CNNSR/MambaSR
                # are fully convolutional. crop_size stays 128 via the data
                # config, so the input path is unchanged; sr_pad (if ever set)
                # is still honoured by the generic pad/crop in _sr_forward.
                self._required_lr = None
        elif upsampler == "sr4rs":
            if sen2sr_dir is None:
                raise ValueError("upsampler='sr4rs' needs sen2sr_dir pointing at "
                                 "the SR4RS_RGBN dir (run extract_sr4rs.py first)")
            if tuple(bands) != SEN2SR_BANDS:
                raise ValueError(f"SR4RS_RGBN requires raw bands {SEN2SR_BANDS}, got {tuple(bands)}")
            if upscale != 4:
                raise ValueError(f"SR4RS is a fixed x4 model; got upscale={upscale}")
            from sr.sr4rs_torch import load_trainable_sr4rs
            # Same reflectance in/out contract as SEN2SR (SR4RS's LRSC/HRSC
            # 1e-4 scaling == our /REFLECTANCE_SCALE). Fully convolutional:
            # no FFT mask, no input-size pin. sr_pad still applies (GAN edge
            # effects), via the generic pad/crop in forward.
            self.sr = load_trainable_sr4rs(sen2sr_dir)
            self._required_lr = None
            if self._sr_hc_on:
                # r4a: mount SEN2SR's constraint on the GAN generator. The
                # operator is architecture-agnostic, and data.crop_size is 128
                # for every arm, so the shipped 512x512 mask applies unchanged
                # — no resize, no re-derivation. Mirror of the sen2sr branch
                # from here on, including the pad-grown mask.
                if not hc_mask_path:
                    raise ValueError(
                        "sr_hc='on' with upsampler='sr4rs' needs hc_mask_path "
                        "pointing at SEN2SR-Lite's hard_constraint.safetensor "
                        "(sen2sr_dir here is the SR4RS_RGBN dir, which ships "
                        "no mask)."
                    )
                self.sr = TrainableSEN2SR(
                    self.sr, build_hard_constraint(hc_mask_path), clamp_min=0.0)
                self._required_lr = (self.sr.hard_constraint.low_pass_mask
                                     .shape[-1] // upscale)
                if sr_pad > 0:
                    pad_low_pass_mask(self.sr, sr_pad, scale=upscale)
        elif upsampler == "bicubic":
            if freeze_sr:
                raise ValueError("freeze_sr is meaningless with the parameter-free bicubic upsampler")
            self.sr = BicubicUpsampler(upscale)
            self._required_lr = None
        else:
            raise ValueError(f"unknown upsampler {upsampler!r} "
                             "(bicubic | sen2sr | sen2sr_full | sr4rs)")

        if freeze_sr:
            for p in self.sr.parameters():
                p.requires_grad = False

        # --- recipe v2: theta_0 snapshot of the trainable SR params ---------
        # Feeds (a) the ALWAYS-ON drift monitor `sr_drift_rel` =
        # ||theta-theta_0|| / ||theta_0||, logged every val epoch -- the
        # measured answer to "is task-driven fine-tuning pulling the generator
        # off its pretrained prior?" -- and (b) the optional L2-SP anchor
        # above. Buffers are non-persistent: checkpoints stay small and
        # theta_0 is rebuilt from sen2sr_dir's pretrained weights on every
        # construction (so it stays the PRETRAINED reference even when
        # resuming a fine-tuned checkpoint).
        sr_train = [p for p in self.sr.parameters() if p.requires_grad]
        self._n_sr_p0 = len(sr_train)
        for i, p in enumerate(sr_train):
            self.register_buffer(f"_sr_p0_{i}", p.detach().clone(),
                                 persistent=False)
        self._sr_warmup_epochs = (float(sr_warmup_epochs)
                                  if (sr_train and not warm_start_unet)
                                  else 0.0)

        # Differentiable post-SR normalisation adapter: the same frozen z-score
        # the baseline's dataloader applies, sliced to this model's bands.
        idx = [b - 1 for b in bands]
        mean = torch.as_tensor([float(norm_mean[i]) for i in idx],
                               dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.as_tensor([float(norm_std[i]) for i in idx],
                              dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("band_mean", mean)
        self.register_buffer("band_std", torch.where(std > 1e-6, std, torch.ones_like(std)))

        # --- adaptive-norm state -------------------------------------------
        # NOTE there is deliberately NO extra persistent buffer for E[x^2]:
        # (band_mean, band_std) already encode it exactly (m2 = std^2 + mean^2),
        # so the EMA reconstructs it in-place each step. That keeps the
        # checkpoint's tensor set unchanged — old checkpoints still load
        # strictly, and a RESUME_FIT resumes the EMA exactly where it stopped.
        # (The one place the identity breaks is the _VAR_FLOOR clamp; that path
        # warns loudly — see _adapt_update.)
        # Non-persistent, exactly like the _sr_p0_* theta_0 snapshots.
        self.register_buffer("_adapt_init_mean", mean.clone(), persistent=False)
        self.register_buffer("_adapt_init_std", self.band_std.clone(),
                             persistent=False)
        # Faster EMAs of the RAW batch moments — the lag diagnostic only. Both
        # moments, not just the mean: the std is what the hard band and the
        # rs/band_std gradient-gain argument actually care about.
        self.register_buffer("_adapt_fast_mean", mean.clone().view(-1),
                             persistent=False)
        self.register_buffer("_adapt_fast_std", self.band_std.clone().view(-1),
                             persistent=False)
        self.register_buffer("_adapt_skips", torch.zeros((), dtype=torch.float32),
                             persistent=False)
        self._adapt_warned = False
        self._adapt_drift_warned = False
        self._adapt_floor_warned = False
        self._recal_done = False
        # Functional drift monitor cache (plain attributes: never checkpointed,
        # deliberately tensors rather than a second model instance — SEN2SR-
        # full's footprint already caused a 44 GB OOM once).
        self._srmon_x = None
        self._srmon_ref = None

        if warm_start_unet:
            self._load_unet_from(warm_start_unet)
            # A warm start REPLACES the starting statistics (the stage-1 arm's,
            # adopted above). The guard band and the drift logs must measure
            # movement from where THIS run actually starts, not from the
            # config's dataset stats — otherwise a stage-1 arm that legitimately
            # recalibrated hands stage 2 a step-0 "drift" it never caused, and
            # for frozen SR4RS (whose output stats differ from the dataset stats
            # by construction) that can trip the hard band before a single
            # gradient step.
            self._adapt_init_mean.copy_(self.band_mean)
            self._adapt_init_std.copy_(self.band_std)
            self._adapt_fast_mean.copy_(self.band_mean.reshape(-1))
            self._adapt_fast_std.copy_(self.band_std.reshape(-1))

        # Consumed by setup(), which must NOT overwrite a warm-started bias.
        self._head_warm_started = bool(warm_start_head)
        if warm_start_head:
            if head != "linear":
                raise ValueError(
                    f"warm_start_head is set but head={head!r} — there is no "
                    "linear probe to warm-start. The staged U-Net path is "
                    "warm_start_unet."
                )
            if warm_start_unet:
                raise ValueError(
                    "warm_start_head and warm_start_unet are mutually exclusive: "
                    "warm_start_unet auto-disables the SR warmup ramp, which the "
                    "LP-FT path must keep (docs/sr_linear_probe.md §2)."
                )
            self._load_head_from(warm_start_head)
            # Same reasoning as the staged U-Net start: the probe was fitted
            # against the twin's statistics, so the guard band and the drift
            # logs must measure movement from where THIS run starts.
            self._adapt_init_mean.copy_(self.band_mean)
            self._adapt_init_std.copy_(self.band_std)
            self._adapt_fast_mean.copy_(self.band_mean.reshape(-1))
            self._adapt_fast_std.copy_(self.band_std.reshape(-1))

    # ------------------------------------------------ LP-FT head warm start
    def _load_head_from(self, ckpt_path: str):
        """Initialise the linear probe from the frozen twin's FINAL ckpt.

        Mirrors :meth:`_load_unet_from` — including adopting the source's
        post-SR normalisation buffers — but deliberately does NOT touch
        ``self._sr_warmup_epochs``. That is the entire reason this is a separate
        code path rather than a branch inside the U-Net one.
        """
        from pathlib import Path
        p = Path(ckpt_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"warm_start_head={ckpt_path!r} not found — fit the frozen twin "
                "(rl1 for rl2, rl3 for rl4) first."
            )
        ck = torch.load(p, map_location="cpu", weights_only=False)
        src_hp = ck.get("hyper_parameters", {})
        src_scale = float(src_hp.get("reflectance_scale", 10000.0))
        if src_scale != float(self.hparams.reflectance_scale):
            raise ValueError(
                f"warm_start_head ckpt was trained with reflectance_scale="
                f"{src_scale} but this model uses "
                f"{self.hparams.reflectance_scale}."
            )
        if src_hp.get("head") != "linear":
            raise ValueError(
                f"warm_start_head ckpt has head={src_hp.get('head')!r}, expected "
                "'linear'. A U-Net checkpoint cannot seed a linear probe — check "
                "STAGE1_TAG points at an rl arm, not an r arm."
            )
        head_sd = {k[len("model."):]: v for k, v in ck["state_dict"].items()
                   if k.startswith("model.")}
        if not head_sd:
            raise ValueError(f"no 'model.*' keys in {ckpt_path}.")
        self.model.load_state_dict(head_sd, strict=True)
        sd = ck["state_dict"]
        if "band_mean" in sd and "band_std" in sd:
            src_mean = sd["band_mean"].to(self.band_mean.dtype)
            src_std = sd["band_std"].to(self.band_std.dtype)
            if src_mean.shape != self.band_mean.shape:
                raise ValueError(
                    f"warm_start_head ckpt has band_mean of shape "
                    f"{tuple(src_mean.shape)}, this model expects "
                    f"{tuple(self.band_mean.shape)} — different band count?")
            d = float((src_mean - self.band_mean).abs().max())
            self.band_mean.copy_(src_mean)
            self.band_std.copy_(src_std)
            print(f"[joint_sr] adopted twin's post-SR norm buffers "
                  f"(max |Δmean| = {d:.4g} vs this config's dataset stats)")
        w = self.model.conv.weight.detach().flatten().tolist()
        b = float(self.model.conv.bias.detach().flatten()[0])
        print(f"[joint_sr] LP-FT: warm-started the linear probe from {p.name} "
              f"(upsampler={src_hp.get('upsampler')!r}, "
              f"freeze_sr={src_hp.get('freeze_sr')}, epoch={ck.get('epoch')})")
        print(f"[joint_sr]   w={['%.4g' % v for v in w]}  b={b:.4g}  "
              f"sr_warmup_epochs stays {self._sr_warmup_epochs} (NOT zeroed)")

    # ----------------------------------------------------- staged warm start
    def _load_unet_from(self, ckpt_path: str):
        """Initialise self.model (the UNet) from a stage-1 JointSR ckpt."""
        from pathlib import Path
        p = Path(ckpt_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"warm_start_unet={ckpt_path!r} not found — fit the stage-1 "
                "(frozen-SR) arm first, or pass warm_start_unet=None."
            )
        ck = torch.load(p, map_location="cpu", weights_only=False)
        src_scale = float(ck.get("hyper_parameters", {})
                            .get("reflectance_scale", 10000.0))
        if src_scale != float(self.hparams.reflectance_scale):
            raise ValueError(
                f"warm_start_unet ckpt was trained with reflectance_scale="
                f"{src_scale} but this model uses "
                f"{self.hparams.reflectance_scale} — a starved-era or "
                "DN-dataset UNet cannot seed a V2 reflectance run."
            )
        unet_sd = {k[len("model."):]: v for k, v in ck["state_dict"].items()
                   if k.startswith("model.")}
        if not unet_sd:
            raise ValueError(f"no 'model.*' keys in {ckpt_path} — not a "
                             "JointSR/UNet Lightning checkpoint?")
        self.model.load_state_dict(unet_sd, strict=True)
        # §4.5: carry the stage-1 normalisation buffers across too. The
        # warm-started UNet was trained against THOSE statistics; rebuilding
        # them from the dataset stats would hand it a step-0 jump in its own
        # input distribution. A no-op (values identical) for a stage-1 arm that
        # ran with neither adaptive_norm nor recalibration.
        sd = ck["state_dict"]
        if "band_mean" in sd and "band_std" in sd:
            src_mean = sd["band_mean"].to(self.band_mean.dtype)
            src_std = sd["band_std"].to(self.band_std.dtype)
            if src_mean.shape != self.band_mean.shape:
                raise ValueError(
                    f"warm_start_unet ckpt has band_mean of shape "
                    f"{tuple(src_mean.shape)}, this model expects "
                    f"{tuple(self.band_mean.shape)} — different band count?")
            d = float((src_mean - self.band_mean).abs().max())
            self.band_mean.copy_(src_mean)
            self.band_std.copy_(src_std)
            print(f"[joint_sr] adopted stage-1 post-SR norm buffers "
                  f"(max |Δmean| = {d:.4g} vs this config's dataset stats)")
        src = ck.get("hyper_parameters", {})
        print(f"[joint_sr] warm-started UNet from {p.name} "
              f"(upsampler={src.get('upsampler')!r}, freeze_sr="
              f"{src.get('freeze_sr')}, epoch={ck.get('epoch')})")

    # ------------------------------------------------------------ unit guards
    def on_load_checkpoint(self, checkpoint):
        """Refuse to restore a ckpt under a different raw->reflectance scale.

        `sr.cli test --config ... --ckpt_path` instantiates from the CONFIG
        and only restores weights — without this guard a pre-fix (starved,
        scale 10000.0) ckpt evaluated under a `reflectance_scale: 1.0` config
        (or vice versa) silently scores garbage."""
        ck = float(checkpoint.get("hyper_parameters", {})
                             .get("reflectance_scale", 10000.0))
        if ck != float(self.hparams.reflectance_scale):
            raise ValueError(
                f"reflectance_scale mismatch: checkpoint trained with {ck}, "
                f"instance configured {self.hparams.reflectance_scale}. Align "
                "the test/viz config with the checkpoint's training scale."
            )
        # Same hole, same fix, for the hard-constraint treatment: the strict
        # state-dict load would also catch a mismatch (the hard_constraint.*
        # keys appear or vanish), but with an error that reads as corruption
        # rather than as a wrong config. Compare RESOLVED values — 'native'
        # means different things for sen2sr and sr4rs, and old checkpoints
        # carry no key at all.
        ck_hp = checkpoint.get("hyper_parameters", {})
        ck_hc = resolve_sr_hc(ck_hp.get("upsampler", self.hparams.upsampler),
                              ck_hp.get("sr_hc", "native"))
        if ck_hc != self._sr_hc_on:
            raise ValueError(
                f"sr_hc mismatch: checkpoint trained with the FFT hard "
                f"constraint {'ON' if ck_hc else 'OFF'} "
                f"(upsampler={ck_hp.get('upsampler')!r}, "
                f"sr_hc={ck_hp.get('sr_hc', 'native')!r}), instance resolves to "
                f"{'ON' if self._sr_hc_on else 'OFF'} "
                f"(upsampler={self.hparams.upsampler!r}, "
                f"sr_hc={getattr(self.hparams, 'sr_hc', 'native')!r}). Align the "
                "test/viz config with the checkpoint's arm."
            )
        # §4.3 provenance: continuing training from an adaptive-norm ckpt under
        # adaptive_norm=false would silently FREEZE the statistics part-way
        # along their trajectory — the UNet would keep co-adapting while the
        # adapter stopped tracking. Restores for test/viz are fine either way
        # (the buffers carry the adapted values), so only the fit path raises.
        ck_adaptive = bool(checkpoint.get("hyper_parameters", {})
                                     .get("adaptive_norm", False))
        if ck_adaptive and not bool(getattr(self.hparams, "adaptive_norm", False)):
            fn = getattr(getattr(self, "_trainer", None), "state", None)
            fitting = str(getattr(fn, "fn", "")).lower().find("fit") >= 0
            msg = ("checkpoint was trained with adaptive_norm=true but this "
                   "instance has adaptive_norm=false")
            if fitting:
                raise ValueError(
                    f"{msg} — continuing the fit would freeze the running "
                    "statistics mid-trajectory. Pass --model.adaptive_norm "
                    "true to resume, or start a new run.")
            print(f"[joint_sr] NOTE: {msg}; the restored buffers carry the "
                  "adapted values, which is correct for test/viz.")

    def on_train_batch_start(self, batch, batch_idx):
        """One-time unit tripwire (the §15 lesson: print your units)."""
        if self.global_step == 0 and batch_idx == 0:
            m = float((batch[0] / self.hparams.reflectance_scale).mean())
            print(f"[joint_sr] reflectance_scale={self.hparams.reflectance_scale} "
                  f"-> SR input mean {m:.4g} (sane reflectance: ~0.05-0.35)")
            if not (1e-3 < m < 2.0):
                raise ValueError(
                    f"SR input mean {m:.3g} is outside any sane reflectance "
                    "range — reflectance_scale is wrong for this dataset."
                )

    # ------------------------------------------------------------- forward
    def _sr_forward(self, x):
        """x: (B, C, P, P) raw -> SR output rescaled back to raw units, fp32.

        Everything the segmentation network's input passes through EXCEPT the
        z-score. Factored out so the recalibration pass (§4.4) and the
        functional drift monitor (§4.3) run exactly the front-end the training
        forward runs, with no risk of the two drifting apart.
        """
        if self._required_lr is not None and x.shape[-1] != self._required_lr:
            raise ValueError(
                f"SEN2SR's shipped FFT mask pins the LR patch to "
                f"{self._required_lr}px, got {x.shape[-1]} — set data.crop_size "
                f"accordingly."
            )
        x = x / self.hparams.reflectance_scale         # raw -> 0-1 reflectance
        # SEN2SR's hard constraint uses torch.fft, which has no BFloat16
        # kernels -> "Unsupported dtype BFloat16" under bf16-mixed autocast.
        # Run the (small, 572K-param) SR stage in fp32 with autocast disabled;
        # the UNet below still runs under the Trainer's mixed precision.
        # Gradients flow through the dtype casts unchanged.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            p = self.hparams.sr_pad
            if p:
                x32 = torch.nn.functional.pad(x32, (p, p, p, p), mode="reflect")
            if self.hparams.freeze_sr:
                with torch.no_grad():
                    hr = self.sr(x32)
            else:
                hr = self.sr(x32)
            if p:
                q = p * self.hparams.upscale
                hr = hr[..., q:-q, q:-q]
            return hr * self.hparams.reflectance_scale

    def forward(self, x):
        """x: (B, C, P, P) raw DN at 10 m -> logits (B, classes, sP, sP)."""
        y = self._sr_forward(x)
        with torch.autocast(device_type=y.device.type, enabled=False):
            if self._adapt_enabled():
                # Update-then-use (immaterial at m=0.01; documented so the
                # convention is not re-litigated). Stop-grad throughout, so the
                # z-score is still a constant affine for this step and the
                # gradient into `hr` is still exactly rs/band_std.
                self._adapt_update(y)
            x_seg = (y - self.band_mean) / self.band_std
            # getattr, not attribute access: checkpoints written before `head`
            # existed have no such hparam, and every already-benched R-arm is
            # one of those.
            if getattr(self.hparams, "head", "unet") == "linear":
                # §6.3, amended: the plan assumed the head already sat inside
                # this fp32 island for the SR arms and that only rl0 (no SR net)
                # would fall out of it. In fact the island ENDS here for every
                # arm — `self.model(...)` has always run under the Trainer's
                # autocast — so a bf16 probe would have been the case across the
                # whole series, not just the anchor.
                #
                # It has to be fp32 for the probe specifically: bf16 carries 8
                # mantissa bits, and a 5-parameter linear projection produces a
                # smooth unsaturated logit distribution with substantial mass
                # near the boundary. At 8 bits that distribution quantises into
                # heavy ties, and AP — the selection criterion for this whole
                # series — is a RANKING statistic, so ties degrade it.
                #
                # Keyed on the head, not on the presence of an SR stage, so the
                # precision policy is constant across all five rl arms. Costs
                # nothing: it is a 1x1 convolution.
                return self.model(x_seg)
        return self.model(x_seg)

    # ------------------------------------------- per-group gradient clipping
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        """Clip the SR parameter group only, at ``clip_sr`` (§6.2).

        ``clip_sr == 0`` (every R-arm) falls straight through to Lightning's
        global behaviour, so nothing about the U-Net path changes.
        """
        clip_sr = float(getattr(self.hparams, "clip_sr", 0.0) or 0.0)
        if clip_sr <= 0:
            return super().configure_gradient_clipping(
                optimizer, gradient_clip_val, gradient_clip_algorithm)
        if gradient_clip_val:
            # Both would apply, and the global one is precisely the treatment-
            # dependent norm this override exists to avoid.
            raise ValueError(
                f"clip_sr={clip_sr} is set for per-group clipping, but the "
                f"Trainer also passes gradient_clip_val={gradient_clip_val}. "
                "Set trainer.gradient_clip_val to 0 — _stages_tv.sh does this "
                "automatically for HEAD=linear."
            )
        if gradient_clip_algorithm not in (None, "norm"):
            raise ValueError(
                f"clip_sr implements NORM clipping; got "
                f"gradient_clip_algorithm={gradient_clip_algorithm!r}.")
        sr_params = self._sr_trainable_params()
        if sr_params:
            torch.nn.utils.clip_grad_norm_(sr_params, clip_sr)
        # The head group is deliberately NOT clipped — in the frozen arms it is
        # the only group, so clipping it there and not here (or vice versa)
        # would make the treatment and the clipping co-vary.

    # ------------------------------------------------ linear-probe bias init
    def setup(self, stage: str | None = None):
        """Measure the road base rate and set the probe's bias to its logit.

        Runs in ``setup``, NOT ``on_fit_start``, and the ordering is the point:
        Lightning calls ``setup`` BEFORE restoring a checkpoint, so a ``--ckpt_path``
        resume has this initial value overwritten by the trained one rather than
        clobbering it.

        That ordering does NOT cover ``warm_start_head``, which loads in
        ``__init__`` — earlier than ``setup``, so the argument runs the other
        way and an explicit guard is needed. See the second check below.

        A no-op for every U-Net arm.
        """
        super().setup(stage)
        if getattr(self.hparams, "head", "unet") != "linear" or stage not in (None, "fit"):
            return
        if getattr(self, "_head_warm_started", False):
            # LP-FT: _load_head_from ran in __init__ and restored BOTH the twin's
            # weights and its bias. __init__ is earlier than setup, so the
            # "setup runs before checkpoint restore" ordering that protects a
            # --ckpt_path resume runs the WRONG WAY here — without this guard the
            # bias init would silently overwrite the converged bias and rl2/rl4
            # would no longer be warm-started in the sense §2 requires.
            b = float(self.model.conv.bias.detach().flatten()[0])
            print(f"[joint_sr] linear probe bias = {b:.4f} (warm-started; "
                  "base-rate init skipped)")
            return
        if self.hparams.head_bias_prior is not None:
            b = self.model.set_bias_prior(self.hparams.head_bias_prior)
            print(f"[joint_sr] linear probe bias = logit("
                  f"{self.hparams.head_bias_prior:.6g}) = {b:.4f} (hparam)")
            return
        prior = self._measure_road_base_rate()
        if prior is None:
            raise RuntimeError(
                "head='linear' needs a road base rate for the bias init and the "
                "training masks could not be streamed. Pass head_bias_prior "
                "explicitly (--model.head_bias_prior)."
            )
        # Record it so the checkpoint carries the value that was actually used
        # and a restore never re-measures against a different split.
        self.hparams.head_bias_prior = prior
        b = self.model.set_bias_prior(prior)
        print(f"[joint_sr] linear probe bias = logit({prior:.6g}) = {b:.4f} "
              f"(measured from the training masks)")

    def _measure_road_base_rate(self) -> float | None:
        """Positive-pixel fraction of the training masks, cached per dataset.

        Bounded by ``head_bias_batches``: this is a bias initialiser, not a
        statistic anyone reports, and the estimate is already far more precise
        than the initialisation needs long before the loader is exhausted.
        """
        import hashlib
        import json
        import os
        from pathlib import Path

        try:
            dm = getattr(self.trainer, "datamodule", None)
        except RuntimeError:          # not attached to a Trainer
            dm = None
        if dm is None:
            return None
        # _preserve_rng, like every other auxiliary loader in this file: building
        # a DataLoader iterator draws _base_seed from the global generator, which
        # shifts the crop stream the run then trains on. Here that would have
        # been particularly nasty — the cache check below returns BEFORE the
        # loop, so a cache miss would consume RNG and a cache hit would not, and
        # the training crop sequence would depend on whether a JSON file happened
        # to exist on disk.
        try:
            with _preserve_rng():
                loader = dm.train_dataloader()
        except Exception as exc:                       # noqa: BLE001
            print(f"[joint_sr] could not build the train loader for the base-rate "
                  f"measurement: {exc}")
            return None

        # Cache beside the dataset's norm stats, keyed on everything that
        # changes the answer, so two arms of the same series reuse one pass and
        # a different split/label source can never silently reuse it.
        cache_path = None
        ds_dir = getattr(dm, "dataset_dir", None)
        if ds_dir:
            key = json.dumps({
                "splits": sorted(getattr(dm, "train_splits", ["train"]) or ["train"]),
                "mask_source": getattr(dm, "mask_source", None),
                "mask_dirname": getattr(dm, "mask_dirname", None),
                "upscale": int(self.hparams.upscale),
                "max_batches": int(self.hparams.head_bias_batches),
            }, sort_keys=True)
            digest = hashlib.sha1(key.encode()).hexdigest()[:12]
            cache_path = Path(ds_dir) / f"road_base_rate_{digest}.json"
            if cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text())
                    print(f"[joint_sr] road base rate {cached['base_rate']:.6g} "
                          f"(cached: {cache_path.name}, {cached['n_batches']} batches)")
                    return float(cached["base_rate"])
                except Exception:                      # noqa: BLE001
                    print(f"[joint_sr] ignoring unreadable cache {cache_path}")

        pos = 0.0
        tot = 0.0
        n = 0
        limit = max(1, int(self.hparams.head_bias_batches))
        with _preserve_rng():
            for batch in loader:
                masks = batch[1] if isinstance(batch, (tuple, list)) else batch["mask"]
                m = masks.detach().float()
                pos += float(m.sum())
                tot += float(m.numel())
                n += 1
                if n >= limit:
                    break
        if tot <= 0:
            return None
        rate = pos / tot
        print(f"[joint_sr] road base rate {rate:.6g} over {n} batches "
              f"({int(tot):,} px)")
        if cache_path is not None:
            try:
                tmp = cache_path.with_suffix(f".tmp.{os.getpid()}")
                tmp.write_text(json.dumps(
                    {"base_rate": rate, "n_batches": n, "n_pixels": int(tot)},
                    indent=2))
                tmp.replace(cache_path)                 # atomic publish
            except Exception as exc:                    # noqa: BLE001
                print(f"[joint_sr] could not cache the base rate ({exc}); "
                      "it will be re-measured next fit.")
        return rate

    # ------------------------------------- adaptive post-SR normalisation
    def _adapt_enabled(self) -> bool:
        return (self.training
                and bool(getattr(self.hparams, "adaptive_norm", False))
                and self.global_step >= int(
                    getattr(self.hparams, "adaptive_norm_warmup_steps", 0) or 0))

    @torch.no_grad()
    def _adapt_update(self, y):
        """EMA-update ``band_mean``/``band_std`` toward this batch's per-band
        moments of the PRE-normalisation SR output ``y``.

        Reconstructs E[x^2] from the live buffers (``std^2 + mean^2``) rather
        than carrying a separate persistent buffer — exact, and it keeps the
        checkpoint tensor set unchanged. Deliberately free of host syncs: the
        finite-check is folded into a ``torch.where`` and the skip counter
        lives on-device, so the hot path never stalls the GPU.
        """
        m = float(self.hparams.adaptive_norm_momentum)
        bm = y.mean(dim=(0, 2, 3)).float() # set of axes reduced over: batch, width height, skipping channels
        bm2 = y.pow(2).mean(dim=(0, 2, 3)).float()

        # DDP: per-rank EMAs would diverge silently, and the buffers are not
        # gradient-synced. Average the batch moments across ranks first.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stacked = torch.stack((bm, bm2))
            torch.distributed.all_reduce(stacked, op=torch.distributed.ReduceOp.SUM)
            stacked /= float(torch.distributed.get_world_size())
            bm, bm2 = stacked[0], stacked[1]

        ok = torch.isfinite(bm).all() & torch.isfinite(bm2).all()
        self._adapt_skips += (~ok).to(self._adapt_skips.dtype) # skip adaptation if nonsense

        mean = self.band_mean.reshape(-1)
        std = self.band_std.reshape(-1)
        m2 = std * std + mean * mean #(recall E(X^2) = Var(X) - E(X)^2)

        new_mean = torch.where(ok, (1.0 - m) * mean + m * bm, mean)
        new_m2 = torch.where(ok, (1.0 - m) * m2 + m * bm2, m2)
        raw_var = new_m2 - new_mean * new_mean
        # Select the ORIGINAL std on the skip path rather than recomputing it
        # from the round-tripped m2: `std^2 + mean^2 - mean^2` is not bit-exact
        # in fp32, so recomputing would nudge band_std by an ulp on a batch we
        # explicitly decided to ignore. A skipped batch must be a true no-op.
        new_std = torch.where(ok, raw_var.clamp_min(_VAR_FLOOR).sqrt(), std)
        self.band_mean.copy_(new_mean.reshape(self.band_mean.shape))
        self.band_std.copy_(new_std.reshape(self.band_std.shape))

        # Same structure for the (diagnostic-only) fast EMAs. Note fm_prev:
        # the second moment must be reconstructed from the mean BEFORE this
        # step's update, not after it.
        fm, fs = self._adapt_fast_mean, self._adapt_fast_std
        f = _ADAPT_FAST_M
        fm_prev = fm.clone()
        fm.copy_(torch.where(ok, (1.0 - f) * fm_prev + f * bm, fm_prev))
        fast_m2 = (1.0 - f) * (fs * fs + fm_prev * fm_prev) + f * bm2
        fs.copy_(torch.where(ok, (fast_m2 - fm * fm).clamp_min(_VAR_FLOOR).sqrt(), fs))

        every = int(getattr(self.hparams, "adaptive_norm_check_every", 50) or 0)
        if every > 0 and self.global_step % every == 0:
            # The floor is the ONE path that breaks the m2 = std^2 + mean^2
            # identity the state relies on, so it must never pass silently.
            # Checked BEFORE the band guard: a collapsed band trips the band
            # guard too, and "this band went constant" is the more specific
            # diagnosis — it must reach the log before the RuntimeError.
            # Folded into the same host sync rather than costing its own.
            if not self._adapt_floor_warned and bool((ok & (raw_var < _VAR_FLOOR)).any()):
                self._adapt_floor_warned = True
                print("[joint_sr] WARN adaptive_norm: variance floor engaged "
                      f"(raw var {[float(v) for v in raw_var]}) — the running "
                      "E[x^2] reconstruction is no longer exact from here. A "
                      "band collapsed to a constant; check the SR output.")
            self._check_std_band()

    def _check_std_band(self):
        """Asymmetric std band (§4.3). Warn once, then RAISE — never clamp.

        A silent floor would be a trapdoor: by the time e.g. a 1e-2 x init_std
        clamp engaged, the gradient gain into the generator would already be
        100x its initial 15-33x, i.e. the run entered the failure mode long
        before. Fail loud and early instead.

        Collapse and growth get different bounds because they carry different
        risk (module constants). Raises :class:`AdaptiveNormBandExit` so a tune
        can prune the trial rather than lose the study.
        """
        lo_r = float(getattr(self.hparams, "std_band_raise_lo", _STD_RAISE_LO))
        hi_r = float(getattr(self.hparams, "std_band_raise_hi", _STD_RAISE_HI))
        lo_w = float(getattr(self.hparams, "std_band_warn_lo", _STD_WARN_LO))
        hi_w = float(getattr(self.hparams, "std_band_warn_hi", _STD_WARN_HI))
        init = self._adapt_init_std.reshape(-1)
        ratio = (self.band_std.reshape(-1) / init.clamp_min(1e-12)).float().cpu()
        r = [float(v) for v in ratio]
        collapsed = [i for i, v in enumerate(r) if v < lo_r]
        grew = [i for i, v in enumerate(r) if v > hi_r]
        if collapsed or grew:
            if collapsed:
                what = (f"band(s) {collapsed} COLLAPSED below {lo_r}x the "
                        "starting std. The adapter's gradient gain into the "
                        "generator is reflectance_scale/band_std, so this is "
                        "positive feedback: lower std -> higher gain -> faster "
                        "drift. This is the runaway the feature exists to stop.")
            else:
                what = (f"band(s) {grew} GREW past {hi_r}x the starting std. "
                        "Growth is self-stabilising (rising std lowers the "
                        "gain), so passing this bound means the scale moved "
                        "very far, not merely upward.")
            raise AdaptiveNormBandExit(
                f"adaptive_norm: {what}\n"
                f"  step                  : {self.global_step}\n"
                f"  std/init_std per band : {['%.3f' % v for v in r]}\n"
                f"  band                  : [{lo_r}x, {hi_r}x]\n"
                f"  band_std              : {[float(v) for v in self.band_std.reshape(-1)]}\n"
                f"  init  std             : {[float(v) for v in init]}\n"
                f"  band_mean             : {[float(v) for v in self.band_mean.reshape(-1)]}\n"
                "First thing to check is lr_sr: under Adam the per-step weight "
                "displacement is ~lr regardless of gradient scale, so the rate "
                "at which the pretrained SR is destroyed is set by the ABSOLUTE "
                "lr_sr. Then: lower adaptive_norm_momentum (a slower EMA lags "
                "more, and that lag IS the restoring force), or enable L2-SP. "
                "Read sr_psnr_vs_init alongside this — moments cannot tell you "
                "whether the SR is still an SR.",
                ratios=r, band=(lo_r, hi_r))
        if not self._adapt_warned and any(v < lo_w or v > hi_w for v in r):
            self._adapt_warned = True
            side = "below" if any(v < lo_w for v in r) else "above"
            print(f"[joint_sr] WARN adaptive_norm: band std at "
                  f"{['%.3f' % v for v in r]}x the starting std (step "
                  f"{self.global_step}) — {side} the warn band "
                  f"[{lo_w}, {hi_w}]. Not fatal; watch sr_psnr_vs_init to see "
                  "whether the SR output is still imagery.")

    def _log_adapt_stats(self):
        """Per-band moments + the three diagnostics the plan asks for."""
        if not bool(getattr(self.hparams, "adaptive_norm", False)):
            return
        mean = self.band_mean.reshape(-1)
        std = self.band_std.reshape(-1)
        init_mean = self._adapt_init_mean.reshape(-1)
        init_std = self._adapt_init_std.reshape(-1)
        for i in range(mean.numel()):
            self.log(f"adapt_mean_b{i}", mean[i], on_epoch=True, sync_dist=False)
            self.log(f"adapt_std_b{i}", std[i], on_epoch=True, sync_dist=False)
        drift = (mean - init_mean).abs().max()
        ratio = std / init_std.clamp_min(1e-12)
        # The slow-EMA vs fast-EMA gap: the lag term. Reported for BOTH moments
        # — the std lag is what the hard band and the rs/band_std gradient-gain
        # argument actually turn on.
        lag = (mean - self._adapt_fast_mean).abs().max()
        lag_std = (std - self._adapt_fast_std).abs().max()
        self.log("adapt_mean_drift_max", drift, on_epoch=True, sync_dist=False)
        self.log("adapt_std_ratio_max", ratio.max(), on_epoch=True, sync_dist=False)
        self.log("adapt_std_ratio_min", ratio.min(), on_epoch=True, sync_dist=False)
        self.log("adapt_lag_max", lag, on_epoch=True, sync_dist=False)
        self.log("adapt_lag_std_max", lag_std, on_epoch=True, sync_dist=False)
        self.log("adapt_skipped_batches", self._adapt_skips, on_epoch=True,
                 sync_dist=False)
        if not self._adapt_drift_warned and float(drift) > _MEAN_DRIFT_WARN:
            self._adapt_drift_warned = True
            print(f"[joint_sr] WARN adaptive_norm: max per-band mean drift "
                  f"{float(drift):.4f} exceeds {_MEAN_DRIFT_WARN} (the "
                  "sanity_viz 'LARGE — investigate!' threshold). The adapter "
                  "is tracking it, but check sr_psnr_vs_init: moments alone "
                  "cannot see ringing or texture hallucination.")
        # Same gate as the in-step check: `check_every=0` means the guard is
        # OFF, not "moved to the val loop", where its traceback would be far
        # less legible.
        if int(getattr(self.hparams, "adaptive_norm_check_every", 50) or 0) > 0:
            self._check_std_band()

    # --------------------------------------------- §4.4 PreciseBN recompute
    def _resolve_recalibrate(self) -> str:
        """`norm_recalibrate` with 'auto' resolved against this arm."""
        mode = str(getattr(self.hparams, "norm_recalibrate", "off"))
        if mode != "auto":
            return mode
        if self.hparams.upsampler == "bicubic":
            return "off"          # R0 stays the untouched deterministic baseline
        if not self._n_sr_p0:
            # Frozen SR (r1/r5): the output distribution is stationary, so one
            # exact recompute BEFORE fitting is the complete fix — the adapter
            # is otherwise stale from step 0.
            return "pre"
        return "post" if bool(getattr(self.hparams, "adaptive_norm", False)) else "off"

    @torch.no_grad()
    def recalibrate_norm_stats(self, dataloader=None, n_batches=None,
                               tag: str = "") -> dict:
        """Exact per-band population mean/var of the SR output; overwrite the
        buffers (Wu & Johnson 2021's PreciseBN, applied to the post-SR adapter).

        Runs the SR front-end ONLY — no UNet, no backward — over `n_batches`
        training batches. Accumulates in float64: at 4 x 512 x 512 px per band
        per batch, 200 batches is ~2e8 samples, where fp32 accumulation of a
        sum of squares is not trustworthy.
        """
        if n_batches is None:
            n_batches = int(self.hparams.norm_recalibrate_batches)
        was_training = self.training
        self.eval()
        C = self.band_mean.numel()
        dev = self.band_mean.device
        s1 = torch.zeros(C, dtype=torch.float64, device=dev)
        s2 = torch.zeros(C, dtype=torch.float64, device=dev)
        n = 0
        seen = 0
        with _preserve_rng():
            if dataloader is None:
                dm = getattr(getattr(self, "_trainer", None), "datamodule", None)
                if dm is None:
                    raise RuntimeError("recalibrate_norm_stats needs a "
                                       "dataloader (no datamodule attached)")
                dataloader = dm.train_dataloader()
            for batch in dataloader:
                if seen >= n_batches:
                    break
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev)
                y = self._sr_forward(x).double()
                s1 += y.sum(dim=(0, 2, 3))
                s2 += y.pow(2).sum(dim=(0, 2, 3))
                n += y.shape[0] * y.shape[2] * y.shape[3]
                seen += 1
        if was_training:
            self.train()
        if n == 0:
            raise RuntimeError("recalibrate_norm_stats: the dataloader yielded "
                               "no batches")

        # Distributed: this walks the DATAMODULE's loader, not the
        # DistributedSampler-wrapped one the fit loop builds, and the JointSR
        # eval/train loaders are shuffle=False — so every rank sees the SAME
        # batches and computes the same statistic. Summing them would just be
        # world_size copies of one rank's answer dressed up as a population
        # statistic. Broadcast rank 0's instead: cheaper, and honest about what
        # the number is (one rank's `n_batches`, agreed across ranks).
        new_mean = (s1 / n)
        new_var = (s2 / n - new_mean * new_mean).clamp_min(_VAR_FLOOR)
        old_mean = self.band_mean.reshape(-1).clone()
        old_std = self.band_std.reshape(-1).clone()
        self.band_mean.copy_(new_mean.to(self.band_mean.dtype)
                             .reshape(self.band_mean.shape))
        self.band_std.copy_(new_var.sqrt().to(self.band_std.dtype)
                            .reshape(self.band_std.shape))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(self.band_mean, src=0)
            torch.distributed.broadcast(self.band_std, src=0)
        # The fast EMAs are diagnostics against the slow one; re-seed them so
        # the lag reading after a recalibration is not a spurious step change.
        self._adapt_fast_mean.copy_(self.band_mean.reshape(-1))
        self._adapt_fast_std.copy_(self.band_std.reshape(-1))

        d_mean = float((self.band_mean.reshape(-1) - old_mean).abs().max())
        d_std = float((self.band_std.reshape(-1) / old_std.clamp_min(1e-12)
                       - 1.0).abs().max())
        label = f"[{tag}] " if tag else ""
        print(f"[joint_sr] {label}recalibrated post-SR norm stats over "
              f"{seen} batches ({n:.3g} px/band):")
        print(f"    mean {[round(float(v), 5) for v in old_mean]} -> "
              f"{[round(float(v), 5) for v in self.band_mean.reshape(-1)]}")
        print(f"    std  {[round(float(v), 5) for v in old_std]} -> "
              f"{[round(float(v), 5) for v in self.band_std.reshape(-1)]}")
        print(f"    max |Δmean| = {d_mean:.4g}   max |Δstd|/std = {d_std:.4g}")
        return {"batches": seen, "pixels_per_band": float(n),
                "d_mean_max": d_mean, "d_std_rel_max": d_std}

    @torch.no_grad()
    def _val_iou_snapshot(self, max_batches: int = _VAL_SNAPSHOT_BATCHES):
        """IoU over a bounded SUBSAMPLE of the val loader, for the before/after
        comparison across a recalibration swap.

        Deliberately not the run's ``val_iou``: it is capped at `max_batches`
        and computed on a fresh, un-DDP-synced metric. Callers must name it as
        a subsample (``recal_valsub_iou_*``) — the whole point of the number is
        to flag EMA lag before the test numbers are read, and a figure that
        looks like the headline val_iou but is not would defeat that.

        Returns None when the run has no val loop (the train+val refit): val is
        inside the training set there, so a number from it would be a train-set
        number wearing a val label.
        """
        if self._val_loop_disabled():
            return None
        dm = getattr(getattr(self, "_trainer", None), "datamodule", None)
        if dm is None:
            return None
        from torchmetrics.classification import BinaryJaccardIndex
        metric = BinaryJaccardIndex().to(self.band_mean.device)
        was_training = self.training
        self.eval()
        with _preserve_rng():
            for i, batch in enumerate(dm.val_dataloader()):
                if i >= max_batches:
                    break
                images, masks = batch[0].to(self.device), batch[1].to(self.device)
                preds = torch.sigmoid(self(images)) > self.hparams.threshold
                metric.update(preds, (masks > 0.5).long())
        if was_training:
            self.train()
        return float(metric.compute())

    # ------------------------------- §4.3 functional (image-space) monitor
    def _functional_monitor_on(self) -> bool:
        return (bool(getattr(self.hparams, "sr_functional_monitor", False))
                and bool(self._n_sr_p0))     # nothing evolves in a frozen SR

    @torch.no_grad()
    def _capture_sr_reference(self):
        """Cache the INITIAL SR net's output on one fixed batch."""
        dm = getattr(getattr(self, "_trainer", None), "datamodule", None)
        if dm is None:
            return
        with _preserve_rng():
            try:
                loader = (dm.train_dataloader() if self._val_loop_disabled()
                          else dm.val_dataloader())
                batch = next(iter(loader))
            except Exception as exc:                   # noqa: BLE001
                print(f"[joint_sr] WARN: functional drift monitor disabled — "
                      f"could not fetch a reference batch ({exc})")
                return
        k = max(1, int(self.hparams.sr_monitor_samples))
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch)[:k]
        was_training = self.training
        self.eval()
        ref = self._sr_forward(x.to(self.device))
        if was_training:
            self.train()
        # Held on CPU: ~2 x 4 x 512 x 512 fp32 = 8 MB, and one 8 MB transfer
        # per val epoch is nothing next to keeping it off a busy GPU.
        self._srmon_x = x.detach().cpu().clone()
        self._srmon_ref = ref.detach().cpu().clone()
        print(f"[joint_sr] functional drift monitor armed on {k} fixed "
              f"{'train' if self._val_loop_disabled() else 'val'} sample(s) "
              f"({tuple(self._srmon_ref.shape)})")

    @torch.no_grad()
    def _log_functional_drift(self):
        """PSNR / RMSE of the current SR output against the cached initial one.

        Both tensors are SR MODEL OUTPUTS on identical inputs — there is no
        ground truth anywhere in this computation, so these are drift
        distances, not quality metrics. `data_range` is one reflectance unit
        expressed in the model's working units.
        """
        if self._srmon_ref is None:
            return
        was_training = self.training
        self.eval()
        cur = self._sr_forward(self._srmon_x.to(self.device))
        if was_training:
            self.train()
        ref = self._srmon_ref.to(cur.device, cur.dtype)
        mse = (cur - ref).pow(2).mean()
        rmse = mse.sqrt()
        data_range = float(self.hparams.reflectance_scale)
        psnr = 10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-20))
        self.log("sr_psnr_vs_init", psnr, on_epoch=True, sync_dist=False)
        self.log("sr_rmse_vs_init", rmse, on_epoch=True, sync_dist=False)

    # ---------------------------------------------- drift monitor + L2-SP
    def _sr_trainable_params(self):
        """Trainable SR params, in the SAME order the theta_0 buffers were
        snapshot in (requires_grad never changes after __init__)."""
        return [p for p in self.sr.parameters() if p.requires_grad]

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        lam = float(getattr(self.hparams, "l2sp_lambda", 0.0))
        if lam > 0.0 and self._n_sr_p0:
            pen = None
            for i, p in enumerate(self._sr_trainable_params()):
                d = (p - getattr(self, f"_sr_p0_{i}")).pow(2).sum()
                pen = d if pen is None else pen + d
            self.log("l2sp_penalty", pen, on_step=False, on_epoch=True,
                     sync_dist=True)
            loss = loss + lam * pen
        return loss

    def _sr_drift(self):
        """Relative L2 drift ||theta - theta_0|| / ||theta_0||, or None."""
        if not self._n_sr_p0:
            return None
        with torch.no_grad():
            num = None
            den = None
            for i, p in enumerate(self._sr_trainable_params()):
                p0 = getattr(self, f"_sr_p0_{i}")
                n = (p - p0).pow(2).sum()
                d = p0.pow(2).sum()
                num = n if num is None else num + n
                den = d if den is None else den + d
            return (num / den.clamp_min(1e-24)).sqrt()

    def _log_sr_drift(self):
        drift = self._sr_drift()
        if drift is not None:
            self.log("sr_drift_rel", drift, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        # The parent hook logs val_theta_star / val_iou_at_theta_star —
        # without this super() call it would be silently shadowed here.
        super().on_validation_epoch_end()
        self._log_sr_drift()
        self._log_adapt_stats()
        self._log_functional_drift()

    def _val_loop_disabled(self):
        """True when the trainer runs no val loop (final train+val refit).

        ``joint_sr_trainval.yaml`` sets ``limit_val_batches: 0`` because val is
        folded into train, which would otherwise silently kill the drift
        monitor — the one signal telling us whether the task loss is quietly
        destroying the pretrained SR weights."""
        t = getattr(self, "_trainer", None)
        return t is not None and not float(getattr(t, "limit_val_batches", 1) or 0)

    # ------------------------------------------------ SR evolution snapshots
    def _snapshot_sr(self, tag: str | None = None):
        if not self._n_sr_p0 or not self.trainer.is_global_zero:
            return
        from pathlib import Path
        d = Path(self.hparams.sr_snapshot_dir)
        d.mkdir(parents=True, exist_ok=True)
        drift = self._sr_drift()
        name = (f"epoch_{self.current_epoch:03d}"
                + (f"_{tag}" if tag else "") + ".pt")
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "upsampler": self.hparams.upsampler,
                # RESOLVED, not the tri-state: the replay path (sr.viz_sr) has
                # to rebuild a module whose state_dict keys match these, and
                # 'native' alone would not tell it which.
                "sr_hc": "on" if self._sr_hc_on else "off",
                "lr_sr": self.hparams.lr_sr,
                "sr_drift_rel": float(drift) if drift is not None else None,
                # Reload: build the SR net via the matching load_trainable_*
                # helper (or a JointSRUNetLightning), then
                # model.sr.load_state_dict(snapshot["sr_state_dict"]).
                "sr_state_dict": {k: v.detach().cpu()
                                  for k, v in self.sr.state_dict().items()},
            },
            d / name,
        )

    # ------------------------------------------------------- fit lifecycle
    def on_fit_start(self):
        """Pre-fit recalibration (§2.1 Phase 0.1) then arm the drift monitor.

        Order matters: recalibrating first means the cached SR reference and
        the statistics both describe the same starting point.
        """
        if bool(getattr(self.hparams, "adaptive_norm", False)):
            # The EMA is an in-place update on the buffers. Under a `*-true`
            # precision the strategy casts module buffers, and at bf16 (8
            # mantissa bits) `(1-m)*mean + m*batch` with m=0.01 rounds away
            # entirely — the EMA would silently stop moving while every guard
            # still read "healthy". Only `-mixed` precisions are configured
            # today, so this is a tripwire, not a live bug.
            if self.band_mean.dtype != torch.float32:
                raise RuntimeError(
                    f"adaptive_norm needs fp32 normalisation buffers, got "
                    f"{self.band_mean.dtype}. A '-true' precision casts them; "
                    "use a '-mixed' precision (the SR stage already runs in an "
                    "fp32 autocast-disabled island for the same reason).")
        if self._resolve_recalibrate() == "pre":
            self.recalibrate_norm_stats(tag="pre-fit")
            # The guard band measures movement from where this run STARTS.
            self._adapt_init_mean.copy_(self.band_mean)
            self._adapt_init_std.copy_(self.band_std)
        if self._functional_monitor_on():
            self._capture_sr_reference()
        if bool(getattr(self.hparams, "adaptive_norm", False)):
            print(f"[joint_sr] adaptive_norm ON  (m="
                  f"{self.hparams.adaptive_norm_momentum}, warmup_steps="
                  f"{self.hparams.adaptive_norm_warmup_steps}, check_every="
                  f"{self.hparams.adaptive_norm_check_every}); "
                  f"norm_recalibrate={self.hparams.norm_recalibrate}"
                  f" -> {self._resolve_recalibrate()}")

    def on_train_end(self):
        """Post-fit recalibration, then REWRITE the checkpoints it must reach.

        Placement is the whole difficulty here. ModelCheckpoint does not write
        in ``on_train_end``: with ``save_on_train_epoch_end`` it writes from
        ``on_train_epoch_end`` of the last epoch, and when monitored it writes
        from ``on_validation_end`` — both strictly earlier than any end-of-fit
        hook, and earlier than the module's own ``on_train_epoch_end`` in the
        monitored case. Callback reordering does not help: it only controls the
        order *within* a hook. So recalibrating in any end-of-fit hook updates
        the in-memory buffers and nothing on disk, while printing a convincing
        log — and ``bench``/``test`` run as separate processes off the file.

        We therefore recalibrate here and explicitly re-save the checkpoints the
        end-of-training weights legitimately belong to:

          * ``last.ckpt`` — always. It IS the end-of-training state.
          * the "best" file — only when the checkpoint is UNMONITORED
            (``monitor: null``, i.e. the train+val refit's
            ``unet_s2rosa_jointsr_final.ckpt``), where "best" means "end of the
            fixed budget" and so is this same state.

        A monitored ``_best.ckpt`` belongs to an earlier epoch's weights and is
        deliberately left alone — overwriting it would staple end-of-run
        statistics onto mid-run weights. That case warns instead.
        """
        if not self._maybe_post_recalibrate():
            return
        cbs = [cb for cb in getattr(self.trainer, "checkpoint_callbacks", [])
               if getattr(cb, "dirpath", None) is not None]
        if not cbs:
            print("[joint_sr] WARN: recalibrated, but this run has no "
                  "ModelCheckpoint — nothing on disk carries the new stats.")
            return
        for cb in cbs:
            targets = []
            if getattr(cb, "last_model_path", ""):
                targets.append(cb.last_model_path)
            if getattr(cb, "monitor", None) is None and getattr(cb, "best_model_path", ""):
                targets.append(cb.best_model_path)
            elif getattr(cb, "monitor", None) is not None:
                print(f"[joint_sr] NOTE: {cb.monitor}-monitored checkpoint "
                      f"{cb.best_model_path!r} keeps ITS epoch's norm stats — "
                      "those weights are not the end-of-training weights, so "
                      "stapling end-of-run statistics onto them would be "
                      "wrong. Recalibration targets last/final only.")
            for path in dict.fromkeys(targets):
                self.trainer.save_checkpoint(path)
                print(f"[joint_sr] rewrote {path} with recalibrated stats")

    def _maybe_post_recalibrate(self) -> bool:
        """Run the `post` recalibration exactly once. Returns True if it ran."""
        if self._recal_done or self._resolve_recalibrate() != "post":
            return False
        before = self._val_iou_snapshot()
        stats = self.recalibrate_norm_stats(tag="post-fit")
        after = self._val_iou_snapshot()
        self._recal_done = True
        self._recal_stats = stats
        # `self.trainer` RAISES when unattached; `_trainer` is the safe read
        # (same pattern as _val_loop_disabled / recalibrate_norm_stats).
        logger = getattr(getattr(self, "_trainer", None), "logger", None)
        metrics = {f"recal_{k}": v for k, v in stats.items()}
        if before is not None and after is not None:
            metrics.update({"recal_valsub_iou_before": before,
                            "recal_valsub_iou_after": after,
                            "recal_valsub_iou_delta": after - before})
            print(f"[joint_sr] post-fit recalibration IoU over a "
                  f"{_VAL_SNAPSHOT_BATCHES}-batch val subsample: {before:.4f} "
                  f"-> {after:.4f} (Δ {after - before:+.4f}). A large Δ means "
                  "the EMA's lag was material — report it alongside the test "
                  "numbers, not after. (Subsample, not the run's val_iou.)")
        else:
            print("[joint_sr] post-fit recalibration: no val loop in this "
                  "protocol (train+val refit), so no metric delta is reported "
                  "— val is inside the training set here and an 'IoU' from it "
                  "would be a train-set number. Buffer deltas above.")
        if logger is not None:
            logger.log_metrics(metrics, step=int(self.global_step))
        return True

    def on_train_start(self):
        # Frame 0 of the evolution: the pretrained SR net before any task
        # gradient touches it.
        if int(getattr(self.hparams, "sr_snapshot_every", 0) or 0) > 0:
            self._snapshot_sr(tag="init")

    def on_train_epoch_end(self):
        k = int(getattr(self.hparams, "sr_snapshot_every", 0) or 0)
        if k > 0 and (self.current_epoch + 1) % k == 0:
            self._snapshot_sr()
        # No val loop -> on_validation_epoch_end never fires. Log the drift
        # here instead so the train+val refit keeps the monitor. Guarded, so
        # runs WITH a val loop still log it exactly once per epoch.
        if self._val_loop_disabled():
            self._log_sr_drift()
            self._log_adapt_stats()
            self._log_functional_drift()

    # ------------------------------------- two LR groups + cosine/warmup
    def configure_optimizers(self):
        groups = [{"params": self.model.parameters(), "lr": self.hparams.lr}]
        sr_params = self._sr_trainable_params()
        if sr_params:
            groups.append({"params": sr_params, "lr": self.hparams.lr_sr})
        opt = torch.optim.Adam(groups)
        schedule = getattr(self.hparams, "lr_schedule", "none")
        if schedule == "none":
            return opt
        if schedule != "cosine":
            raise ValueError(f"lr_schedule={schedule!r} (cosine | none)")
        import math

        # Per-step cosine to 0 over the WHOLE run (T_max == the trainer's own
        # budget, so trials and refits each see a complete schedule scaled to
        # their budget, and "lr" means "peak of a full cosine" in both).
        # Clamped at total: the lr can never oscillate back up.
        total = max(1, int(self.trainer.estimated_stepping_batches))
        steps_per_epoch = max(1, round(total / max(1, self.trainer.max_epochs)))
        # Warmup is ABSOLUTE (steps), not proportional to budget: the random-
        # decoder noise phase lasts the same number of steps however long the
        # run is (~10% of a 10-epoch trial, ~1% of a 100-epoch refit).
        warm = int(round(self._sr_warmup_epochs * steps_per_epoch))

        def cosine(step):
            t = min(step, total) / total
            return 0.5 * (1.0 + math.cos(math.pi * t))

        lambdas = [cosine]                     # UNet group: no warmup -- the
        if sr_params:                          # critic must learn full-speed
            def sr_lambda(step):
                ramp = min(1.0, step / warm) if warm > 0 else 1.0
                return ramp * cosine(step)
            lambdas.append(sr_lambda)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambdas)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
