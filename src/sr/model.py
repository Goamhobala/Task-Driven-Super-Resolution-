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
"""
from __future__ import annotations

import torch

from sr.sen2sr_loader import (
    SEN2SR_SCALE,
    BicubicUpsampler,
    load_trainable_sen2sr,
    load_trainable_sen2sr_full,
    pad_low_pass_mask,
)
from unet.model import UNetLightning

# DN -> surface reflectance divisor expected by SEN2SR.
REFLECTANCE_SCALE = 10000.0

# SEN2SR's required input bands (V2 layout, 1-based): [B4, B3, B2, B8].
SEN2SR_BANDS = (1, 2, 3, 4)


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
        tversky_alpha: float = 0.7,
        cl_alpha: float = 0.3,
        cl_iters: int = 5,
        sr_w: float = 1.0,
        sr_radius: int = 1,
        warmup_start: int = 30,
        warmup_ramp: int = 10,
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
            tl_ell=tl_ell, tl_theta=tl_theta, tversky_alpha=tversky_alpha,
            cl_alpha=cl_alpha, cl_iters=cl_iters, sr_w=sr_w,
            sr_radius=sr_radius, warmup_start=warmup_start,
            warmup_ramp=warmup_ramp,
        )
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
            self.sr = (load_trainable_sen2sr(sen2sr_dir) if upsampler == "sen2sr"
                       else load_trainable_sen2sr_full(sen2sr_dir))
            # The shipped FFT low-pass mask fixes the HR size -> LR patches are
            # pinned to mask_size / scale (512 / 4 = 128). Checked in forward.
            # (Computed BEFORE any pad-resize: it constrains the MODEL-facing
            # input; sr_pad grows the mask so the padded input still fits.)
            self._required_lr = self.sr.hard_constraint.low_pass_mask.shape[-1] // SEN2SR_SCALE
            if sr_pad > 0:
                # Border-artifact mitigation: the FFT constraint assumes a
                # periodic patch, so edge discontinuities ring at the borders.
                # Reflect-padding the input and cropping the output moves the
                # ring into discarded context (see forward).
                pad_low_pass_mask(self.sr, sr_pad)
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

        if warm_start_unet:
            self._load_unet_from(warm_start_unet)

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
    def forward(self, x):
        """x: (B, C, P, P) raw DN at 10 m -> logits (B, classes, sP, sP)."""
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
            x_seg = (hr * self.hparams.reflectance_scale
                     - self.band_mean) / self.band_std
        return self.model(x_seg)

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

    def on_validation_epoch_end(self):
        drift = self._sr_drift()
        if drift is not None:
            self.log("sr_drift_rel", drift, on_epoch=True, sync_dist=True)

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

    def on_train_start(self):
        # Frame 0 of the evolution: the pretrained SR net before any task
        # gradient touches it.
        if int(getattr(self.hparams, "sr_snapshot_every", 0) or 0) > 0:
            self._snapshot_sr(tag="init")

    def on_train_epoch_end(self):
        k = int(getattr(self.hparams, "sr_snapshot_every", 0) or 0)
        if k > 0 and (self.current_epoch + 1) % k == 0:
            self._snapshot_sr()

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
