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

    # ------------------------------------------------------ two LR groups
    def configure_optimizers(self):
        groups = [{"params": self.model.parameters(), "lr": self.hparams.lr}]
        sr_params = [p for p in self.sr.parameters() if p.requires_grad]
        if sr_params:
            groups.append({"params": sr_params, "lr": self.hparams.lr_sr})
        return torch.optim.Adam(groups)
