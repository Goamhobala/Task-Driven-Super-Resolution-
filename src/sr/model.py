"""Joint SR + UNet Lightning module — the unet baseline with an SR front-end.

Subclasses :class:`unet.model.UNetLightning` so the segmentation network, the
Dice + weighted-BCE loss, the per-crop IoU/F1 metrics and the ``val_iou``
monitoring convention are IDENTICAL to the baseline; the only differences are:

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
    ):
        super().__init__(
            encoder_name=encoder_name, encoder_weights=encoder_weights,
            in_channels=in_channels, classes=classes, lr=lr,
            pos_weight=pos_weight, bands=bands, image_size=image_size,
            threshold=threshold, normalize=normalize,
            norm_mean=norm_mean, norm_std=norm_std,
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

    # ------------------------------------------------------------- forward
    def forward(self, x):
        """x: (B, C, P, P) raw DN at 10 m -> logits (B, classes, sP, sP)."""
        if self._required_lr is not None and x.shape[-1] != self._required_lr:
            raise ValueError(
                f"SEN2SR's shipped FFT mask pins the LR patch to "
                f"{self._required_lr}px, got {x.shape[-1]} — set data.crop_size "
                f"accordingly."
            )
        x = x / REFLECTANCE_SCALE                      # DN -> reflectance
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
            x_seg = (hr * REFLECTANCE_SCALE - self.band_mean) / self.band_std
        return self.model(x_seg)

    # ------------------------------------------------------ two LR groups
    def configure_optimizers(self):
        groups = [{"params": self.model.parameters(), "lr": self.hparams.lr}]
        sr_params = [p for p in self.sr.parameters() if p.requires_grad]
        if sr_params:
            groups.append({"params": sr_params, "lr": self.hparams.lr_sr})
        return torch.optim.Adam(groups)
