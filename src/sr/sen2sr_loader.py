"""Build a *trainable* SEN2SR-Lite (RGBN ×4) as a plain nn.Module.

Why not `mlstac.load(dir).compiled_model()`: that path is inference-only — it
constructs `CNNSR(..., train_mode=False)` and sets `requires_grad=False`
everywhere. Worse, `train_mode=False` makes every `Conv3XC` block re-collapse
its conv/sk branch into a frozen `eval_conv` with *detached* weights on every
forward, so even with `requires_grad=True` no gradient can ever reach a
parameter. `mlstac.load(dir).trainable_model()` keeps `requires_grad=True` but
still passes `train_mode=False`, so it has the same dead-gradient problem.

What we do instead (verified equivalent to the compiled model, max|Δ| ~2e-6):

  * instantiate `CNNSR(4, 4, 24, 4, bias=True, train_mode=True, num_blocks=6)`
    — the exact architecture from the model card's `load.py`, but with the
    differentiable conv+sk branch active — and strictly load the shipped
    `model.safetensor` (it contains the train-branch weights, not just the
    collapsed ones);
  * rebuild the frozen FFT `HardConstraint`, re-registering its `low_pass_mask`
    as a *buffer* (upstream stores it as a plain attribute, which Lightning's
    `.to(device)` would silently leave on CPU);
  * wrap both in a module whose forward replicates upstream exactly:
    `hard_constraint(x, clamp(sr_model(x), min=0))`.

Known properties to respect downstream:

  * Input: surface reflectance (DN / 10000), float32, channel order
    [B04, B03, B02, B08] = R, G, B, NIR. Run `torch.nan_to_num` first.
  * The shipped `low_pass_mask` is a fixed 512×512 FFT mask, so the LR patch
    size is pinned to 128×128 (→ 512×512 output).
  * The hard constraint takes the DC Fourier bin entirely from the (bicubic
    upsampled) LR input, so any loss that only probes the spatial mean has
    *exactly zero* gradient w.r.t. SEN2SR parameters. Real segmentation losses
    are spatially varying and flow fine — but a `output.mean()` smoke probe
    will falsely report a dead network.
  * Upstream `CNNSR.forward` feeds `out_feature` (the stem output) to every
    SPAB block instead of chaining them, so blocks 1–4 are structurally
    disconnected — at inference and under fine-tuning alike. The weights were
    trained with this topology, so we keep it: ~240k of the 472k trainable
    parameters actually receive gradients. Do not "fix" the chaining; it would
    invalidate the pretrained weights.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# Direct-download source for the exact variant we use (RGB+NIR, ×4, Lite).
SEN2SR_RGBN_X4_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)

# The channel order SEN2SR was trained on. Our combined COGs store bands
# [B4, B3, B2, B8, ...] (see sentinel2data.dataset.bands.S2_V2_BANDS), so
# the M0 slice [0, 1, 2, 3] is ALREADY in this order — no permutation needed.
# Kept here as the single point of truth should the COG layout ever change.
SEN2SR_BAND_ORDER = ("B4", "B3", "B2", "B8")  # = R, G, B, NIR

# Architecture constants from the model card's load.py:
#   CNNSR(in=4, out=4, feature_channels=24, upscale=4, bias=True, ..., num_blocks=6)
_CNNSR_ARGS = dict(in_channels=4, out_channels=4, feature_channels=24,
                   upscale=4, bias=True, num_blocks=6)
SEN2SR_SCALE = 4


def download_sen2sr(model_dir) -> Path:
    """Fetch the SEN2SR-Lite RGBN ×4 weights via mlstac (idempotent)."""
    import mlstac

    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensor").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        mlstac.download(file=SEN2SR_RGBN_X4_URL, output_dir=str(model_dir))
    return model_dir


class TrainableSEN2SR(nn.Module):
    """SEN2SR-Lite with the differentiable branch active and a movable mask.

    Forward contract (identical to the upstream `srmodel` wrapper):
        reflectance (B, 4, H, W) -> reflectance (B, 4, 4H, 4W)
    """

    def __init__(self, sr_model: nn.Module, hard_constraint: nn.Module):
        super().__init__()
        self.sr_model = sr_model
        self.hard_constraint = hard_constraint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sr = torch.clamp(self.sr_model(x), min=0.0)
        return self.hard_constraint(x, sr)


def load_trainable_sen2sr(model_dir) -> TrainableSEN2SR:
    """Load SEN2SR-Lite RGBN ×4 from `model_dir` as a trainable nn.Module."""
    import safetensors.torch
    from sen2sr.models.opensr_baseline.cnn import CNNSR
    from sen2sr.models.tricks import HardConstraint

    model_dir = Path(model_dir)
    weights = safetensors.torch.load_file(model_dir / "model.safetensor")
    sr_model = CNNSR(**_CNNSR_ARGS, train_mode=True)
    sr_model.load_state_dict(weights, strict=True)
    # The collapsed eval_conv weights ship frozen and are unused when
    # train_mode=True; leave them requires_grad=False so param groups built
    # from `requires_grad` only carry the live branch.

    # CNNSR.forward consumes only block 0's and the last block's outputs (see
    # module docstring), so the middle SPAB blocks are structurally dead.
    # Freeze them: training is unchanged on one GPU, and DDP stops erroring on
    # parameters that never receive gradients.
    for blk in sr_model.blocks[1:-1]:
        for p in blk.parameters():
            p.requires_grad = False

    mask = safetensors.torch.load_file(model_dir / "hard_constraint.safetensor")["weights"]
    hard_constraint = HardConstraint(low_pass_mask=mask, bands="all")
    for p in hard_constraint.parameters():
        p.requires_grad = False
    # Upstream keeps low_pass_mask as a plain tensor attribute; re-register it
    # as a buffer so `.to(device)` / Lightning device placement move it too.
    del hard_constraint.low_pass_mask
    hard_constraint.register_buffer("low_pass_mask", mask)

    return TrainableSEN2SR(sr_model, hard_constraint)


class BicubicUpsampler(nn.Module):
    """Parameter-free ×`scale` bicubic upsampling — the R0 deterministic baseline.

    Same forward contract as `TrainableSEN2SR` so `JointSRSegModule` can swap
    between them via config.
    """

    def __init__(self, scale: int = SEN2SR_SCALE):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bicubic", antialias=True, align_corners=False
        )
