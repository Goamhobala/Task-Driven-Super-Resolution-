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

The constraint is also detachable and re-mountable (docs/hc_2x2_plan.md): the
same operator can be taken OFF SEN2SR (r2b) or put ON SR4RS (r4a) to separate
the hard constraint from the generator architecture. See `resolve_sr_hc` for
the tri-state flag and `TrainableSEN2SR` for why the clamp travels with it.

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

# Tri-state for the `sr_hc` treatment flag (docs/hc_2x2_plan.md §5.1).
SR_HC_MODES = ("native", "on", "off")


def resolve_sr_hc(upsampler: str, sr_hc: str | None = "native") -> bool:
    """Resolve the tri-state `sr_hc` flag to on/off for a given upsampler.

    ``native`` = the per-generator default each upsampler has always shipped
    with (SEN2SR applies the FFT hard constraint, SR4RS and bicubic do not), so
    any checkpoint written before this flag existed resolves to exactly the
    behaviour it was trained under. ``on``/``off`` force the constraint on or
    off, which is what crosses it with the generator in the HC 2x2.

    ``bicubic + on`` is rejected rather than run: HardConstraint(bicubic(x),
    bicubic(x)) splices a spectrum with itself, so the cell would be a
    near-identity dressed up as a treatment.
    """
    mode = sr_hc or "native"
    if mode not in SR_HC_MODES:
        raise ValueError(f"sr_hc={sr_hc!r} (choose from {SR_HC_MODES})")
    if mode == "native":
        return upsampler in ("sen2sr", "sen2sr_full")
    if mode == "on" and upsampler == "bicubic":
        raise ValueError(
            "sr_hc='on' is meaningless for upsampler='bicubic': the hard "
            "constraint splices the bicubic upsampling of the input into the "
            "SR output, which for a bicubic 'generator' is a near-identity."
        )
    return mode == "on"


def download_sen2sr(model_dir) -> Path:
    """Fetch the SEN2SR-Lite RGBN ×4 weights via mlstac (idempotent)."""
    import mlstac

    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensor").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        mlstac.download(file=SEN2SR_RGBN_X4_URL, output_dir=str(model_dir))
    return model_dir


class TrainableSEN2SR(nn.Module):
    """A generator plus, optionally, the frozen FFT hard-constraint bundle.

    Forward contract (with the constraint mounted, identical to the upstream
    `srmodel` wrapper):
        reflectance (B, 4, H, W) -> reflectance (B, 4, 4H, 4W)

    The clamp and the frequency splice are ONE treatment, not two knobs
    (docs/hc_2x2_plan.md §4, "Scheme B"): upstream ships them fused, and the
    paper defines the hard-constraint layer by both of its conditions —
    positivity (the clamp) and spectral consistency (the splice). So
    `hard_constraint=None` comes with `clamp_min=None` and yields the RAW
    generator output, which is what makes the HC-off column compare raw
    generator to raw generator. `clamp_min` stays a separate argument only so
    the bundle's components are legible; do not build the half-way combination
    without a reason stated in the arm's script.
    """

    def __init__(self, sr_model: nn.Module, hard_constraint: nn.Module | None = None,
                 clamp_min: float | None = 0.0):
        super().__init__()
        self.sr_model = sr_model
        # nn.Module.__setattr__ registers a Module here and falls through to a
        # plain attribute for None, so `self.hard_constraint` is always safe to
        # read and the state_dict simply has no `hard_constraint.*` keys when
        # the constraint is off (a wrong-config restore then fails loudly on the
        # strict load, which is the intent).
        self.hard_constraint = hard_constraint
        self.clamp_min = clamp_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sr = self.sr_model(x)
        if self.clamp_min is not None:
            sr = torch.clamp(sr, min=self.clamp_min)
        if self.hard_constraint is None:
            return sr
        return self.hard_constraint(x, sr)


def hard_constraint_from_mask(mask: torch.Tensor) -> nn.Module:
    """Frozen `HardConstraint` around an already-loaded low-pass mask.

    Upstream keeps `low_pass_mask` as a plain tensor attribute; re-register it
    as a buffer so `.to(device)` / Lightning device placement move it too.
    """
    from sen2sr.models.tricks import HardConstraint

    hc = HardConstraint(low_pass_mask=mask, bands="all")
    for p in hc.parameters():
        p.requires_grad = False
    del hc.low_pass_mask
    hc.register_buffer("low_pass_mask", mask)
    return hc


def build_hard_constraint(mask_path) -> nn.Module:
    """Load a `hard_constraint.safetensor` as a frozen `HardConstraint`.

    Factored out of `load_trainable_sen2sr` so a different generator (SR4RS,
    r4a) can mount the IDENTICAL operator: the shipped SEN2SR-Lite mask is a
    single 512x512 sigma=35 Gaussian shared across bands, and `HardConstraint`
    itself is a pure function of (lr, sr, mask) that touches no generator
    internals. Reuse the file byte-for-byte rather than re-deriving a mask —
    the deployed cutoff is the value Table 4 of Aybar et al. optimised.
    """
    import safetensors.torch

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"{mask_path} missing — the FFT hard constraint needs the shipped "
            "mask (it ships inside the SEN2SR-Lite model dir as "
            "hard_constraint.safetensor)."
        )
    return hard_constraint_from_mask(
        safetensors.torch.load_file(mask_path)["weights"])


def load_trainable_sen2sr(model_dir, hard_constraint: bool = True) -> TrainableSEN2SR:
    """Load SEN2SR-Lite RGBN ×4 from `model_dir` as a trainable nn.Module.

    `hard_constraint=False` returns the BARE CNNSR generator (no clamp, no FFT
    splice) — the r2b cell of the HC 2x2. Note SEN2SR-Lite was trained with the
    bundle in the loop, so its raw output was never a deployed product.
    """
    import safetensors.torch
    from sen2sr.models.opensr_baseline.cnn import CNNSR

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

    if not hard_constraint:
        return TrainableSEN2SR(sr_model, None, clamp_min=None)
    return TrainableSEN2SR(
        sr_model,
        build_hard_constraint(model_dir / "hard_constraint.safetensor"),
        clamp_min=0.0,
    )


def _enable_mamba_grad_checkpointing(sr_model: nn.Module) -> int:
    """Per-VSSBlock activation checkpointing for the full (Mamba) SEN2SR.

    Training through MambaSR at the FFT-pinned 128 px LR patch stores the
    SS2D internals (4 directional scans over a 16k+ token sequence, in fp32 —
    the SR stage runs autocast-disabled for the FFT constraint) for EVERY
    VSSBlock: >44 GB even at batch_size=2 (r3 OOM). Checkpointing stores only
    block-boundary tensors and recomputes the internals during backward —
    numerically identical, ~25-30% slower SR stage, activation memory drops
    by roughly the per-layer block count.

    Upstream ships a ``use_checkpoint`` flag but its branch is broken:
    ``checkpoint.checkpoint(blk, x)`` drops VSSBlock's required ``x_size``
    argument (instant TypeError). So we bind a corrected forward onto each
    ``BasicLayer`` INSTANCE instead — no wrapper modules, so parameter names
    (and therefore existing Lightning checkpoints) are untouched.

    Returns the number of patched layers (0 = architecture had no
    BasicLayers; caller should warn, not crash).
    """
    import torch.utils.checkpoint as _ckpt
    from sen2sr.models.opensr_baseline.mamba import BasicLayer

    def _make_forward(layer: nn.Module):
        def forward(x, x_size):
            for blk in layer.blocks:
                if torch.is_grad_enabled() and x.requires_grad:
                    # use_reentrant=False: supports the non-tensor x_size arg
                    # and preserves DropPath RNG semantics on recompute.
                    x = _ckpt.checkpoint(blk, x, x_size, use_reentrant=False)
                else:
                    # eval / frozen-SR no_grad path: checkpointing would be
                    # pure recompute overhead with nothing to save.
                    x = blk(x, x_size)
            if layer.downsample is not None:
                x = layer.downsample(x)
            return x

        return forward

    n = 0
    for m in sr_model.modules():
        if isinstance(m, BasicLayer):
            m.forward = _make_forward(m)
            n += 1
    return n


def load_trainable_sen2sr_full(model_dir, hard_constraint: bool = True) -> TrainableSEN2SR:
    """Load the FULL (Mamba) SEN2SR RGBN x4 from `model_dir` as trainable.

    Unlike the Lite/CNN path, `MambaSR` has no train_mode/eval_conv collapse
    quirk, so mlstac's ``trainable_model()`` supplies sound architecture +
    weights — but its model card needs TWO corrections at load time:

      * ``device="cpu"`` must be passed explicitly. The FULL card defaults to
        ``device="cuda:0"`` (the Lite card defaults to ``"cpu"``) and eagerly
        ``.to()``s at construction — inside ``JointSRUNetLightning.__init__``,
        i.e. before Lightning does any device placement. In any process where
        CUDA is masked (e.g. a tune worker pinned to a nonexistent ordinal)
        that dies with torch's "No CUDA GPUs are available". Construct on CPU
        and let Lightning move things, exactly like the Lite/SR4RS loaders.
      * the returned object is upstream's ``srmodel`` wrapper (SR net + its
        own `HardConstraint` INSIDE forward). We must unwrap ``.sr_model``:
        keeping the wrapper would (a) apply the FFT constraint twice (ours on
        top of its), (b) leave its plain-attribute mask stranded on the
        construction device after Lightning moves the model, and (c) bypass
        `pad_low_pass_mask`, so any ``sr_pad > 0`` run would crash on a
        mask/input FFT size mismatch (r3a: pad 8 -> 576px vs 512 mask).

    We then re-add the frozen FFT `HardConstraint` from
    ``hard_constraint.safetensor`` as a movable buffer and wrap in the same
    `TrainableSEN2SR` interface as the Lite loader, so `JointSRUNetLightning`
    treats both variants identically.

    Requires the ``mamba_ssm`` package (CUDA build) in the training venv:
        uv pip install mamba-ssm   # on a node with nvcc / matching torch
    """
    import mlstac

    model_dir = Path(model_dir)
    wrapper = mlstac.load(str(model_dir)).trainable_model(device="cpu")
    # srmodel wrapper -> raw MambaSR; tolerate a future card returning the
    # bare model (getattr falls through, and we add our own constraint below).
    sr_model = getattr(wrapper, "sr_model", wrapper)
    if not any(p.requires_grad for p in sr_model.parameters()):
        raise RuntimeError(
            f"mlstac's trainable_model() for {model_dir} yielded NO trainable "
            "parameters — inspect the model dir (this loader assumes the full "
            "MambaSR variant, which unlike Lite/CNNSR needs no train_mode fix)."
        )

    n_ckpt = _enable_mamba_grad_checkpointing(sr_model)
    if n_ckpt:
        print(f"[sen2sr_loader] gradient checkpointing enabled on {n_ckpt} "
              "MambaSR BasicLayers (joint training does not fit 44 GB without it)")
    else:
        print("[sen2sr_loader] WARN: no BasicLayer found to checkpoint — "
              "non-Mamba card? joint training may OOM.")

    if not hard_constraint:
        return TrainableSEN2SR(sr_model, None, clamp_min=None)
    return TrainableSEN2SR(
        sr_model,
        build_hard_constraint(model_dir / "hard_constraint.safetensor"),
        clamp_min=0.0,
    )


def pad_low_pass_mask(model: TrainableSEN2SR, pad: int, scale: int = SEN2SR_SCALE):
    """Resize the FFT hard-constraint mask so the model accepts inputs
    reflect-padded by ``pad`` native px per side (HR size grows by
    ``2*pad*scale``).

    Valid because `HardConstraint` applies the mask in fftSHIFTED space (DC at
    the centre): bicubic resampling of the centred radial low-pass mask keeps
    the cutoff at the same fraction of Nyquist. In-place; returns the model.
    """
    if pad <= 0:
        return model
    hc = getattr(model, "hard_constraint", None)
    if hc is None:
        raise ValueError(
            "pad_low_pass_mask called on a model with no hard constraint — "
            "there is no mask to grow. The generic reflect-pad/crop in "
            "JointSRUNetLightning._sr_forward covers sr_pad on its own."
        )
    m = hc.low_pass_mask
    h, w = m.shape[-2:]
    new_hw = (h + 2 * pad * scale, w + 2 * pad * scale)
    flat = m.reshape(1, -1, h, w).float()          # (1, C*, H, W) for interpolate
    resized = nn.functional.interpolate(flat, size=new_hw, mode="bicubic",
                                        align_corners=False)
    resized = resized.reshape(*m.shape[:-2], *new_hw).to(m.dtype)
    del model.hard_constraint.low_pass_mask
    model.hard_constraint.register_buffer("low_pass_mask", resized)
    return model


class BicubicUpsampler(nn.Module):
    """Parameter-free ×`scale` bicubic upsampling — the R0 deterministic baseline.

    Same forward contract as `TrainableSEN2SR` so `JointSRUNetLightning` can
    swap between them via config.
    """

    def __init__(self, scale: int = SEN2SR_SCALE):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bicubic", antialias=True, align_corners=False
        )
