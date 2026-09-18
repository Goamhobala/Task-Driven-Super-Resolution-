"""Unit tests for the SR4RS torch port's TRAINING path (src/sr/sr4rs_torch.py).

`python -m sr.sr4rs_torch --model-dir <SR4RS_RGBN>` already checks the port
against the shipped TF reference taps, but it needs the 300 MB checkpoint dir
and it only exercises `features()`. These tests are self-contained (throwaway
weights, tiny channel counts) and pin the three things `features()` parity
cannot see:

  * `forward` — the lean path training actually runs — is bit-for-bit the
    `features()["out_4x"]` that parity certifies. Without this the two code
    paths could drift and the parity run would still print PASSED;
  * SR4RS_GRAD_CKPT changes NOTHING numerically, outputs and every gradient
    bit-identical. That is the licence for turning it on for one arm of a
    campaign whose whole point is between-arm comparability;
  * EffConv's TF-'SAME' padding is fused into the conv (shape-preserving) and
    an even kernel — which TF pads ASYMMETRICALLY, and `conv2d(padding=)`
    cannot — is rejected instead of silently shifting the image half a pixel.

  * `pixel_norm` keeps the ~1 GiB-per-tensor activations in the caller's dtype
    while computing its reduction in fp32, and is bit-identical to the naive
    formula in fp32. That is what makes bf16 actually buy memory.

Why fused padding, a lean forward and the dtype of a normalisation matter at
all: at bs=4 / 128 px LR the generator retains ~18 GiB of activations in fp32,
which is what OOMed rl4 on a 24 GB L4. bf16 (~9 GiB) and checkpointing
(~6 GiB) are the two levers that fix it; rl4 now uses the former.
"""
from __future__ import annotations

import pytest
import torch

from sr.sr4rs_torch import (
    EffConv, SR4RSGenerator, pixel_norm, sr4rs_grad_ckpt_enabled)

C1, C2, NB = 6, 8, 2          # LR channels, upsampled channels, resblocks
BANDS, P = 4, 8               # 4 bands, 8 px LR patch -> 32 px SR


def _toy_tensors(seed=0):
    """The exact key set `SR4RSGenerator.__init__` reads, at toy sizes."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.1

    t = {
        "gen/encoder/conv1_9x9/weight": r(C1, BANDS, 5, 5),
        "gen/encoder/conv1_9x9/bias": r(C1),
        "gen/res_1x/conv1/weight": r(C1, C1, 3, 3),
        "gen/res_1x/conv1/bias": r(C1),
        "gen/res_1x/output/weight": r(BANDS, C1, 1, 1),
        "gen/res_1x/output/bias": r(BANDS),
    }
    for i in range(NB):
        for c in ("conv1", "conv2"):
            t[f"gen/encoder/ResBlock{i}/{c}/weight"] = r(C1, C1, 3, 3)
            t[f"gen/encoder/ResBlock{i}/{c}/bias"] = r(C1)
    for scope, cin, k3 in (("res_2x", C1, 5), ("res_4x", C2, 9)):
        t[f"gen/{scope}/conv_upsample/weight"] = r(C2, cin, 3, 3)
        t[f"gen/{scope}/blur_filter"] = torch.rand(3, 3, C2, 1, generator=g)
        for c, k in (("conv1", 3), ("conv2", 3), ("conv3", k3)):
            t[f"gen/{scope}/{c}/weight"] = r(C2, C2, k, k)
            t[f"gen/{scope}/{c}/bias"] = r(C2)
        t[f"gen/{scope}/output/weight"] = r(BANDS, C2, 1, 1)
        t[f"gen/{scope}/output/bias"] = r(BANDS)
    return t


_META = {"resblocks": NB, "lrelu_alpha": 0.2, "pixelnorm_eps": 1e-8}


def _model(grad_ckpt=False):
    m = SR4RSGenerator(_toy_tensors(), _META)
    m.res2x.grad_ckpt = m.res4x.grad_ckpt = grad_ckpt
    return m


def test_forward_matches_features_out_4x_exactly():
    """`forward` is lean (no duplicate upsample, no unused heads, no tap dict)
    but must stay the SAME function as the parity-verified `features`."""
    m = _model()
    x = torch.rand(2, BANDS, P, P) * 0.3
    with torch.no_grad():
        assert torch.equal(m(x), m.features(x)["out_4x"])


def test_output_is_4x_and_4_band():
    m = _model()
    with torch.no_grad():
        y = m(torch.rand(1, BANDS, P, P))
    assert y.shape == (1, BANDS, 4 * P, 4 * P)


@pytest.mark.parametrize("shape", [(1, BANDS, P, P), (3, BANDS, P, P)])
def test_grad_checkpointing_is_bit_exact(shape):
    """Outputs AND every gradient identical with checkpointing on/off.

    This generator has no dropout and no RNG, so the recomputed forward is the
    stored one — an arm run with SR4RS_GRAD_CKPT=1 is comparable with one run
    without it. If this ever fails, the flag is no longer free.
    """
    x = torch.rand(*shape) * 0.3

    def run(ckpt):
        m = _model(grad_ckpt=ckpt)
        y = m(x)
        y.square().mean().backward()
        return y.detach(), {n: p.grad.clone() for n, p in m.named_parameters()
                            if p.grad is not None}

    y0, g0 = run(False)
    y1, g1 = run(True)
    assert torch.equal(y0, y1)
    assert g0.keys() == g1.keys() and g0, "checkpointing changed the grad set"
    bad = {k: float((g0[k] - g1[k]).abs().max()) for k in g0
           if not torch.equal(g0[k], g1[k])}
    assert not bad, f"checkpointing perturbed gradients: {bad}"


def test_grad_checkpointing_is_inert_under_no_grad():
    """Inference must not pay for — or trip over — the checkpoint wrapper."""
    m = _model(grad_ckpt=True)
    x = torch.rand(1, BANDS, P, P) * 0.3
    with torch.no_grad():
        assert torch.equal(m(x), _model(grad_ckpt=False)(x))


@pytest.mark.parametrize("value,want", [
    (None, False), ("0", False), ("", False), ("false", False), ("no", False),
    ("1", True), ("true", True),
])
def test_ckpt_env_flag(monkeypatch, value, want):
    monkeypatch.delenv("SR4RS_GRAD_CKPT", raising=False)
    if value is not None:
        monkeypatch.setenv("SR4RS_GRAD_CKPT", value)
    assert sr4rs_grad_ckpt_enabled() is want


@pytest.mark.parametrize("k", [1, 3, 5, 9])
def test_effconv_same_padding_preserves_shape(k):
    """Fused `padding=k//2` is TF 'SAME' for odd kernels at stride 1: same
    output grid, and identical to the explicit-pad-then-VALID form it replaced."""
    conv = EffConv(torch.randn(3, 2, k, k) * 0.1, torch.randn(3) * 0.1)
    x = torch.randn(1, 2, 11, 13)
    got = conv(x)
    assert got.shape == (1, 3, 11, 13)
    if k > 1:
        ref = torch.nn.functional.conv2d(
            torch.nn.functional.pad(x, (k // 2,) * 4), conv.weight, conv.bias)
        assert torch.allclose(got, ref, atol=0, rtol=0) or \
            torch.allclose(got, ref, atol=1e-6)


def test_effconv_rejects_even_kernels():
    """TF pads even kernels asymmetrically; a symmetric `padding=` would shift
    the output half a pixel, silently. All SR4RS kernels are odd."""
    with pytest.raises(ValueError, match="even kernel"):
        EffConv(torch.randn(3, 2, 4, 4))


# ------------------------------------------------------------- pixel_norm dtype
def test_pixel_norm_fp32_matches_the_naive_formula_bit_for_bit():
    """The fp32 path must be untouched by the fp32-reduction rewrite.

    `.float()` on an fp32 tensor and `.to(torch.float32)` on an fp32 result are
    both no-ops, so this is an identity — but it is the identity that lets every
    checkpoint written before the rewrite, and the TF parity run, stand.
    """
    x = torch.randn(2, 16, 5, 5, generator=torch.Generator().manual_seed(3))
    naive = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)
    assert torch.equal(pixel_norm(x, 1e-8), naive)


def test_pixel_norm_does_not_upcast_its_output():
    """bf16 in -> bf16 out, so the big activations stay half-width.

    Pinned because the natural spelling does NOT do this under CUDA autocast:
    `pow`/`rsqrt` promote to fp32 there and `bf16 * fp32 -> fp32`, which would
    hand back every PixelNorm output — ~1.07 GiB each at 256 ch / 512 px — in
    fp32 and undo most of what bf16 was adopted for. Device-independent: the
    invariant is the cast, not the autocast policy that motivates it.
    """
    x = torch.randn(2, 16, 5, 5,
                    generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    y = pixel_norm(x, 1e-8)
    assert y.dtype is torch.bfloat16
    # ...and the values are still the fp32-statistic ones to within bf16's own
    # quantisation (~0.4% per rounding, two roundings here) — i.e. the dtype of
    # the OUTPUT was traded away, the correctness of the NORMALISER was not.
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + 1e-8)
    assert (y.float() - ref).abs().max() <= 1e-2 * ref.abs().max()
