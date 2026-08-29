"""PyTorch port of the SR4RS generator (Cresson 2020, TF1 -> torch).

Architecture (verified against the checkpoint's graph:
  stem   conv 9x9 (4->64) + bias -> LeakyReLU                       -> E
  16x    ResBlock(64): conv3x3+b -> LReLU -> PixelNorm ->
                       conv3x3+b -> PixelNorm -> (+ block input)
  res_1x conv3x3+b -> PixelNorm -> (+ E long skip)   [64ch, LR grid]
         head: 1x1 conv+b -> 4ch                     (out_1x)
  res_2x fused conv_transpose x2 (StyleGAN2 shifted-kernel-sum, 64->256,
         no bias) -> Blur2D (depthwise) -> LReLU -> PixelNorm ->
         conv1 3x3 -> LReLU -> PN -> conv2 3x3 -> LReLU -> PN ->
         conv3 5x5 -> LReLU -> PN                    -> F2 (256ch, 2x grid)
         head: 1x1 conv+b -> 4ch                     (out_2x)
  res_4x same as res_2x from F2 (256->256, conv3 is 9x9)  -> F4 (4x grid)
         head: 1x1 conv+b -> 4ch                     (out_4x = THE output)

Weights come from `extract_sr4rs.py` (gen_weights.safetensors): kernels are
the graph's EFFECTIVE kernels (weight x equalized-LR constant, already OIHW),
so this module uses plain convs — no runtime eq-LR scaling.

I/O domain: reflectance (DN x 1e-4) in AND out — the TF graph's mul_3/mul_4
DN<->scaled conversions are deliberately outside this module, which makes it
contract-compatible with `JointSRUNetLightning.forward` (x/1e4 -> sr -> x1e4).

VRAM. This is a big generator run on a big grid: at bs=4, 128px LR -> 512px SR,
fp32 (which is what `JointSRUNetLightning._sr_forward`'s autocast-disabled
island gives it), the saved-for-backward set is ~18 GiB, of which res_4x alone
— 256 channels at 512px, ~1.07 GiB per retained tensor — is ~13 GiB. That is
what OOMs a 24 GB L4 on rl4. Set SR4RS_GRAD_CKPT=1 to checkpoint the res_2x /
res_4x stages (see `sr4rs_grad_ckpt_enabled`); it is off by default and
numerically exact when on.

Parity check against the TF reference taps (run wherever torch exists):
    python -m sr.sr4rs_torch --model-dir <...>/SR4RS_RGBN
"""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

SR4RS_SCALE = 4
SR4RS_BANDS = 4

_CKPT_ENV = "SR4RS_GRAD_CKPT"


def sr4rs_grad_ckpt_enabled() -> bool:
    """Whether to activation-checkpoint the res_2x / res_4x stages. OFF by default.

    Opt in with ``SR4RS_GRAD_CKPT=1``. An env flag rather than an hparam on
    purpose: it changes no tensor in the checkpoint and no number in the loss,
    so it must not become part of a run's identity — the rl series' between-arm
    contrasts stay valid whether or not an arm was run with it.

    It is exact, not approximate: this generator has no dropout and no RNG, so
    the recomputed forward reproduces the stored one bit-for-bit. The cost is
    one extra forward pass of each checkpointed stage.
    """
    return os.environ.get(_CKPT_ENV, "0").strip().lower() not in ("", "0", "false", "no")


def pixel_norm(x, eps):
    """x / sqrt(mean(x^2, channels) + eps) — TF reduces NHWC axis 3 == our 1."""
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


class EffConv(nn.Module):
    """Plain conv with TF-'SAME' padding FUSED into the convolution.

    For an odd kernel at stride 1, TF 'SAME' is exactly symmetric zero-padding
    by k//2, which is what ``F.conv2d(..., padding=k//2)`` does — so no
    separate ``F.pad`` is needed and the result is unchanged. Every SR4RS
    kernel is odd (1, 3, 5, 9); an EVEN kernel would need ASYMMETRIC padding
    (smaller pad first), which conv2d's scalar `padding=` cannot express, so it
    is rejected loudly here instead of being silently shifted half a pixel.

    Why fuse: a separate ``F.pad`` materialises a padded COPY of the input, and
    it is that copy — not the input — that autograd saves for the conv's
    backward, so the padding ring is carried for the whole backward pass and
    both tensors are live while the pad runs. Fused, the conv saves the
    already-live unpadded input instead.
    """

    def __init__(self, w, b=None):
        super().__init__()
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b) if b is not None else None
        kh, kw = int(w.shape[-2]), int(w.shape[-1])
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(
                f"EffConv got an even kernel {(kh, kw)}. TF 'SAME' pads even "
                "kernels asymmetrically ((k-1)//2 before, k//2 after) and "
                "conv2d's symmetric `padding=` cannot express that. All SR4RS "
                "kernels are odd, so this means the weights are not SR4RS's.")
        self.padding = (kh // 2, kw // 2)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=self.padding)


class FusedUpsample(nn.Module):
    """StyleGAN2 fused x2 upsample-conv: the 3x3 kernel is turned into a 4x4
    transposed-conv kernel by summing 4 shifted copies, then
    conv_transpose(stride 2, VALID) cropped to exactly 2x the input, followed
    by the shipped depthwise blur (SAME padding). Mirrors the graph's
    Pad/StridedSlice/AddN/Conv2DBackpropInput/Blur2D sequence."""

    def __init__(self, w_oihw, blur_filter, blur_same=True):
        super().__init__()
        self.weight = nn.Parameter(w_oihw)      # (out, in, 3, 3) effective
        out_c = w_oihw.shape[0]
        # The graph's filter_blur2d const ships in TF depthwise layout
        # (H, W, C, mult=1); use it EXACTLY as-is (no re-normalising), stored
        # as the torch depthwise weight (C, 1, H, W). A plain 2D (k, k)
        # filter is also accepted and replicated per channel.
        blur = torch.as_tensor(blur_filter, dtype=torch.float32)
        if blur.dim() == 4:                      # (H, W, C, 1) -> (C, 1, H, W)
            blur = blur.permute(2, 3, 0, 1).contiguous()
        elif blur.dim() == 2:                    # (k, k) -> (C, 1, k, k)
            k = blur.shape[-1]
            blur = blur.view(1, 1, k, k).expand(out_c, 1, k, k).contiguous()
        else:
            raise ValueError(f"unexpected blur filter shape {tuple(blur.shape)}")
        if blur.shape[0] != out_c:
            raise ValueError(f"blur has {blur.shape[0]} channels, conv outputs {out_c}")
        self.register_buffer("blur", blur)
        self.blur_same = blur_same

    def forward(self, x):
        out_c = self.weight.shape[0]
        # shifted-sum: pad 3x3 -> 4x4 (per StyleGAN2 upsample_conv_2d)
        w = F.pad(self.weight, (1, 1, 1, 1))
        w = (w[..., 1:, 1:] + w[..., :-1, 1:] + w[..., 1:, :-1] + w[..., :-1, :-1])
        # conv_transpose kernel layout: (in, out, kH, kW)
        wt = w.permute(1, 0, 2, 3)
        B, C, H, W = x.shape
        y = F.conv_transpose2d(x, wt, stride=2)          # (2H+2, 2W+2)
        y = y[..., 1:-1, 1:-1]                           # -> exactly (2H, 2W); TF SAME
        # depthwise blur (weight prepared as (C, 1, k, k) at init)
        k = self.blur.shape[-1]
        if self.blur_same:
            # TF SAME puts the SMALLER pad first: ((k-1)//2 before, k//2 after).
            # Identical for the shipped odd 3x3 blur; ordered correctly anyway
            # so an even blur kernel could never silently shift the output.
            #
            # NOT fused into the conv (unlike EffConv): the input here is a
            # non-contiguous crop of the conv_transpose output, so conv2d would
            # copy it anyway AND keep the larger uncropped base alive. The
            # explicit pad produces one contiguous tensor and lets the base go.
            y = F.pad(y, ((k - 1) // 2, k // 2, (k - 1) // 2, k // 2))
        return F.conv2d(y, self.blur, groups=out_c)


class SR4RSBlock(nn.Module):
    """res_2x / res_4x: upsample -> blur -> LReLU -> PN -> 3 convs, each
    followed by LReLU -> PN. `forward` returns (features, head_output).

    Activation checkpointing (SR4RS_GRAD_CKPT=1) is applied PER STAGE, not to
    the block as a whole. One segment per block would save nothing: backward
    walks res_4x first, so recomputing it materialises its full ~13 GiB while
    everything upstream is still held — the same peak as not checkpointing at
    all. Four segments cap the recompute transient at one stage's worth.
    """

    def __init__(self, t, scope, alpha, eps):
        super().__init__()
        self.alpha, self.eps = alpha, eps
        self.up = FusedUpsample(t[f"gen/{scope}/conv_upsample/weight"],
                                t[f"gen/{scope}/blur_filter"])
        self.convs = nn.ModuleList(
            EffConv(t[f"gen/{scope}/{c}/weight"], t[f"gen/{scope}/{c}/bias"])
            for c in ("conv1", "conv2", "conv3"))
        self.head = EffConv(t[f"gen/{scope}/output/weight"], t[f"gen/{scope}/output/bias"])
        self.grad_ckpt = sr4rs_grad_ckpt_enabled()

    def _up_stage(self, x):
        return pixel_norm(F.leaky_relu(self.up(x), self.alpha), self.eps)

    def _conv_stage(self, i, x):
        return pixel_norm(F.leaky_relu(self.convs[i](x), self.alpha), self.eps)

    def features(self, x):
        """Block features, without the 1x1 head."""
        stages = [self._up_stage]
        stages += [partial(self._conv_stage, i) for i in range(len(self.convs))]
        use_ckpt = self.grad_ckpt and torch.is_grad_enabled()
        for fn in stages:
            # use_reentrant=False so params-only grad still works and so the
            # recompute participates properly in the autograd graph.
            x = checkpoint(fn, x, use_reentrant=False) if use_ckpt else fn(x)
        return x

    def forward(self, x):
        f = self.features(x)
        return f, self.head(f)


class SR4RSGenerator(nn.Module):
    """4-band reflectance (B, 4, H, W) -> (B, 4, 4H, 4W) reflectance."""

    def __init__(self, tensors, meta):
        super().__init__()
        t = {k: torch.as_tensor(v) for k, v in tensors.items()}
        self.alpha = meta["lrelu_alpha"]
        self.eps = meta["pixelnorm_eps"]
        a, e = self.alpha, self.eps
        self.stem = EffConv(t["gen/encoder/conv1_9x9/weight"], t["gen/encoder/conv1_9x9/bias"])
        n = meta.get("resblocks", 16)
        self.blocks = nn.ModuleList()
        for i in range(n):
            self.blocks.append(nn.ModuleList([
                EffConv(t[f"gen/encoder/ResBlock{i}/conv1/weight"],
                        t[f"gen/encoder/ResBlock{i}/conv1/bias"]),
                EffConv(t[f"gen/encoder/ResBlock{i}/conv2/weight"],
                        t[f"gen/encoder/ResBlock{i}/conv2/bias"]),
            ]))
        self.res1x_conv = EffConv(t["gen/res_1x/conv1/weight"], t["gen/res_1x/conv1/bias"])
        self.res1x_head = EffConv(t["gen/res_1x/output/weight"], t["gen/res_1x/output/bias"])
        self.res2x = SR4RSBlock(t, "res_2x", a, e)
        self.res4x = SR4RSBlock(t, "res_4x", a, e)

    def _trunk(self, x, out=None):
        """stem -> 16 ResBlocks -> res_1x conv + long skip. Returns r1.

        The ONE implementation of the LR path: `forward` and `features` both
        go through it, so the parity-verified taps cannot drift away from what
        training actually runs. `out`, when given, collects the reference taps.
        """
        e = F.leaky_relu(self.stem(x), self.alpha)
        if out is not None:
            out["stem"] = e
        h = e
        for i, (c1, c2) in enumerate(self.blocks):
            y = pixel_norm(F.leaky_relu(c1(h), self.alpha), self.eps)
            y = pixel_norm(c2(y), self.eps)
            h = y + h
            if out is not None and i == 0:
                out["resblock0"] = h
        r1 = pixel_norm(self.res1x_conv(h), self.eps) + e
        if out is not None:
            out["res_1x_add"] = r1
        return r1

    def features(self, x):
        """All intermediate taps (mirrors extract_sr4rs.py's reference names).

        DIAGNOSTIC ONLY — `verify` and the viz scripts. It deliberately runs the
        res_2x upsample twice (once bare, for the `res_2x_blur` tap, once inside
        the block) and computes the out_1x/out_2x heads that nothing consumes;
        `forward` does none of that. Call it under `torch.no_grad`.
        """
        out = {}
        r1 = self._trunk(x, out)
        out["out_1x"] = self.res1x_head(r1)
        out["res_2x_blur"] = self.res2x.up(r1)
        f2, out["out_2x"] = self.res2x(r1)
        out["res_2x_feat"] = f2
        _, out["out_4x"] = self.res4x(f2)
        return out

    def forward(self, x):
        """The training path: out_4x only, with nothing computed that it does
        not need. (Was `features(x)["out_4x"]`, which additionally ran the
        res_2x upsample a second time, built two unused heads, and pinned the
        whole tap dict — and everything autograd had saved for it — alive for
        the duration of res_4x.)"""
        r1 = self._trunk(x)
        f2 = self.res2x.features(r1)
        return self.res4x.head(self.res4x.features(f2))


def load_trainable_sr4rs(model_dir) -> SR4RSGenerator:
    """Build the generator from extract_sr4rs.py's outputs in `model_dir`."""
    import json
    import safetensors.numpy
    model_dir = Path(model_dir)
    w = model_dir / "gen_weights.safetensors"
    if not w.exists():
        raise FileNotFoundError(
            f"{w} missing — run scripts/sr4rs/extract_sr4rs.py (TF venv) first.")
    tensors = safetensors.numpy.load_file(w)
    meta = json.loads((model_dir / "gen_meta.json").read_text())
    return SR4RSGenerator(tensors, meta)


def verify(model_dir, atol=1e-4):
    """Layer-by-layer parity against the TF reference taps."""
    import numpy as np
    ref = np.load(Path(model_dir) / "gen_reference.npz")
    model = load_trainable_sr4rs(model_dir).eval()
    with torch.no_grad():
        x = torch.as_tensor(ref["input"])
        got = model.features(x)
        # `forward` is a separate, leaner code path from `features` — assert it
        # is the SAME function, or the parity above certifies code training
        # never runs.
        fwd = model(x)
    ok = True
    for k in ("stem", "resblock0", "res_1x_add", "res_2x_blur", "res_2x_feat",
              "out_1x", "out_2x", "out_4x"):
        if k not in ref:
            continue
        d = float(np.abs(got[k].numpy() - ref[k]).max())
        status = "OK " if d <= atol else "FAIL"
        ok &= d <= atol
        print(f"  {status} {k:12s} max|dt-tf| = {d:.3e}")
    d = float((fwd - got["out_4x"]).abs().max())
    exact = bool(torch.equal(fwd, got["out_4x"]))
    ok &= d == 0.0
    print(f"  {'OK ' if exact else 'FAIL'} forward==features  max|d| = {d:.3e}")
    print("PARITY PASSED" if ok else "PARITY FAILED — fix before training!")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()
    raise SystemExit(0 if verify(args.model_dir, args.atol) else 1)
