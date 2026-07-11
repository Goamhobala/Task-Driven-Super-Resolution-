"""PyTorch port of the SR4RS generator (Cresson 2020, TF1 -> torch).

Architecture (verified against the checkpoint's graph, not guessed):
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

Parity check against the TF reference taps (run wherever torch exists):
    python -m sr.sr4rs_torch --model-dir <...>/SR4RS_RGBN
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SR4RS_SCALE = 4
SR4RS_BANDS = 4


def pixel_norm(x, eps):
    """x / sqrt(mean(x^2, channels) + eps) — TF reduces NHWC axis 3 == our 1."""
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


class EffConv(nn.Module):
    """Plain conv with TF-'SAME' padding (asymmetric for even kernels; all
    SR4RS kernels are odd, so symmetric F.pad matches TF exactly here)."""

    def __init__(self, w, b=None):
        super().__init__()
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b) if b is not None else None
        k = w.shape[-1]
        self.pad = (k // 2,) * 4 if k > 1 else None

    def forward(self, x):
        if self.pad:
            x = F.pad(x, self.pad)  # zero-pad == TF SAME (odd kernels, stride 1)
        return F.conv2d(x, self.weight, self.bias)


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
        out_c, in_c = self.weight.shape[:2]
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
            y = F.pad(y, (k // 2, (k - 1) // 2, k // 2, (k - 1) // 2))
        return F.conv2d(y, self.blur, groups=out_c)


class SR4RSBlock(nn.Module):
    """res_2x / res_4x: upsample -> blur -> LReLU -> PN -> 3 convs, each
    followed by LReLU -> PN. Returns (features, head_output)."""

    def __init__(self, t, scope, alpha, eps):
        super().__init__()
        self.alpha, self.eps = alpha, eps
        self.up = FusedUpsample(t[f"gen/{scope}/conv_upsample/weight"],
                                t[f"gen/{scope}/blur_filter"])
        self.convs = nn.ModuleList(
            EffConv(t[f"gen/{scope}/{c}/weight"], t[f"gen/{scope}/{c}/bias"])
            for c in ("conv1", "conv2", "conv3"))
        self.head = EffConv(t[f"gen/{scope}/output/weight"], t[f"gen/{scope}/output/bias"])

    def forward(self, x):
        x = pixel_norm(F.leaky_relu(self.up(x), self.alpha), self.eps)
        for conv in self.convs:
            x = pixel_norm(F.leaky_relu(conv(x), self.alpha), self.eps)
        return x, self.head(x)


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

    def features(self, x):
        """All intermediate taps (mirrors extract_sr4rs.py's reference names)."""
        out = {}
        e = F.leaky_relu(self.stem(x), self.alpha)
        out["stem"] = e
        h = e
        for i, (c1, c2) in enumerate(self.blocks):
            y = pixel_norm(F.leaky_relu(c1(h), self.alpha), self.eps)
            y = pixel_norm(c2(y), self.eps)
            h = y + h
            if i == 0:
                out["resblock0"] = h
        r1 = pixel_norm(self.res1x_conv(h), self.eps) + e
        out["res_1x_add"] = r1
        out["out_1x"] = self.res1x_head(r1)
        out["res_2x_blur"] = self.res2x.up(r1)
        f2, out["out_2x"] = self.res2x(r1)
        out["res_2x_feat"] = f2
        _, out["out_4x"] = self.res4x(f2)
        return out

    def forward(self, x):
        return self.features(x)["out_4x"]


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
        got = model.features(torch.as_tensor(ref["input"]))
    ok = True
    for k in ("stem", "resblock0", "res_1x_add", "res_2x_blur", "res_2x_feat",
              "out_1x", "out_2x", "out_4x"):
        if k not in ref:
            continue
        d = float(np.abs(got[k].numpy() - ref[k]).max())
        status = "OK " if d <= atol else "FAIL"
        ok &= d <= atol
        print(f"  {status} {k:12s} max|dt-tf| = {d:.3e}")
    print("PARITY PASSED" if ok else "PARITY FAILED — fix before training!")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--atol", type=float, default=1e-4)
    raise SystemExit(0 if verify(ap.parse_args().model_dir,
                                 ap.parse_args().atol) else 1)
