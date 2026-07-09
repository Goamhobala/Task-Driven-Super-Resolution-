"""Visual comparison of SEN2SR before vs after joint task-driven fine-tuning.

Renders a 1x4 matplotlib grid from one Sentinel-2 tile:

    [ original 10 m | bicubic x4 (R0) | pretrained SEN2SR (R1) | fine-tuned SEN2SR (R2) ]

The pretrained panel loads the shipped SEN2SR-Lite weights from --sen2sr-dir;
the fine-tuned panel extracts the ``sr.*`` weights from a JointSRUNetLightning
checkpoint (the sr.cli refit). All three panels share ONE display stretch,
computed from the original image, so brightness/contrast differences you see
are real model behaviour, not per-panel autoscaling — deterioration of
perceptual quality under task-only training (Haris et al.) is expected and is
the point of the figure.

    python -m sr.viz --image <tile.tif> \
        --sen2sr-dir /scratch/$USER/InstaRoad/models/SEN2SRLite_RGBN \
        --ckpt runs/sr_optuna_graph_seed0/checkpoints/unet_s2rosa_jointsr_best.ckpt \
        [--ckpt-pre <other.ckpt>]     # optional: 2nd panel from a ckpt instead
        [--row 128 --col 256]         # crop offset in native px (default 0,0)
        [--out sr_compare.png]        # default: <image stem>_sr_compare.png

The crop is pinned to 128x128 native px (SEN2SR's shipped FFT mask).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from sr.model import REFLECTANCE_SCALE, SEN2SR_BANDS
from sr.sen2sr_loader import (
    SEN2SR_SCALE,
    BicubicUpsampler,
    load_trainable_sen2sr,
    pad_low_pass_mask,
)

NODATA = -32768
CROP = 128  # pinned by SEN2SR's 512x512 FFT low-pass mask


def read_patch(image_path, row, col):
    """(4, 128, 128) float32 reflectance in [B4,B3,B2,B8] order."""
    with rasterio.open(image_path) as src:
        if src.height < row + CROP or src.width < col + CROP:
            raise SystemExit(
                f"Crop [{row}:{row + CROP}, {col}:{col + CROP}] exceeds the "
                f"{src.height}x{src.width} image — adjust --row/--col."
            )
        x = src.read(list(SEN2SR_BANDS),
                     window=Window(col, row, CROP, CROP)).astype("float32")
    x[x == NODATA] = 0.0
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x / REFLECTANCE_SCALE


def load_sr_from_ckpt(ckpt_path, sen2sr_dir):
    """Rebuild SEN2SR + the ``sr.*`` weights from a Lightning ckpt.

    Returns ``(model, pad)``: if the run used ``model.sr_pad``, the ckpt's
    hard-constraint mask is larger than the shipped one — the pad is inferred
    from the size difference and the fresh mask resized to match before the
    strict load, so padded and unpadded checkpoints both work.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sr_sd = {k[len("sr."):]: v for k, v in sd.items() if k.startswith("sr.")}
    if not sr_sd:
        raise SystemExit(
            f"{ckpt_path} has no 'sr.*' keys — is this a JointSRUNetLightning "
            "checkpoint (sr.cli), not a plain unet one?"
        )
    model = load_trainable_sen2sr(sen2sr_dir)  # architecture + buffer plumbing
    base = model.hard_constraint.low_pass_mask.shape[-1]
    ckpt_mask = sr_sd["hard_constraint.low_pass_mask"].shape[-1]
    pad = (ckpt_mask - base) // (2 * SEN2SR_SCALE)
    if pad:
        pad_low_pass_mask(model, pad)
    model.load_state_dict(sr_sd, strict=True)
    return model, pad


@torch.no_grad()
def run_sr(model, x, pad=0):
    """(4,128,128) reflectance -> (4,512,512) reflectance (fp32, eval mode).

    ``pad``: reflect-pad the input, crop the output — mirrors model.sr_pad
    (moves the FFT border ring into discarded context)."""
    model.eval()
    t = torch.from_numpy(x)[None].float()
    if pad:
        t = torch.nn.functional.pad(t, (pad,) * 4, mode="reflect")
    out = model(t)[0]
    if pad:
        q = pad * SEN2SR_SCALE
        out = out[..., q:-q, q:-q]
    return out.numpy()


def to_rgb(x, lo, hi):
    """(4,H,W) reflectance -> (H,W,3) display RGB with the SHARED stretch."""
    rgb = np.transpose(x[:3], (1, 2, 0))  # bands are [B4,B3,B2,...] = R,G,B
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Sentinel-2 tile GeoTIFF (V2 layout, bands 1-4 raw)")
    ap.add_argument("--sen2sr-dir", required=True, help="shipped SEN2SR-Lite weights dir")
    ap.add_argument("--ckpt", required=True, help="fine-tuned JointSRUNetLightning .ckpt (3rd panel)")
    ap.add_argument("--ckpt-pre", default=None,
                    help="optional .ckpt for the 2nd panel (default: shipped pretrained weights)")
    ap.add_argument("--row", type=int, default=0, help="crop row offset, native px")
    ap.add_argument("--col", type=int, default=0, help="crop col offset, native px")
    ap.add_argument("--pad", type=int, default=0,
                    help="reflect-pad (native px) around the PRETRAINED panel to "
                         "suppress the FFT border ring (try 8). Ckpt panels "
                         "infer their own pad from the checkpoint.")
    ap.add_argument("--out", default=None, help="output PNG (default <image stem>_sr_compare.png)")
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98),
                    metavar=("PLO", "PHI"), help="shared percentile stretch (default 2 98)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # headless (HPC) safe
    import matplotlib.pyplot as plt

    x = read_patch(args.image, args.row, args.col)

    if args.ckpt_pre:
        pre_model, pre_pad = load_sr_from_ckpt(args.ckpt_pre, args.sen2sr_dir)
    else:
        pre_pad = args.pad
        pre_model = pad_low_pass_mask(load_trainable_sen2sr(args.sen2sr_dir), pre_pad)
    post_model, post_pad = load_sr_from_ckpt(args.ckpt, args.sen2sr_dir)
    if pre_pad or post_pad:
        print(f"pad: pretrained={pre_pad}px  fine-tuned={post_pad}px (from ckpt)")

    bicubic = run_sr(BicubicUpsampler(4), x)  # same op as the R0 baseline
    sr_pre = run_sr(pre_model, x, pre_pad)
    sr_post = run_sr(post_model, x, post_pad)

    # One stretch for all panels, from the ORIGINAL patch (fair comparison).
    lo, hi = np.percentile(x[:3], args.stretch)
    hi = max(hi, lo + 1e-6)

    diff = float(np.abs(sr_post - sr_pre).mean())
    print(f"mean |post - pre| reflectance over the SR outputs: {diff:.5f}")

    panels = [
        (to_rgb(x, lo, hi), f"Original 10 m ({CROP}px)"),
        (to_rgb(bicubic, lo, hi), "Bicubic x4 (R0)"),
        (to_rgb(sr_pre, lo, hi),
         ("SEN2SR pretrained (R1)" if not args.ckpt_pre else "SEN2SR pre ckpt")
         + (f", pad {pre_pad}" if pre_pad else "")),
        (to_rgb(sr_post, lo, hi),
         "SEN2SR fine-tuned (R2)" + (f", pad {post_pad}" if post_pad else "")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(f"{Path(args.image).stem}  crop r{args.row} c{args.col}  "
                 f"(shared {args.stretch[0]:g}-{args.stretch[1]:g}% stretch)",
                 fontsize=10)
    fig.tight_layout()

    out = args.out or f"{Path(args.image).stem}_sr_compare.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
