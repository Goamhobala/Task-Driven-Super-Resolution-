"""Four-panel horizontal strip for the thesis: one 128 px crop, four views.

    | Original 10 m | Bicubic x4 | SEN2SR-Lite | SR4RS |

No figure title — panels carry only their label; the caption is written by hand
in the document. Every panel shares ONE percentile stretch, computed on the
original 10 m crop, so a model cannot look better merely by moving its
histogram.

Both generators take 0-1 reflectance in and out. SEN2SR-Lite's FFT hard
constraint pins the LR patch at 128 px, so it is fed the crop verbatim; SR4RS
is a GAN with a ~32 px invalid border, so it gets a 32 px reflect pad that is
cropped back off its output (the same `sr_pad` treatment the joint runs use).

    PYTHONPATH=src nice -n 19 python scripts/local/fig_sr_strip.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set before torch loads its Metal allocator, per the guard rails in viz_sr:
# an unbounded MPS cache on a 128 px crop is still enough to stall the machine.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from sr.model import REFLECTANCE_SCALE, SEN2SR_BANDS

NODATA = -32768
CROP = 128          # SEN2SR's FFT mask pins the LR patch size
SR4RS_PAD = 32      # SR4RS's invalid GAN margin, in LR pixels

EXAMPLES_DIR = "src/sr/examples"
DEFAULT_IMAGE = "Durban_r4_c3.tif"
SR4RS_DIR = "models/SR4RS_RGBN"


def read_patch(image_path, row, col, crop=CROP):
    """(4, crop, crop) float32 in [B4,B3,B2,B8] order, as stored on disk."""
    with rasterio.open(image_path) as src:
        x = src.read(list(SEN2SR_BANDS),
                     window=Window(col, row, crop, crop)).astype("float32")
    x[x == NODATA] = 0.0
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def to_rgb(x, lo, hi):
    rgb = np.transpose(np.asarray(x)[:3], (1, 2, 0))   # [B4,B3,B2] = R,G,B
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


@torch.no_grad()
def run_sr(module, refl, pad=0, upscale=4):
    """One SR net's output on a (1,4,H,W) reflectance tensor, via a reflect pad
    that is cropped back off at HR scale."""
    t = torch.nn.functional.pad(refl, (pad,) * 4, mode="reflect") if pad else refl
    hr = module(t)
    if pad:
        q = pad * upscale
        hr = hr[..., q:-q, q:-q]
    return hr[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", default=EXAMPLES_DIR)
    ap.add_argument("--image", default=None, help=f"default: {DEFAULT_IMAGE}")
    ap.add_argument("--sr4rs-dir", default=SR4RS_DIR)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98),
                    metavar=("PLO", "PHI"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="figures/sr_strip.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = Path(args.examples_dir)
    image = Path(args.image) if args.image else examples / DEFAULT_IMAGE
    if not image.exists():
        raise SystemExit(f"image not found: {image}")

    x = read_patch(image, args.row, args.col)
    lo, hi = np.percentile(x[:3], args.stretch)
    hi = max(hi, lo + 1e-6)
    # Storage units of THIS tile: V2 COGs already hold 0-1 reflectance.
    tile_scale = 1.0 if float(np.nanmax(x)) <= 1.5 else REFLECTANCE_SCALE
    refl = torch.from_numpy(x)[None].float().to(args.device) / tile_scale

    from sr.sen2sr_loader import BicubicUpsampler, load_trainable_sen2sr
    from sr.sr4rs_torch import load_trainable_sr4rs

    bicubic = run_sr(BicubicUpsampler(4).eval().to(args.device), refl)
    sen2sr = run_sr(
        load_trainable_sen2sr(str(examples), hard_constraint=True)
        .eval().to(args.device), refl)
    sr4rs = run_sr(
        load_trainable_sr4rs(args.sr4rs_dir).eval().to(args.device),
        refl, pad=SR4RS_PAD)

    # The original is nearest-upsampled x4 so all four panels sit on the same
    # 512 px grid: the blockiness IS the 10 m resolution, not an artefact.
    original = np.repeat(np.repeat(x / tile_scale, 4, axis=1), 4, axis=2)

    panels = [(original, "Original 10 m"),
              (bicubic, "Bicubic $\\times$4"),
              (sen2sr, "SEN2SR-Lite"),
              (sr4rs, "SR4RS")]

    lo_r, hi_r = lo / tile_scale, hi / tile_scale
    fig, axes = plt.subplots(1, 4, figsize=(4 * 2.6, 2.85))
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(to_rgb(img, lo_r, hi_r), interpolation="nearest")
        ax.set_xlabel(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.03, left=0.005, right=0.995,
                        top=0.995, bottom=0.075)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
