"""Sanity check for the converted PRETRAINED SR4RS (no fine-tuning): what the
UNet sees at epoch 0. Renders  [ original 10 m | bicubic x4 | SR4RS x4 ]  with
one shared stretch and prints per-band input/output statistics (the SR output
should stay in the same reflectance ballpark as the input — that is what the
frozen norm-stats z-score downstream assumes).

    PYTHONPATH=src python scripts/sr4rs/sanity_viz.py \
        [--model-dir models/SR4RS_RGBN] [--image src/sr/examples/<tile>.tif] \
        [--row 0 --col 0] [--out sr4rs_sanity.png]

Input contract (verified against the shipped model's graph): bands 1-4 of the
V2 COG = [B4 red, B3 green, B2 blue, B8 NIR], values DN/10000 (LRSC0.0001).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from sr.sr4rs_torch import SR4RS_BANDS, load_trainable_sr4rs  # noqa: F401

CROP = 128
BANDS = (1, 2, 3, 4)          # [B4_R, B3_G, B2_B, B8_NIR] in the V2 layout
NAMES = ("B4 red", "B3 green", "B2 blue", "B8 nir")


@torch.inference_mode()
def tiled_sr(model, t, tile=32, ctx=8, scale=4):
    """Memory-safe SR: run overlapping input tiles, keep only each tile's core.

    Peak memory scales with (tile+2*ctx)^2 instead of the full frame — on CPU a
    whole 128px frame can demand ~20 GB (im2col of the 9x9 conv over 256ch at
    512px), which freezes small machines; tile=32/ctx=8 stays under ~1 GB.
    The ctx margin absorbs each tile's own edge effects; faint seams may remain
    (sanity-viz only — training/HPC runs the full frame)."""
    _, C, H, W = t.shape
    out = torch.zeros(1, C, H * scale, W * scale)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            ya, xa = max(0, y0 - ctx), max(0, x0 - ctx)
            yb, xb = min(H, y0 + tile + ctx), min(W, x0 + tile + ctx)
            sr = model(t[..., ya:yb, xa:xb])
            cy, cx = (y0 - ya) * scale, (x0 - xa) * scale
            h = (min(H, y0 + tile) - y0) * scale
            w = (min(W, x0 + tile) - x0) * scale
            out[..., y0 * scale:y0 * scale + h,
                x0 * scale:x0 * scale + w] = sr[..., cy:cy + h, cx:cx + w].cpu()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models/SR4RS_RGBN")
    ap.add_argument("--image", default=None,
                    help="V2 tile (default: first .tif in src/sr/examples)")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98))
    ap.add_argument("--out", default="sr4rs_sanity.png")
    ap.add_argument("--tile", type=int, default=32,
                    help="input tile size for memory-safe SR (0 = whole frame; "
                         "only do that with >=8GB of GPU VRAM)")
    ap.add_argument("--ctx", type=int, default=8, help="tile overlap margin (px)")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | mps (default: cuda if available, else cpu)")
    ap.add_argument("--threads", type=int, default=4,
                    help="cap CPU threads so the machine stays responsive")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.image is None:
        tifs = sorted(t for t in Path("src/sr/examples").glob("*.tif")
                      if not t.stem.endswith("_mask"))
        if not tifs:
            raise SystemExit("no example tile found — pass --image")
        args.image = str(tifs[0])
        print(f"image: {args.image}")

    with rasterio.open(args.image) as src:
        x = src.read(list(BANDS),
                     window=Window(args.col, args.row, CROP, CROP)).astype("float32")
    x[x == -32768] = 0.0
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # SR4RS wants 0-1 reflectance. Autodetect the stored units: V2 COGs are
    # ALREADY reflectance (values <~1); DN-valued rasters need /10000.
    scale = 10000.0 if float(x.max()) > 20.0 else 1.0
    print(f"input units: {'DN -> /10000' if scale > 1 else 'already 0-1 reflectance'}"
          f" (raw max {x.max():.3f})")
    x /= scale

    model = load_trainable_sr4rs(args.model_dir).eval().to(device)
    t = torch.from_numpy(x)[None].to(device)
    with torch.inference_mode():
        if args.tile > 0:
            sr = tiled_sr(model, t, tile=args.tile, ctx=args.ctx)[0].numpy()
        else:
            sr = model(t)[0].cpu().numpy()
        bic = torch.nn.functional.interpolate(
            t, scale_factor=4, mode="bicubic", antialias=True)[0].cpu().numpy()

    print(f"\n{'band':10s} {'in mean':>9s} {'in std':>8s} {'sr mean':>9s} "
          f"{'sr std':>8s} {'sr min':>8s} {'sr max':>8s}")
    for i, n in enumerate(NAMES):
        print(f"{n:10s} {x[i].mean():9.4f} {x[i].std():8.4f} {sr[i].mean():9.4f} "
              f"{sr[i].std():8.4f} {sr[i].min():8.4f} {sr[i].max():8.4f}")
    drift = float(np.abs(sr.mean(axis=(1, 2)) - x.mean(axis=(1, 2))).max())
    print(f"\nmax per-band mean drift (sr vs input): {drift:.4f} "
          f"{'(OK — same domain)' if drift < 0.05 else '(LARGE — investigate!)'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = np.percentile(x[:3], args.stretch)
    hi = max(hi, lo + 1e-6)
    rgb = lambda a: np.clip((np.transpose(a[:3], (1, 2, 0)) - lo) / (hi - lo), 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, img, title in zip(
            axes, (rgb(x), rgb(bic), rgb(sr)),
            (f"Original 10 m ({CROP}px)", "Bicubic x4", "SR4RS x4 (pretrained port)")):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(f"{Path(args.image).stem}  crop r{args.row} c{args.col}  "
                 f"(shared {args.stretch[0]:g}-{args.stretch[1]:g}% stretch)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
