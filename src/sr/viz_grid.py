"""2xN thesis comparison grid: UNet inputs (top) vs predictions (bottom).

Layout (with the canonical five experiments):

    | Original 10m | R0 input (bicubic) | R1b input | R1a input | R2b input | R2a input |
    | GT mask      | R0 prediction      | R1b pred  | R1a pred  | R2b pred  | R2a pred  |

Column 1 is fixed (original crop / ground-truth mask); every further column is
one ``--exp`` in the order given. For each experiment the FULL
`JointSRUNetLightning` checkpoint is loaded and its forward pass is replayed
faithfully: top row shows the exact tensor its UNet consumed (the SR/bicubic
output in reflectance, AFTER that ckpt's own `sr_pad` pad+crop, BEFORE the
z-score adapter); bottom row shows sigmoid(logits) > threshold. Checkpoint
`sr_pad` and upsampler come from the saved hparams, so padded/unpadded ckpts
mix freely. All image panels share ONE percentile stretch computed from the
original crop (per-panel autoscaling would hide the contrast drift the figure
exists to show).

Simplest use — everything defaults from the (gitignored) src/sr/examples/
folder, which holds the ckpts (naming convention below), the example tile and
the SEN2SR weights; missing ckpts are skipped so partial grids render:

    python -m sr.viz_grid                       # convention mode, zero args
    python -m sr.viz_grid --row 256 --col 128 \
        --metrics R2a:iou=0.31,f1=0.44 --metrics R0:iou=0.29,f1=0.41

Test metrics are optional (not stored in ckpts) — given via --metrics they are
rendered under the prediction label. Explicit mode overrides everything:

    python -m sr.viz_grid --image <tile.tif> --sen2sr-dir <weights_dir> \
        --exp R0=<ckpt> --exp R1b=<ckpt> ...

Ground truth: `{tile}_mask.tif` beside the image (or --mask). A 10 m mask
(dims == tile) is nearest-upsampled x4 for display; a 2.5 m mask (dims == 4x)
is read at the scaled window. Missing -> empty GT panel with a warning.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from sr.model import REFLECTANCE_SCALE, JointSRUNetLightning, SEN2SR_BANDS

NODATA = -32768
CROP = 128  # SEN2SR's FFT mask pins the LR patch size

# Default (gitignored) folder holding ckpts + example tile + SEN2SR weights.
EXAMPLES_DIR = "src/sr/examples"

# Checkpoint naming convention in the examples folder, in COLUMN ORDER.
# Missing files are skipped with a warning, so partial grids render while the
# remaining experiments are still training.
CONVENTION = {
    "R0":  "unet_s2rosa_bicubic_best.ckpt",
    "R1b": "unet_s2rosa_pretrained_nopad_best.ckpt",
    "R1a": "unet_s2rosa_pretrained_pad_best.ckpt",
    "R2b": "unet_s2rosa_jointsr_nopad_best.ckpt",
    "R2a": "unet_s2rosa_jointsr_pad_best.ckpt",
}


# --------------------------------------------------------------------- inputs
def read_patch(image_path, row, col):
    """(4, 128, 128) float32 reflectance in [B4,B3,B2,B8] order."""
    with rasterio.open(image_path) as src:
        if src.height < row + CROP or src.width < col + CROP:
            raise SystemExit(f"crop exceeds the {src.height}x{src.width} tile")
        x = src.read(list(SEN2SR_BANDS),
                     window=Window(col, row, CROP, CROP)).astype("float32")
    x[x == NODATA] = 0.0
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x  # RAW values as stored (V2 COGs: already 0-1 reflectance)


def read_gt(image_path, mask_path, row, col, upscale=4):
    """(512, 512) binary GT + a panel label. No searching: the mask is
    `{stem}_mask.tif` beside the image (or --mask). Both mask resolutions are
    handled — a 10 m mask (dims == tile dims) is read at the native window and
    nearest-upsampled x4 for display; a 2.5 m mask (dims == 4x) is read at the
    scaled window."""
    p = Path(image_path)
    mask_path = Path(mask_path) if mask_path else p.parent / (p.stem + "_mask.tif")
    hw = CROP * upscale
    if not mask_path.exists():
        print(f"WARN: {mask_path.name} not found — GT panel left empty.")
        return np.zeros((hw, hw), dtype="float32"), "GT (missing)"
    with rasterio.open(image_path) as img:
        tile_dims = (img.height, img.width)
    with rasterio.open(mask_path) as src:
        if (src.height, src.width) == tile_dims:            # native 10 m mask
            m = (src.read(1, window=Window(col, row, CROP, CROP)) > 0)
            m = np.kron(m.astype("float32"), np.ones((upscale, upscale), "float32"))
            return m, "GT mask (10 m, nearest x4)"
        m = src.read(1, window=Window(col * upscale, row * upscale, hw, hw))
        return (m > 0).astype("float32"), "GT mask (2.5 m)"


# ---------------------------------------------------------------- experiments
def parse_exp(spec):
    """'NAME=path.ckpt[:iou=0.31,f1=0.44]' -> (name, path, {metric: value})."""
    name, rest = spec.split("=", 1)
    path, metrics = rest, {}
    m = re.match(r"^(.*?):((?:\w+=[\d.]+,?)+)$", rest)
    if m:
        path = m.group(1)
        metrics = {k: float(v) for k, v in
                   (kv.split("=") for kv in m.group(2).split(",") if kv)}
    return name.strip(), Path(path), metrics


@torch.no_grad()
def run_experiment(ckpt, sen2sr_dir, x, threshold, device):
    """Replay one checkpoint's forward. Returns (unet_input_hw3, pred_hw).

    unet_input = the reflectance tensor entering the z-score adapter (i.e.
    after this ckpt's own SR net + sr_pad pad/crop) — exactly what its UNet
    saw, up to normalisation."""
    model = JointSRUNetLightning.load_from_checkpoint(
        str(ckpt), map_location=device, sen2sr_dir=str(sen2sr_dir),
        # The restore below supplies the trained UNet weights anyway; skipping
        # the stage-1 warm-start init lets R6/R7 ckpts render on machines
        # without the stage-1 run dir.
        warm_start_unet=None).eval().to(device)
    t = torch.from_numpy(x)[None].float().to(device)   # RAW values (as the dataloader feeds)
    # Each ckpt carries its own raw->reflectance divisor (old ckpts default to
    # 10000.0, matching how they were trained; new runs on the V2 reflectance
    # datasets use 1.0). Replicate the SR stage with the ckpt's own scale.
    rs = float(getattr(model.hparams, "reflectance_scale", 10000.0))
    t_ref = t / rs
    p = int(model.hparams.sr_pad)
    t_sr = torch.nn.functional.pad(t_ref, (p,) * 4, mode="reflect") if p else t_ref
    hr = model.sr(t_sr)
    if p:
        q = p * int(model.hparams.upscale)
        hr = hr[..., q:-q, q:-q]
    logits = model(t)  # faithful end-to-end: forward divides by its own scale
    pred = (torch.sigmoid(logits)[0, 0] > threshold).float().cpu().numpy()
    return hr[0].cpu().numpy(), pred


# ------------------------------------------------------------------ rendering
def to_rgb(x, lo, hi):
    rgb = np.transpose(np.asarray(x)[:3], (1, 2, 0))  # [B4,B3,B2,...] = R,G,B
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", default=EXAMPLES_DIR,
                    help="folder with ckpts (naming convention), example tile and "
                         "SEN2SR weights — everything else defaults from here")
    ap.add_argument("--image", default=None,
                    help="V2 tile GeoTIFF (default: first .tif in --examples-dir)")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--exp", action="append", default=None, metavar="NAME=CKPT",
                    help="explicit experiments in column order; default: the "
                         "R0,R1b,R1a,R2b,R2a convention ckpts found in --examples-dir")
    ap.add_argument("--metrics", action="append", default=[], metavar="NAME:iou=..,f1=..",
                    help="optional test metrics rendered under a prediction, "
                         "e.g. --metrics R2a:iou=0.31,f1=0.44 (repeatable)")
    ap.add_argument("--sen2sr-dir", default=None,
                    help="SEN2SR weights dir (default: --examples-dir, which holds "
                         "model.safetensor + hard_constraint.safetensor)")
    ap.add_argument("--mask", default=None,
                    help="explicit HR GT raster (default: {tile}.parquet or "
                         "{tile}_mask.tif beside the image, or dataset masks_graph/)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98), metavar=("PLO", "PHI"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = Path(args.examples_dir)
    sen2sr_dir = args.sen2sr_dir or examples
    if args.image is None:
        tifs = sorted(t for t in examples.glob("*.tif")
                      if not t.stem.endswith("_mask"))
        if not tifs:
            raise SystemExit(f"no .tif found in {examples} — pass --image")
        args.image = str(tifs[0])
        print(f"image: {args.image}")

    # metrics: from --metrics (simple mode) and/or the --exp suffix syntax
    metric_map = {}
    for spec in args.metrics:
        name, _, kvs = spec.partition(":")
        metric_map[name.strip()] = {k: float(v) for k, v in
                                    (kv.split("=") for kv in kvs.split(",") if kv)}

    if args.exp:
        exps = [parse_exp(s) for s in args.exp]
    else:
        # convention mode: ALWAYS all five columns, in order — experiments whose
        # ckpt isn't there yet keep their slot and render as blank placeholders,
        # so partial figures stay column-aligned while runs finish.
        exps = [(name, examples / fname, {}) for name, fname in CONVENTION.items()]
    exps = [(n, c, {**m, **metric_map.get(n, {})}) for n, c, m in exps]

    x = read_patch(args.image, args.row, args.col)
    gt, gt_label = read_gt(args.image, args.mask, args.row, args.col)
    lo, hi = np.percentile(x[:3], args.stretch)
    hi = max(hi, lo + 1e-6)
    n = 1 + len(exps)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 7.0))

    axes[0, 0].imshow(to_rgb(x, lo, hi), interpolation="nearest")
    axes[0, 0].set_title(f"Original 10 m ({CROP}px)", fontsize=10)
    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(gt_label, fontsize=10)

    for j, (name, ckpt, metrics) in enumerate(exps, start=1):
        if not Path(ckpt).exists():
            axes[0, j].set_title(f"{name} (pending)", fontsize=10, color="0.55")
            axes[1, j].set_title(f"{name} prediction (pending)",
                                 fontsize=10, color="0.55")
            print(f"{name}: ckpt not found ({Path(ckpt).name}) — column left blank")
            continue
        unet_in, pred = run_experiment(ckpt, sen2sr_dir, x,
                                       args.threshold, args.device)
        axes[0, j].imshow(to_rgb(unet_in, lo, hi))
        axes[0, j].set_title(f"{name} UNet input", fontsize=10)
        axes[1, j].imshow(pred, cmap="gray", vmin=0, vmax=1)
        sub = "  ".join(f"{k.upper()} {v:.3f}" for k, v in metrics.items())
        axes[1, j].set_title(f"{name} prediction" + (f"\n{sub}" if sub else ""),
                             fontsize=10)
        print(f"{name}: pred road fraction {pred.mean():.4f}"
              + (f"  ({sub})" if sub else ""))

    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle(f"{Path(args.image).stem}  crop r{args.row} c{args.col}  "
                 f"(shared {args.stretch[0]:g}-{args.stretch[1]:g}% stretch; "
                 f"threshold {args.threshold})", fontsize=10)
    fig.tight_layout()
    out = args.out or f"{Path(args.image).stem}_grid.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
