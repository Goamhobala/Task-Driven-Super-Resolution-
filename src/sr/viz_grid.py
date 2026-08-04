"""2x4 comparison grid: SR inputs (top) vs predictions (bottom), on ONE crop.

Sibling of `viz_single`, sharing its checkpoint-replay and native-GT machinery.
Four columns on ONE 128 px crop:

    | Original (nearest x4) | Bicubic x4   | SEN2SR (original) | SEN2SR (finetuned) |
    | Native 2.5 m GT       | Bicubic pred | (no prediction)   | finetuned pred     |

Column 1 is the reference: the raw 10 m crop nearest-upsampled x4 (blocky, so it
reads as the un-enhanced input) over the native 2.5 m ground-truth mask. Bicubic
and "SEN2SR (finetuned)" are real `JointSRUNetLightning` checkpoints whose forward
is replayed faithfully — top row is the exact reflectance tensor the UNet consumed
(its SR net + `sr_pad` pad/crop, BEFORE the z-score adapter), bottom row is
sigmoid(logits) > threshold. "SEN2SR (original)" is NOT a checkpoint: it is the
crop passed through the pretrained (un-finetuned) SEN2SR weights in --sr-dir — the
finetuned ckpt's starting point, via that ckpt's own pad/crop so the two SR panels
are directly comparable. It has no matching UNet, so its prediction panel is left
blank. All image panels share ONE percentile stretch computed from the original
crop.

The raw->reflectance divisor is a property of the TILE, not each ckpt: the SR
nets want 0-1 reflectance and forward re-multiplies the SR output by the ckpt's
scale before the z-score adapter (so scale and the baked band stats are coupled).
Each crop is therefore fed to every ckpt in that ckpt's OWN raw units
(reflectance x its scale), reproducing its training pipeline exactly, so DN-era
(scale 10000) and 0-1 (scale 1.0) ckpts stay directly comparable.

Zero-argument default (everything from the gitignored src/sr/examples/ folder):

    python -m sr.viz_grid                        # -> {tile}_grid.png

Ground truth defaults to a NATIVE 2.5 m mask (`{tile}_mask_high.tif`) so the GT
panel matches what the HR UNet is scored on. If that cache is absent it is
rasterised fresh from the tile's road-graph parquet (--graph, else
`{tile}_graph.parquet` beside the image, else --dataset-dir's
masks_graph/{tile}.parquet) and written back. With no graph found we fall back
to the 10 m `{tile}_mask.tif` (bicubic x4); --mask forces a raster. Override any
piece:

    python -m sr.viz_grid --image <tile.tif> --row 224 --col 288 --threshold 0.4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio import Affine
from rasterio.windows import Window

from sr.model import REFLECTANCE_SCALE, JointSRUNetLightning, SEN2SR_BANDS

NODATA = -32768
CROP = 128  # SEN2SR's FFT mask pins the LR patch size

# Default (gitignored) folder holding ckpts + example tile + SEN2SR weights.
EXAMPLES_DIR = "src/sr/examples"
DEFAULT_IMAGE = "Skukuza_r2_c2.tif"   # zero-arg tile, matching viz_single

# Native 2.5 m ground truth. A tile's 10 m `_mask.tif` (or none) is not what the
# HR UNet is scored against — the model sees 2.5 m, so the GT panel should too.
# When no `{stem}_mask_high.tif` is cached beside the image, we rasterise it
# fresh from the tile's road-graph parquet (the buffered centrelines the dataset
# was built from) at 4x resolution and cache it. Graphs are looked up under this
# ROSA dataset dir ({split}/masks_graph/{stem}.parquet); override with
# --dataset-dir, or point --graph straight at a parquet. The example tiles come
# from RandomSampling110zones.
MASK_DATASET_DIR = "/Volumes/MAC_KIOXIA/Data/ROSA_RandomSampling110zones"
HR_MASK_SUFFIX = "_mask_high.tif"

# The two checkpoints behind the grid's four columns. "SEN2SR (original)" is NOT
# a checkpoint — it is the crop through the pretrained (un-finetuned) SEN2SR
# weights in --sr-dir (the finetuned ckpt's starting point), rendered as the SR
# panel only, with no prediction. FINETUNED_CKPT supplies both the finetuned SR
# panel + prediction and, via its pristine starting weights, the original SR
# panel. Missing files render as a blank, labelled column.
BICUBIC_CKPT = "unet_s2rosa_bicubic.ckpt"
FINETUNED_CKPT = "unet_s2rosa_jointsr_sen2sr_best.ckpt"


# --------------------------------------------------------------------- inputs
def read_patch(image_path, row, col, crop=CROP):
    """(4, crop, crop) float32 reflectance in [B4,B3,B2,B8] order."""
    with rasterio.open(image_path) as src:
        if src.height < row + crop or src.width < col + crop:
            raise SystemExit(f"crop exceeds the {src.height}x{src.width} tile")
        x = src.read(list(SEN2SR_BANDS),
                     window=Window(col, row, crop, crop)).astype("float32")
    x[x == NODATA] = 0.0
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x  # RAW values as stored (V2 COGs: already 0-1 reflectance)


def _find_graph_parquet(image_path, dataset_dir):
    """The road-graph parquet the tile was cut from — `{split}/masks_graph/
    {stem}.parquet` under `dataset_dir` (train/val/test searched). None if not
    found (or dataset_dir missing)."""
    if not dataset_dir:
        return None
    stem = Path(image_path).stem
    for split in ("train", "val", "test"):
        p = Path(dataset_dir) / split / "masks_graph" / f"{stem}.parquet"
        if p.exists():
            return p
    return None


def _rasterize_hr_mask(image_path, graph_path, out_path, upscale=4):
    """Rasterise the whole tile's buffered centrelines at `upscale`x resolution
    (10 m -> 2.5 m) and cache it as a GeoTIFF beside the image. Reuses the
    dataloader's own `_graph_mask` so the panel is the exact HR label the model
    is trained/scored on; that helper is square-only, which the ROSA tiles are."""
    from sentinel2data.dataset.upscale_dataset import _graph_mask
    from sentinel2data.generator.io import write_mask_cog

    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        if h != w:
            raise ValueError(
                f"{Path(image_path).name} is {h}x{w}; the HR mask rasteriser is "
                "square-only (ROSA tiles are square).")
        mask = _graph_mask(str(graph_path), src, Window(0, 0, w, h),
                           h * upscale, upscale)
        profile = src.profile.copy()
        hr_transform = src.transform * Affine.scale(1.0 / upscale)
    write_mask_cog(out_path, (mask > 0).astype("uint8"), profile,
                   transform=hr_transform, tiled=False)
    return Path(out_path)


def _read_mask_file(image_path, mask_path, row, col, upscale, hw):
    """Read an existing mask beside the image at the crop window. A 10 m mask
    (dims == tile dims) is read native and bicubically upsampled x4; a 2.5 m mask
    (dims == 4x) is read at the scaled window. Missing -> empty panel."""
    if not mask_path.exists():
        print(f"WARN: {mask_path.name} not found — GT panel left empty.")
        return np.zeros((hw, hw), dtype="float32"), "GT (missing)"
    crop = hw // upscale
    with rasterio.open(image_path) as img:
        tile_dims = (img.height, img.width)
    with rasterio.open(mask_path) as src:
        if (src.height, src.width) == tile_dims:            # native 10 m mask
            m = (src.read(1, window=Window(col, row, crop, crop)) > 0)
            t = torch.from_numpy(m.astype("float32"))[None, None]
            m = torch.nn.functional.interpolate(
                t, scale_factor=upscale, mode="bicubic", align_corners=False,
            ).clamp_(0, 1)[0, 0].numpy()
            return m, "GT mask (10 m, bicubic x4)"
        m = src.read(1, window=Window(col * upscale, row * upscale, hw, hw))
        return (m > 0).astype("float32"), "GT mask (2.5 m)"


def read_gt(image_path, mask_path, row, col, upscale=4, graph_path=None,
            dataset_dir=MASK_DATASET_DIR, crop=CROP):
    """(crop*upscale, crop*upscale) binary GT + a panel label.

    Default source is a NATIVE 2.5 m mask cached beside the image as
    `{stem}_mask_high.tif`, read at the scaled window. When that cache is
    missing, it is rasterised fresh from the tile's road-graph parquet (`--graph`,
    else `{stem}_graph.parquet` beside the image, else `dataset_dir`'s
    `masks_graph/{stem}.parquet`) and written back, so the GT panel is the exact
    2.5 m label the HR UNet is scored on rather than a bicubic blow-up. With no
    graph found we fall back to the 10 m `{stem}_mask.tif` (bicubic x4). An
    explicit `mask_path` always wins and is read as-is."""
    p = Path(image_path)
    hw = crop * upscale

    # Explicit --mask: honour it verbatim, no HR generation.
    if mask_path:
        return _read_mask_file(image_path, Path(mask_path), row, col, upscale, hw)

    hr_path = p.parent / (p.stem + HR_MASK_SUFFIX)
    if not hr_path.exists():
        # Resolve a road-graph parquet, most-specific first: explicit --graph,
        # a deliberately-placed `{stem}_graph.parquet` beside the image, then the
        # dataset default's masks_graph/.
        beside = p.parent / (p.stem + "_graph.parquet")
        graph = (Path(graph_path) if graph_path
                 else beside if beside.exists()
                 else _find_graph_parquet(p, dataset_dir))
        if graph and Path(graph).exists():
            print(f"generating native {upscale}x GT mask: {Path(graph).name} "
                  f"-> {hr_path.name}")
            _rasterize_hr_mask(p, graph, hr_path, upscale)
        else:
            print(f"WARN: no road-graph parquet for {p.stem} "
                  f"(looked in --graph / {p.stem}_graph.parquet / {dataset_dir}) "
                  "— falling back to the 10 m mask.")

    if hr_path.exists():
        with rasterio.open(hr_path) as src:
            m = src.read(1, window=Window(col * upscale, row * upscale, hw, hw))
        return (m > 0).astype("float32"), "GT mask (2.5 m, native)"

    # Fallback: the conventional 10 m mask beside the image.
    return _read_mask_file(image_path, p.parent / (p.stem + "_mask.tif"),
                           row, col, upscale, hw)


# ------------------------------------------------------------------ checkpoint
def nearest_x4(x):
    """Raw crop nearest-neighbour upsampled x4 — blocky, so the reference panel
    reads as the un-enhanced 10 m input on the same 512 px grid as the SR panels."""
    a = np.asarray(x)
    return np.repeat(np.repeat(a, 4, axis=1), 4, axis=2)


@torch.no_grad()
def run_checkpoint(ckpt, sr_dir, x, threshold, device, tile_scale, pristine=False):
    """Replay one checkpoint's forward on crop `x`. Returns (sr_chw, pred_hw,
    pristine_sr_chw); `pristine_sr_chw` is None unless `pristine` is set.

    sr = the reflectance tensor its UNet consumed (this ckpt's SR net + sr_pad
    pad/crop, BEFORE the z-score adapter); pred = sigmoid(logits) > threshold.
    pristine_sr = the SAME crop through the ckpt's UN-finetuned SR weights (loaded
    fresh from `sr_dir` via the identical pad/crop) — the "original" SR output,
    directly comparable to the finetuned one.

    `x` holds reflectance*`tile_scale` (the tile's storage units). A ckpt expects
    raw = reflectance*its OWN scale, because forward re-multiplies the SR output
    by that scale before the z-score adapter — scale and the baked band stats are
    coupled, so it cannot be overridden. We convert the crop into each ckpt's raw
    units, reproducing its training pipeline exactly, so DN-era and 0-1 ckpts are
    all replayed faithfully and stay comparable.
    """
    # map_location="cpu", then .to(device): deserialising straight onto MPS puts
    # the WHOLE ckpt there — incl. Adam moments viz never uses. warm_start_unet
    # =None skips the stage-1 init (the restore supplies the trained UNet anyway)
    # so ckpts render on machines without the stage-1 run dir.
    model = JointSRUNetLightning.load_from_checkpoint(
        str(ckpt), map_location="cpu", sen2sr_dir=str(sr_dir),
        warm_start_unet=None).eval().to(device)
    if getattr(model.hparams, "reflectance_scale", 10000.0) is None:
        model.hparams.reflectance_scale = 1.0   # some s2rosa ckpts saved it None
    ckpt_scale = float(model.hparams.reflectance_scale)
    p = int(model.hparams.sr_pad)
    up = int(model.hparams.upscale)

    reflectance = torch.from_numpy(x)[None].float().to(device) / tile_scale
    raw = reflectance * ckpt_scale                     # this ckpt's expected input

    def sr_img(sr_module):
        """One SR net's output for this crop, via the ckpt's own pad/crop."""
        t_sr = (torch.nn.functional.pad(reflectance, (p,) * 4, mode="reflect")
                if p else reflectance)
        hr = sr_module(t_sr)
        if p:
            q = p * up
            hr = hr[..., q:-q, q:-q]
        return hr[0].cpu().numpy()

    sr = sr_img(model.sr)
    logits = model(raw)   # faithful end-to-end: forward divides by its own scale
    pred = (torch.sigmoid(logits)[0, 0] > threshold).float().cpu().numpy()

    pristine_sr = None
    if pristine:
        # Lazy import avoids the viz_single<->viz_grid import cycle.
        from sr.viz_single import load_pristine_sr
        pmod = load_pristine_sr(model.hparams.upsampler, Path(sr_dir), p).eval().to(device)
        pristine_sr = sr_img(pmod)
    return sr, pred, pristine_sr


# ------------------------------------------------------------------ rendering
def to_rgb(x, lo, hi):
    rgb = np.transpose(np.asarray(x)[:3], (1, 2, 0))  # [B4,B3,B2,...] = R,G,B
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", default=EXAMPLES_DIR,
                    help="folder with the ckpts, example tile and SEN2SR weights — "
                         "everything else defaults from here")
    ap.add_argument("--image", default=None,
                    help=f"V2 tile GeoTIFF (default: {DEFAULT_IMAGE})")
    ap.add_argument("--row", type=int, default=None,
                    help="top of the 128 px crop (default: centred in the tile)")
    ap.add_argument("--col", type=int, default=None,
                    help="left of the 128 px crop (default: centred in the tile)")
    ap.add_argument("--sr-dir", default=None,
                    help="SEN2SR weights dir (default: --examples-dir, which holds "
                         "model.safetensor + hard_constraint.safetensor)")
    ap.add_argument("--mask", default=None,
                    help="explicit GT raster read as-is (overrides the native-HR "
                         "logic; 10 m -> bicubic x4, 2.5 m -> scaled window)")
    ap.add_argument("--graph", default=None,
                    help="road-graph parquet to rasterise the native 2.5 m GT "
                         "from (default: {tile}_graph.parquet beside the image, "
                         "else --dataset-dir's masks_graph/{tile}.parquet)")
    ap.add_argument("--dataset-dir", default=MASK_DATASET_DIR,
                    help="ROSA dataset root holding {split}/masks_graph/ used to "
                         "find a tile's road-graph parquet for the native 2.5 m GT")
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
    sr_dir = Path(args.sr_dir) if args.sr_dir else examples
    image = Path(args.image) if args.image else examples / DEFAULT_IMAGE
    if not image.exists():
        raise SystemExit(f"image not found: {image}")

    # Centre the crop in the tile unless an explicit row/col was given (matches
    # viz_single), so the default panels show the middle of the scene.
    with rasterio.open(image) as src:
        h, w = src.height, src.width
    row = args.row if args.row is not None else max(0, (h - CROP) // 2)
    col = args.col if args.col is not None else max(0, (w - CROP) // 2)

    x = read_patch(image, row, col)
    gt, gt_label = read_gt(image, args.mask, row, col,
                           graph_path=args.graph, dataset_dir=args.dataset_dir)
    lo, hi = np.percentile(x[:3], args.stretch)
    hi = max(hi, lo + 1e-6)
    # Storage units of THIS tile (V2 COGs already hold 0-1 reflectance -> 1.0;
    # DN COGs hold reflectance*10000). run_checkpoint re-expresses the crop in
    # each ckpt's own raw units from this.
    tile_scale = 1.0 if float(np.nanmax(x)) <= 1.5 else REFLECTANCE_SCALE

    fig, axes = plt.subplots(2, 4, figsize=(3.3 * 4, 7.4))

    # Col 0 — reference: nearest-x4 original over the native 2.5 m GT.
    axes[0, 0].imshow(to_rgb(nearest_x4(x), lo, hi), interpolation="nearest")
    axes[0, 0].set_title("Original 10 m (nearest x4)", fontsize=10)
    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(gt_label, fontsize=10)

    # Col 1 — bicubic ckpt: SR input + prediction.
    bicubic = examples / BICUBIC_CKPT
    if bicubic.exists():
        sr, pred, _ = run_checkpoint(bicubic, sr_dir, x, args.threshold,
                                     args.device, tile_scale)
        axes[0, 1].imshow(to_rgb(sr, lo, hi))
        axes[0, 1].set_title("Bicubic x4 -> UNet input", fontsize=10)
        axes[1, 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[1, 1].set_title(f"Prediction (road frac {pred.mean():.3f})", fontsize=10)
        print(f"Bicubic x4: pred road fraction {pred.mean():.4f}")
    else:
        axes[0, 1].set_title("Bicubic x4 (missing)", fontsize=10, color="0.55")
        print(f"Bicubic x4: ckpt not found ({BICUBIC_CKPT}) — column left blank")

    # Cols 2 & 3 — the finetuned ckpt supplies BOTH the finetuned SR + prediction
    # (col 3) and, from its pristine starting weights, the "original" SEN2SR SR
    # panel (col 2). Col 2 has no matching UNet, so its prediction is left blank.
    finetuned = examples / FINETUNED_CKPT
    if finetuned.exists():
        sr_ft, pred_ft, sr_orig = run_checkpoint(
            finetuned, sr_dir, x, args.threshold, args.device, tile_scale,
            pristine=True)
        axes[0, 2].imshow(to_rgb(sr_orig, lo, hi))
        axes[0, 2].set_title("SEN2SR (original) -> UNet input", fontsize=10)
        axes[0, 3].imshow(to_rgb(sr_ft, lo, hi))
        axes[0, 3].set_title("SEN2SR (finetuned) -> UNet input", fontsize=10)
        axes[1, 3].imshow(pred_ft, cmap="gray", vmin=0, vmax=1)
        axes[1, 3].set_title(f"Prediction (road frac {pred_ft.mean():.3f})", fontsize=10)
        print(f"SEN2SR (finetuned): pred road fraction {pred_ft.mean():.4f}")
    else:
        axes[0, 2].set_title("SEN2SR (original) (missing)", fontsize=10, color="0.55")
        axes[0, 3].set_title("SEN2SR (finetuned) (missing)", fontsize=10, color="0.55")
        print(f"SEN2SR: ckpt not found ({FINETUNED_CKPT}) — columns left blank")

    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle(f"{image.stem}  crop r{row} c{col}  "
                 f"(shared {args.stretch[0]:g}-{args.stretch[1]:g}% stretch; "
                 f"threshold {args.threshold})", fontsize=10)
    fig.tight_layout()
    out = args.out or f"{image.stem}_grid.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
