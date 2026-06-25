"""Evaluate a trained UNet checkpoint on S2-ROSA-V2 (native CRS, no torchgeo).

Per zone: native-pixel sliding window (zero-padded edges) + cosine-blended stitch
(:func:`unet.sliding.predict_zone`) -> one probability raster in the zone's native
CRS. Metrics are scored **once per ground pixel** over the stitched raster (global
TP/FP/FN), never per-overlapping-tile averaging. Each zone is written as a 2-band
COG (binary + probability) plus a 3-panel comparison PNG.

    python -m unet.inference <dataset_dir> --checkpoint ckpt.ckpt --split test
    # non-overlapping windows:  --overlap 0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch

from unet.model import UNetLightning
from unet.patch_dataset import DEFAULT_BANDS
from unet.sliding import predict_zone

KAGGLE_DATASET_DIR = "/kaggle/working/InstaRoadPrototype/dataset/s2rosa"


def _bands(value):
    return tuple(int(x) for x in value.split(","))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UNet on S2-ROSA-V2 (stitched).")
    p.add_argument("dataset_dir", nargs="?", default=KAGGLE_DATASET_DIR)
    p.add_argument("--checkpoint", default="checkpoints/unet_s2rosa_best.ckpt")
    p.add_argument("--output-dir", default="predictions")
    p.add_argument("--split", default="test")
    p.add_argument("--bands", type=_bands, default=DEFAULT_BANDS, help="e.g. 21,22,23")
    p.add_argument("--image-size", type=int, default=256, help="Sliding window edge (px).")
    p.add_argument("--overlap", type=int, default=128, help="Window overlap (px); 0 = none.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-metrics", action="store_true", help="Skip the metrics pass.")
    p.add_argument("--no-write", action="store_true", help="Skip writing COGs/PNGs.")
    p.add_argument(
        "--zones",
        type=lambda s: [z for z in s.split(",") if z],
        default=None,
        help="Comma-separated zone_name prefixes (default: every zone).",
    )
    p.add_argument("--max-zones", type=int, default=None, help="Cap zones predicted.")
    return p.parse_args()


def _zone_pairs(dataset_dir, split, zones=None, max_zones=None):
    """(image_path, mask_path) per zone in a split, from its splits CSV."""
    df = pd.read_csv(Path(dataset_dir) / "splits" / f"{split}.csv")
    if zones:
        df = df[df["zone_name"].astype(str).apply(lambda z: any(z.startswith(p) for p in zones))]
    if max_zones:
        df = df.head(max_zones)
    root = Path(dataset_dir)
    return [(str(root / r.image_path), str(root / r.mask_path)) for r in df.itertuples(index=False)]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = not args.no_normalize

    print(f"Loading checkpoint {args.checkpoint}...")
    model = UNetLightning.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()

    pairs = _zone_pairs(args.dataset_dir, args.split, args.zones, args.max_zones)
    if not pairs:
        print("No zones matched the selection; nothing to do.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    tp = fp = fn = tn = 0.0
    for image_path, mask_path in pairs:
        name = Path(image_path).stem
        print(f"  {name}: stitch (size {args.image_size}, overlap {args.overlap})...")
        prob, profile = predict_zone(
            model, image_path, args.bands, size=args.image_size,
            overlap=args.overlap, normalize=normalize,
        )
        if not args.no_metrics:
            with rasterio.open(mask_path) as m:
                gt = m.read(1) > 0
            pred = prob > args.threshold
            tp += float(np.sum(pred & gt)); fp += float(np.sum(pred & ~gt))
            fn += float(np.sum(~pred & gt)); tn += float(np.sum(~pred & ~gt))
        if not args.no_write:
            cog_out = os.path.join(args.output_dir, f"{name}_pred.tif")
            comp_out = os.path.join(args.output_dir, f"{name}_comparison.png")
            write_prediction_cog(prob, profile, cog_out, threshold=args.threshold)
            save_cog_comparison(image_path, mask_path, prob, comp_out, threshold=args.threshold)
            print(f"    -> {cog_out}\n    -> {comp_out}")

    if not args.no_metrics:
        eps = 1e-6
        metrics = {
            "IoU": tp / (tp + fp + fn + eps),
            "F1 Score": 2 * tp / (2 * tp + fp + fn + eps),
            "Precision": tp / (tp + fp + eps),
            "Recall": tp / (tp + fn + eps),
            "Accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        }
        print("\nStitched metrics (scored once per ground pixel):")
        for k, v in metrics.items():
            print(f"  - {k}: {round(float(v), 4)}")
    print("Done.")


# -- writers ---------------------------------------------------------------
def write_prediction_cog(prob, profile, out_path, threshold=0.5):
    """Write the stitched prediction as a 2-band COG (band1 binary, band2 prob)."""
    cog_profile = profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "interleave", "predictor"):
        cog_profile.pop(key, None)

    prob = prob.astype(np.float32)
    binary = (prob > threshold).astype(np.float32)
    data = np.stack([binary, prob])  # (2, H, W)
    cog_profile.update(driver="COG", dtype="float32", count=2, nodata=None, compress="DEFLATE")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **cog_profile) as dst:
        dst.write(data)
        dst.set_band_description(1, "road_mask_binary")
        dst.set_band_description(2, "road_probability")


def _stretch_rgb(img, percentile_range=(2, 98)):
    """Percentile contrast-stretch a (3, H, W) float array to a (H, W, 3) uint8 image."""
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        band = img[i]
        valid = band[band > 0]
        if valid.size:
            lo, hi = np.percentile(valid, list(percentile_range))
            if hi > lo:
                out[i] = np.clip((np.clip(band, lo, hi) - lo) / (hi - lo), 0, 1)
    return (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8)


def save_cog_comparison(tile_path, mask_path, prob, out_path, threshold=0.5):
    """Whole-zone three-panel figure: satellite RGB | ground-truth | predicted road."""
    with rasterio.open(tile_path) as src:
        rgb = _stretch_rgb(src.read([1, 2, 3]).astype(np.float32))
    with rasterio.open(mask_path) as msrc:
        true_mask = (msrc.read(1) > 0).astype(np.float32)
    pred_mask = (prob > threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(rgb)
    axes[0].set_title("Satellite (RGB)")
    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth Road")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Road")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
