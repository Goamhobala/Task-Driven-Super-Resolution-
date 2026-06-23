"""Evaluate a trained UNet checkpoint on the tiled S2-ROSA dataset.

Sliding-window inference: a torchgeo ``GridGeoSampler`` slides ``image_size``
windows over each tile COG, the model predicts each patch, and the predictions
are stitched back into a full-tile road map (overlapping patches are averaged
when ``--stride < image_size``). Each stitched tile is written as a 2-band COG
(binary mask + probability) plus a 3-panel comparison PNG.

Metrics are computed over the split with a dense ``GridGeoSampler`` loader (the
same patches the model would train on), reusing ``unet.geo_dataset``.

    python -m unet.inference <dataset_dir> --checkpoint ckpt.ckpt --split test
    # overlap-averaged seams:  --stride 128
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler

from sentinel2data.torchgeo_dataset import (
    WGS84,
    S2RosaImage,
    _split_paths,
    build_dataset,
)
from unet.geo_dataset import DEFAULT_BANDS, _band_names, _collate, _make_transform
from unet.model import UNetLightning

KAGGLE_DATASET_DIR = "/kaggle/working/InstaRoadPrototype/dataset/s2rosa"


def _bands(value):
    return tuple(int(x) for x in value.split(","))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UNet on the tiled S2-ROSA dataset")
    p.add_argument("dataset_dir", nargs="?", default=KAGGLE_DATASET_DIR)
    p.add_argument("--checkpoint", default="checkpoints/unet_s2rosa_best.ckpt")
    p.add_argument("--output-dir", default="predictions/tiled")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bands", type=_bands, default=DEFAULT_BANDS, help="e.g. 1,2,3")
    p.add_argument("--image-size", type=int, default=256, help="Sliding window edge (px).")
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding window stride (px); default = image-size (no overlap).",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-metrics", action="store_true", help="Skip the metrics pass.")
    p.add_argument(
        "--zones",
        type=lambda s: [z for z in s.split(",") if z],
        default=None,
        help="Comma-separated zone_name prefixes to predict (default: every tile).",
    )
    p.add_argument("--max-tiles", type=int, default=None, help="Cap tiles predicted.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band_names = _band_names(args.bands)
    normalize = not args.no_normalize

    print(f"Loading checkpoint {args.checkpoint}...")
    model = UNetLightning.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()

    if not args.no_metrics:
        print(f"\nEvaluating on the {args.split!r} split (dense grid)...")
        loader = _eval_loader(
            args.dataset_dir, args.split, band_names, args.image_size,
            args.batch_size, args.num_workers, normalize,
        )
        metrics = evaluate_metrics(model, loader, device, threshold=args.threshold)
        print("Metrics:")
        for name, value in metrics.items():
            print(f"  - {name}: {value}")

    print("\nSliding-window tile inference...")
    run_tiled_inference(
        model,
        args.dataset_dir,
        args.split,
        args.output_dir,
        device,
        bands=band_names,
        patch_size=args.image_size,
        stride=args.stride or args.image_size,
        threshold=args.threshold,
        normalize=normalize,
        batch_size=args.batch_size,
        zones=args.zones,
        max_tiles=args.max_tiles,
    )
    print("Done.")


# -- metrics loader --------------------------------------------------------
def _eval_loader(dataset_dir, split, band_names, patch_size, batch_size, num_workers, normalize):
    """Dense GridGeoSampler loader over a split, yielding (image, mask, names)."""
    ds = build_dataset(dataset_dir, split=split, bands=band_names, crs=WGS84)
    ds.transforms = _make_transform(normalize)
    sampler = GridGeoSampler(ds, size=patch_size, stride=patch_size)
    return DataLoader(
        ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
    )


# -- sliding-window stitch -------------------------------------------------
def predict_tile_sliding(
    model, image_path, device, bands, patch_size=256, stride=256,
    normalize=True, batch_size=16,
):
    """Slide a GridGeoSampler over one tile COG and stitch a full-size road map.

    Runs in the tile's **native CRS** (no warp) so patch geo-origins map to exact
    integer pixel offsets. Overlapping patches (stride < patch_size) are averaged.

    Returns ``(prob, profile)`` - a ``(H, W)`` float32 probability map aligned to
    the source raster + its rasterio profile (CRS/transform preserved).
    """
    imds = S2RosaImage(paths=[str(image_path)], bands=list(bands), crs=None)
    imds.transforms = _make_transform(normalize)  # same prep as training
    sampler = GridGeoSampler(imds, size=patch_size, stride=stride)
    loader = DataLoader(imds, sampler=sampler, batch_size=batch_size, collate_fn=stack_samples)

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        profile = src.profile.copy()

    prob = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            out = torch.sigmoid(model(images))[:, 0].cpu().numpy()  # (B, ps, ps)
            tfs = batch["transform"].cpu().numpy()                  # (B, 9) flat affine
            for pred, tf in zip(out, tfs):
                # patch affine = [a, b, c(xmin), d, e, f(ymax), 0, 0, 1]
                col = int(round((tf[2] - transform.c) / transform.a))
                row = int(round((tf[5] - transform.f) / transform.e))
                ph, pw = pred.shape
                # Clip in case a snapped edge patch overhangs the raster.
                ph = min(ph, height - row)
                pw = min(pw, width - col)
                prob[row : row + ph, col : col + pw] += pred[:ph, :pw]
                count[row : row + ph, col : col + pw] += 1.0

    count[count == 0] = 1.0
    return prob / count, profile


def run_tiled_inference(
    model, dataset_dir, split, output_dir, device, bands=DEFAULT_BANDS,
    patch_size=256, stride=256, threshold=0.5, normalize=True, batch_size=16,
    zones=None, max_tiles=None,
):
    """Stitch + write a prediction COG and comparison PNG for every split tile."""
    os.makedirs(output_dir, exist_ok=True)
    img_paths, msk_paths = _split_paths(dataset_dir, split)
    pairs = list(zip(img_paths, msk_paths))
    if zones:
        pairs = [(i, m) for i, m in pairs if any(Path(i).stem.startswith(z) for z in zones)]
    if max_tiles:
        pairs = pairs[:max_tiles]
    if not pairs:
        print("No tiles matched the selection; nothing to do.")
        return

    for image_path, mask_path in pairs:
        name = Path(image_path).stem
        print(f"  {name}: sliding window ({patch_size}px / stride {stride})...")
        prob, profile = predict_tile_sliding(
            model, image_path, device, bands, patch_size=patch_size,
            stride=stride, normalize=normalize, batch_size=batch_size,
        )
        cog_out = os.path.join(output_dir, f"{name}_pred.tif")
        comp_out = os.path.join(output_dir, f"{name}_comparison.png")
        write_prediction_cog(prob, profile, cog_out, threshold=threshold)
        save_cog_comparison(image_path, mask_path, prob, comp_out, threshold=threshold)
        print(f"    -> {cog_out}\n    -> {comp_out}")


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
    """Whole-tile three-panel figure: satellite RGB | ground-truth road | predicted road."""
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


def evaluate_metrics(model, dataloader, device, threshold=0.5):
    """IoU/F1/precision/recall/accuracy for the road class over a loader."""
    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0.0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = (torch.sigmoid(model(images)) > threshold).float()

            preds = preds.view(-1)
            masks = masks.view(-1)
            total_tp += torch.sum(preds * masks).item()
            total_fp += torch.sum(preds * (1 - masks)).item()
            total_fn += torch.sum((1 - preds) * masks).item()
            total_tn += torch.sum((1 - preds) * (1 - masks)).item()

    eps = 1e-6
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)

    return {
        "IoU": round(iou, 4),
        "F1 Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(accuracy, 4),
    }


if __name__ == "__main__":
    main()
