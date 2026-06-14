"""Evaluate a trained UNet checkpoint on the S2-ROSA test split.
"""

import argparse
import math
import os
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window

from unet.dataset import (
    DEFAULT_BANDS,
    MASK_PATH_COL,
    ROSADataModule,
    TILE_PATH_COL,
    ZONE_COL,
    read_metadata,
)
from unet.model import UNetLightning

KAGGLE_DATASET_DIR = "/kaggle/working/InstaRoadPrototype/dataset/s2rosa"


def _bands(value):
    return tuple(int(x) for x in value.split(","))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UNet on the S2-ROSA test split")
    p.add_argument("dataset_dir", nargs="?", default=KAGGLE_DATASET_DIR)
    p.add_argument("--checkpoint", default="checkpoints/unet_s2rosa_best.ckpt")
    p.add_argument("--output-dir", default="predictions/test_set")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bands", type=_bands, default=DEFAULT_BANDS)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--predict-cog",
        action="store_true",
        help="Run whole-tile inference and write prediction COGs + comparison figures.",
    )
    p.add_argument("--cog-output-dir", default="predictions/cog")
    p.add_argument(
        "--cog-zones",
        type=lambda s: [z for z in s.split(",") if z],
        default=None,
        help="Comma-separated zone_name list to predict (default: every tile).",
    )
    p.add_argument(
        "--max-cogs", type=int, default=None, help="Cap the number of tiles predicted."
    )
    p.add_argument(
        "--cog-prob",
        action="store_true",
        help="Write a float32 probability COG instead of a binary {0,1} road mask.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = ROSADataModule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bands=args.bands,
        image_size=args.image_size,
        seed=args.seed,
    )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    print(f"Loading checkpoint {args.checkpoint}...")
    model = UNetLightning.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()

    print("\nEvaluating model on test set...")
    metrics = evaluate_metrics(model, test_loader, device)
    print("Test set metrics:")
    for name, value in metrics.items():
        print(f"  - {name}: {value}")

    print("\nSaving comparative predictions...")
    save_predictions(model, test_loader, args.output_dir, device, save_comparison=True)
    print("Done! Test predictions saved.")

    if args.predict_cog:
        print("\nRunning whole-COG inference...")
        run_cog_inference(
            model,
            args.dataset_dir,
            args.cog_output_dir,
            device,
            bands=args.bands,
            image_size=args.image_size,
            threshold=args.threshold,
            zones=args.cog_zones,
            max_cogs=args.max_cogs,
            write_prob=args.cog_prob,
        )
        print("Done! Whole-COG predictions saved.")

def save_predictions(model, dataloader, output_dir, device, save_comparison=False):
    model.eval()
    with torch.no_grad():
        for images, masks, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)

            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(filenames)):
                pred_mask = preds[i].squeeze()

                if save_comparison:
                    # Extract the first three bands (C, H, W) -> (H, W, C)
                    img = images_np[i][:3].transpose(1, 2, 0)
                    true_mask = masks_np[i].squeeze()
                    
                    # Apply 2nd-98th percentile stretch
                    img_stretched = np.zeros_like(img)
                    for c in range(3):
                        band = img[:, :, c]
                        lo, hi = np.percentile(band, (2, 98))
                        if hi > lo:
                            # Stretch back to [0, 1] for matplotlib
                            img_stretched[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
                        else:
                            # Fallback
                            img_stretched[:, :, c] = np.clip(band, 0, 1)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(img_stretched)
                    axes[0].set_title("Original Image")
                    axes[0].axis("off")
                    
                    axes[1].imshow(true_mask, cmap="gray")
                    axes[1].set_title("True Road Label")
                    axes[1].axis("off")
                    
                    axes[2].imshow(pred_mask, cmap="gray")
                    axes[2].set_title("Predicted Road")
                    axes[2].axis("off")
                    
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(output_dir, f"comp_{filenames[i]}"),
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                else:
                    pred_uint8 = (pred_mask * 255).astype(np.uint8)
                    Image.fromarray(pred_uint8).save(os.path.join(output_dir, filenames[i]))


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


def predict_cog(
    model, tile_path, device, bands=DEFAULT_BANDS, image_size=256, normalize=True, batch_size=16
):
    """Run the model over every block window of a tile COG and stitch a full-size map.

    Preprocessing mirrors ``ROSADataset`` exactly (per-window resize + clip + per-image
    standardization) so each window matches the training input distribution.

    Returns ``(prob, profile)`` - a ``(H, W)`` float32 road-probability array aligned to
    the source raster, plus the source rasterio profile (CRS/transform preserved).
    """
    bands = list(bands)
    resize = A.Resize(image_size, image_size)
    model.eval()
    with rasterio.open(tile_path) as src:
        height, width = src.height, src.width
        block_h, block_w = src.block_shapes[0]
        profile = src.profile.copy()
        prob = np.zeros((height, width), dtype=np.float32)

        windows = [
            Window(
                c * block_w,
                r * block_h,
                min(block_w, width - c * block_w),
                min(block_h, height - r * block_h),
            )
            for r in range(math.ceil(height / block_h))
            for c in range(math.ceil(width / block_w))
        ]

        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch_wins = windows[start : start + batch_size]
                tensors = []
                for win in batch_wins:
                    image = src.read(bands, window=win).astype(np.float32)
                    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                    image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                    image = resize(image=image)["image"]
                    image = np.clip(image, 0.0, 1.0)
                    if normalize:
                        # Per-image, per-channel standardization (matches ROSADataset).
                        mean = image.mean(axis=(0, 1), keepdims=True)
                        std = image.std(axis=(0, 1), keepdims=True) + 1e-6
                        image = (image - mean) / std
                    tensors.append(image.transpose(2, 0, 1))

                batch = torch.from_numpy(np.ascontiguousarray(np.stack(tensors))).to(device)
                out = torch.sigmoid(model(batch))[:, 0].cpu().numpy()  # (B, image_size, image_size)

                for pred, win in zip(out, batch_wins):
                    h, w = int(win.height), int(win.width)
                    if (h, w) != (image_size, image_size):
                        # Edge block: resize the prediction back to the window's true size.
                        pred = A.Resize(h, w)(image=pred)["image"]
                    r0, c0 = int(win.row_off), int(win.col_off)
                    prob[r0 : r0 + h, c0 : c0 + w] = pred
    return prob, profile


def write_prediction_cog(prob, profile, out_path, threshold=0.5, write_prob=False):
    """Write the stitched prediction as a Cloud-Optimized GeoTIFF.

    ``write_prob`` -> float32 probabilities; otherwise a binary {0, 1} uint8 road mask
    (matching the ground-truth mask COG, nodata=0).
    """
    cog_profile = profile.copy()
    # The COG driver manages tiling/overviews itself; drop conflicting source keys.
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        cog_profile.pop(key, None)

    if write_prob:
        data = prob.astype(np.float32)
        cog_profile.update(driver="COG", dtype="float32", count=1, nodata=None, compress="DEFLATE")
    else:
        data = (prob > threshold).astype(np.uint8)
        cog_profile.update(driver="COG", dtype="uint8", count=1, nodata=0, compress="DEFLATE")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **cog_profile) as dst:
        dst.write(data, 1)


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


def run_cog_inference(
    model,
    dataset_dir,
    output_dir,
    device,
    bands=DEFAULT_BANDS,
    image_size=256,
    threshold=0.5,
    zones=None,
    max_cogs=None,
    write_prob=False,
):
    """Predict whole tiles end-to-end: write a prediction COG + comparison PNG per tile."""
    os.makedirs(output_dir, exist_ok=True)
    base = Path(dataset_dir)
    df = read_metadata(dataset_dir)
    tiles = df.drop_duplicates(subset=[TILE_PATH_COL])[[ZONE_COL, TILE_PATH_COL, MASK_PATH_COL]]
    if zones:
        tiles = tiles[tiles[ZONE_COL].isin(zones)]
    if max_cogs:
        tiles = tiles.head(max_cogs)

    if tiles.empty:
        print("No tiles matched the COG selection; nothing to do.")
        return

    for _, row in tiles.iterrows():
        zone = row[ZONE_COL]
        tile_path = str(base / row[TILE_PATH_COL])
        mask_path = str(base / row[MASK_PATH_COL])
        print(f"  zone {zone}: predicting whole tile...")
        prob, profile = predict_cog(
            model, tile_path, device, bands=bands, image_size=image_size
        )
        cog_out = os.path.join(output_dir, f"{zone}_pred.tif")
        comp_out = os.path.join(output_dir, f"{zone}_comparison.png")
        write_prediction_cog(prob, profile, cog_out, threshold=threshold, write_prob=write_prob)
        save_cog_comparison(tile_path, mask_path, prob, comp_out, threshold=threshold)
        print(f"    -> {cog_out}")
        print(f"    -> {comp_out}")


def evaluate_metrics(model, dataloader, device, threshold=0.5):
    """IoU/F1/precision/recall/accuracy for the road class over the test set."""
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
