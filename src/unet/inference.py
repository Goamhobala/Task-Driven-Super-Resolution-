"""Evaluate a trained UNet checkpoint on the S2-ROSA test split.

Local:
    python inference.py /Volumes/MacOSFiles/S2ROSA --checkpoint checkpoints/unet_s2rosa_best.ckpt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from unet.dataset import DEFAULT_BANDS, ROSADataModule
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
                    # Use the first three bands (RGB) for display.
                    img = images_np[i][:3].transpose(1, 2, 0)
                    true_mask = masks_np[i].squeeze()

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(np.clip(img, 0, 1))
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
