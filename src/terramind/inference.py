import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A

from terramind.dataset import SentinelRoadsDataset, sentinel2_data_partition
from terramind.model import build_model


def main():
    BASE_DIR    = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    IMG_DIR     = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR    = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    # Fine-tuned model saved by train.py (backbone + decoder + head weights)
    FINETUNED_CKPT   = '/kaggle/working/terramind_v1_base_roads_finetuned.pth'
    PREDICTIONS_PATH = '/kaggle/working/predictions/terramind_test_set'
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_list = sentinel2_data_partition(BASE_DIR)
    transform = A.Compose([A.Resize(256, 256)])

    test_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, test_list, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Build architecture (no backbone ckpt needed — fine-tuned state dict covers all weights)
    print("Initialising TerraMind model architecture...")
    model = build_model().to(device)

    print(f"Loading fine-tuned weights from {FINETUNED_CKPT}...")
    model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device))
    model.eval()

    print("\nEvaluating TerraMind on Test Set...")
    test_metrics = evaluate_metrics(model, test_loader, device)

    print("Test Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"  - {metric}: {value}")

    print("\nSaving comparative predictions...")
    save_predictions(model, test_loader, PREDICTIONS_PATH, device, save_comparison=True)
    print("Done! Test predictions saved.")


def save_predictions(model, dataloader, output_dir, device, save_comparison=False):
    model.eval()
    with torch.no_grad():
        for images, masks, filenames in dataloader:
            images = images.to(device)

            outputs = model({"RGB": images})
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float().cpu().numpy()

            images_np = images.cpu().numpy()
            masks_np  = masks.cpu().numpy()

            for i in range(len(filenames)):
                pred_mask = preds[i].squeeze()

                if save_comparison:
                    img       = images_np[i].transpose(1, 2, 0)
                    true_mask = masks_np[i].squeeze()

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(img)
                    axes[0].set_title("Original Image")
                    axes[0].axis("off")

                    axes[1].imshow(true_mask, cmap="gray")
                    axes[1].set_title("True Road Label")
                    axes[1].axis("off")

                    axes[2].imshow(pred_mask, cmap="gray")
                    axes[2].set_title("Predicted Road (TerraMind)")
                    axes[2].axis("off")

                    plt.tight_layout()
                    out_path = os.path.join(output_dir, f"comp_{filenames[i]}")
                    plt.savefig(out_path, bbox_inches="tight")
                    plt.close(fig)
                else:
                    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                    out_path = os.path.join(output_dir, filenames[i])
                    Image.fromarray(pred_mask_uint8).save(out_path)


def evaluate_metrics(model, dataloader, device, threshold=0.5):
    model.eval()

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model({"RGB": images})
            preds   = torch.sigmoid(outputs)
            preds   = (preds > threshold).float()

            preds = preds.view(-1)
            masks = masks.view(-1)

            total_tp += torch.sum(preds * masks).item()
            total_fp += torch.sum(preds * (1 - masks)).item()
            total_fn += torch.sum((1 - preds) * masks).item()
            total_tn += torch.sum((1 - preds) * (1 - masks)).item()

    eps = 1e-6

    iou       = total_tp / (total_tp + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall    = total_tp / (total_tp + total_fn + eps)
    f1        = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    accuracy  = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)

    return {
        "IoU":       round(iou, 4),
        "F1 Score":  round(f1, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "Accuracy":  round(accuracy, 4),
    }


if __name__ == "__main__":
    main()
