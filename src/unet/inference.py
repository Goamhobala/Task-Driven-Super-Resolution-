import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from unet.model import build_model
import albumentations as A

def main():
    # Input path
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2'
    # DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_1024'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    # IMG_DIR = os.path.join(DATASET_DIR, 'images_1024')
    # MASK_DIR = os.path.join(DATASET_DIR, 'clean_masks')
    IMG_DIR = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    CHECKPOINT_PATH = '/kaggle/working/unetplusplus_resnet50_roads.pth'
    PREDICTIONS_PATH = '/kaggle/working/predictions/test_set'

    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Setup
    _, _, test_list = sentinel2_data_partition(BASE_DIR)
    # transform = A.Compose([A.Resize(1024, 1024)])
    transform = A.Compose([A.Resize(256, 256)])

    test_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, test_list, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Load the Model
    print("Initializing model architecture...")
    model = build_model().to(device)

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    # map_location ensures it loads safely even if moving from a GPU to a CPU instance
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # Put the model in evaluation mode (disables dropout, fixes batch norm)
    model.eval()

    # Run Inference
    print("\nEvaluating Model on Test Set...")
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
            outputs = model(images)

            # Apply sigmoid to convert logits to probabilities, threshold at 0.5
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float().cpu().numpy()

            # Move images and true masks to CPU and convert to numpy for visualization
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(filenames)):
                pred_mask = preds[i].squeeze()

                if save_comparison:
                    # 1. Format the Original Image: (C, H, W) -> (H, W, C) for Matplotlib
                    img = images_np[i].transpose(1, 2, 0)

                    # 2. Format the True Mask
                    true_mask = masks_np[i].squeeze()

                    # 3. Create a 1x3 subplot
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(img)
                    axes[0].set_title("Original Image")
                    axes[0].axis("off")

                    axes[1].imshow(true_mask, cmap='gray')
                    axes[1].set_title("True Road Label")
                    axes[1].axis("off")

                    axes[2].imshow(pred_mask, cmap='gray')
                    axes[2].set_title("Predicted Road")
                    axes[2].axis("off")

                    plt.tight_layout()

                    # Save the composite figure
                    out_path = os.path.join(output_dir, f"comp_{filenames[i]}")
                    plt.savefig(out_path, bbox_inches='tight')

                    # Crucial: Close the figure to prevent RAM overflow in the notebook
                    plt.close(fig)

                else:
                    # Original behavior: save just the prediction mask
                    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                    out_path = os.path.join(output_dir, filenames[i])
                    Image.fromarray(pred_mask_uint8).save(out_path)


def evaluate_metrics(model, dataloader, device, threshold=0.5):
    """
    Evaluates a PyTorch segmentation model and returns a dictionary of metrics.
    """
    model.eval()

    # Accumulators for the confusion matrix components
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # Binarize predictions based on the threshold
            preds = (preds > threshold).float()

            # Flatten the tensors to 1D arrays for easy element-wise operations
            preds = preds.view(-1)
            masks = masks.view(-1)

            # Compute TP, FP, FN, TN for this batch and add to totals
            total_tp += torch.sum(preds * masks).item()
            total_fp += torch.sum(preds * (1 - masks)).item()
            total_fn += torch.sum((1 - preds) * masks).item()
            total_tn += torch.sum((1 - preds) * (1 - masks)).item()

    # Epsilon prevents ZeroDivisionError if the model predicts nothing
    eps = 1e-6

    # Calculate final metrics
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)

    # F1 Score is also known as the Dice Coefficient
    f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)

    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)

    # Package into a readable dictionary
    metrics = {
        "IoU": round(iou, 4),
        "F1 Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(accuracy, 4)
    }

    return metrics


if __name__ == "__main__":
    main()