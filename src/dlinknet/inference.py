import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from unet.model import DLinkNet34 # Import the new model architecture
import albumentations as A

def main():
    # Input path
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    IMG_DIR = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    CHECKPOINT_PATH = '/kaggle/working/dlinknet34_resnet34_roads.pth' # Changed path
    PREDICTIONS_PATH = '/kaggle/working/predictions/test_set'

    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Setup
    _, _, test_list = sentinel2_data_partition(DATASET_DIR)
    test_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, test_list, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Load Model
    print(f"Loading best DLinkNet34 model from {CHECKPOINT_PATH}...")
    model = DLinkNet34(num_classes=1)
    
    # Needs matching map_location for environments handling CUDA differently
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    threshold = 0.5 

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for i, (images, masks, filenames) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # Binarize predictions based on the threshold
            preds = (preds > threshold).float()
            
            # --- VISUALIZATION & SAVING ---
            for j in range(images.size(0)):
                pred_mask = preds[j].squeeze().cpu().numpy() * 255.0  # Scale 0-1 back to 0-255
                pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
                save_file = os.path.join(PREDICTIONS_PATH, f"{filenames[j]}_pred.png")
                pred_mask_img.save(save_file)

            # Flatten the tensors to 1D arrays for easy element-wise operations
            preds = preds.view(-1)
            masks = masks.view(-1)

            # Compute TP, FP, FN, TN for this batch and add to totals
            total_tp += torch.sum(preds * masks).item()
            total_fp += torch.sum(preds * (1 - masks)).item()
            total_fn += torch.sum((1 - preds) * masks).item()
            total_tn += torch.sum((1 - preds) * (1 - masks)).item()

    # Calculate final metrics
    eps = 1e-6
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)

    # Package into a readable dictionary
    metrics = {
        "IoU": round(iou, 4),
        "F1 Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(accuracy, 4),
    }
    
    print("\n--- Final Test Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Predictions saved to {PREDICTIONS_PATH}")

if __name__ == "__main__":
    main()