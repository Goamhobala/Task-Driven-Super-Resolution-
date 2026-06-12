import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from unet.inference import save_predictions, evaluate_metrics
from unet.model import build_model # Import the new model architecture
import albumentations as A

def main():
    # Input path
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/S2IndianRegions'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    IMG_DIR = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    CHECKPOINT_PATH = '/kaggle/working/dlinknet34_resnet34_roads.pth' # Changed path
    PREDICTIONS_PATH = '/kaggle/working/predictions/test_set'

    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Setup
    _, _, test_list = sentinel2_data_partition(BASE_DIR)
    transform = A.Compose([A.Resize(256, 256)])

    test_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, test_list, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Load Model
    print("Initializing model architecture...")
    model = build_model().to(device)
    
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    # Needs matching map_location for environments handling CUDA differently
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
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



if __name__ == "__main__":
    main()