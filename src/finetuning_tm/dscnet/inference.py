import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
import lightning as L

from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from finetuning_tm.dscnet.lightning_utils import LightningWrapper, PredictionSaverCallback
from dscnet.model import build_model

def main():
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/S2IndianRegions'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    IMG_DIR = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    # NOTE: this now points at the Lightning checkpoint from the ModelCheckpoint
    # callback in training (a .ckpt, not the old .pth)
    CHECKPOINT_PATH = '/kaggle/working/checkpoints/dlinknet34_resnet34_roads-epoch=09-val_loss=0.1234.ckpt'
    PREDICTIONS_PATH = '/kaggle/working/predictions/dlinknet_test_set'

    image_net_mean = (0.485, 0.456, 0.406)
    image_net_std = (0.229, 0.224, 0.225)

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=image_net_mean, std=image_net_std),
    ])

    _, _, test_list = sentinel2_data_partition(BASE_DIR)
    test_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, test_list, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    base_model = build_model()
    lightning_model = LightningWrapper.load_from_checkpoint(CHECKPOINT_PATH, model=base_model)

    # --- metrics: replaces evaluate_metrics() ---
    metrics_trainer = L.Trainer(accelerator="auto", devices="auto", logger=False)
    test_results = metrics_trainer.test(lightning_model, dataloaders=test_loader)
    print("Test Set Metrics:")
    for metric, value in test_results[0].items():
        print(f"  - {metric}: {value:.4f}")

    # --- predictions/visualization: replaces save_predictions() ---
    saver = PredictionSaverCallback(PREDICTIONS_PATH, save_comparison=True)
    predict_trainer = L.Trainer(accelerator="auto", devices="auto", logger=False, callbacks=[saver])
    predict_trainer.predict(lightning_model, dataloaders=test_loader, return_predictions=False)

    print("Done! Test predictions saved.")


if __name__ == "__main__":
    main()