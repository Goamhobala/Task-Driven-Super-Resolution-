import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
import lightning as L

from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from dlinknet.model import build_model, DLinkNet34Module


class PredictionSaverCallback(L.Callback):
    """Writes comparison plots (or raw masks) per batch instead of buffering them."""
    def __init__(self, output_dir, save_comparison=True, threshold=0.5):
        super().__init__()
        self.output_dir = output_dir
        self.save_comparison = save_comparison
        self.threshold = threshold
        os.makedirs(output_dir, exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images, masks, filenames = batch
        preds = (outputs > self.threshold).float().cpu().numpy()
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(len(filenames)):
            pred_mask = preds[i].squeeze()

            if self.save_comparison:
                img = images_np[i].transpose(1, 2, 0)
                true_mask = masks_np[i].squeeze()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img); axes[0].set_title("Original Image"); axes[0].axis("off")
                axes[1].imshow(true_mask, cmap="gray"); axes[1].set_title("True Road Label"); axes[1].axis("off")
                axes[2].imshow(pred_mask, cmap="gray"); axes[2].set_title("Predicted Road"); axes[2].axis("off")
                plt.tight_layout()
                fig.savefig(os.path.join(self.output_dir, f"comp_{filenames[i]}"), bbox_inches="tight")
                plt.close(fig)
            else:
                pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_mask_uint8).save(os.path.join(self.output_dir, filenames[i]))


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
    lightning_model = DLinkNet34Module.load_from_checkpoint(CHECKPOINT_PATH, model=base_model)

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