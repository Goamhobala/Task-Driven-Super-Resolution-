import os
from torch.utils.data import DataLoader
import albumentations as A
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb

from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from dlinknet.model import build_model, LightningWrapper


def main():
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/S2IndianRegions'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'

    IMG_DIR = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    image_net_mean = (0.485, 0.456, 0.406)
    image_net_std = (0.229, 0.224, 0.225)

    config = {
        "learning_rate": 0.001,
        "architecture": "DLinkNet34",
        "encoder": "resnet34",
        "dataset": "Sentinel-2",
        "epochs": 1,
        "batch_size": 4,
        "image_size": 1024,
    }

    wandb_logger = WandbLogger(project="dlinknet_sentinel2_baseline", config=config)

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=image_net_mean, std=image_net_std),
    ])

    train_list, val_list, _ = sentinel2_data_partition(BASE_DIR)
    train_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, train_list, transform=transform)
    val_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, val_list, transform=transform)

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    base_model = build_model()
    lightning_model = LightningWrapper(base_model, learning_rate=config["learning_rate"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="dlinknet34_resnet34_roads-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Training Complete. Best model saved at: {checkpoint_callback.best_model_path}")
    wandb.finish()


if __name__ == "__main__":
    main()