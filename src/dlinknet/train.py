import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp
import wandb

from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from dlinknet.model import build_model, train_model  # Updated imports

def main():
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/S2IndianRegions'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    CHECKPOINT_PATH = '/kaggle/working/dlinknet34_resnet34_roads.pth'

    IMG_DIR = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    image_net_mean = (0.485, 0.456, 0.406)
    image_net_std = (0.229, 0.224, 0.225)

    wandb.init(
        project="dlinknet_sentinel2_baseline",
        config={
            "learning_rate": 0.001,
            "architecture": "DLinkNet34",
            "encoder": "resnet34",
            "dataset": "Sentinel-2",
            "epochs": 1,
            "batch_size": 16, 
            "image_size": 256
        }
    )

    # dinknet24 built for 1024, resnet34 need normalisation 
    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=image_net_mean, std=image_net_std),
        ])


    train_list, val_list, _ = sentinel2_data_partition(BASE_DIR)

    train_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, train_list, transform=transform)
    val_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, val_list, transform=transform)

    batch_size = wandb.config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model cleanly
    model = build_model().to(device)

    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=wandb.config.epochs,
        save_path=CHECKPOINT_PATH
    )

    wandb.finish()
    print("Training Complete. Best model saved.")

if __name__ == "__main__":
    main()