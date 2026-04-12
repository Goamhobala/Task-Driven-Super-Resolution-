import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp
import wandb

from terramind.dataset import SentinelRoadsDataset, sentinel2_data_partition
from terramind.model import build_model, train_model


def main():
    # Input paths (same layout as the UNet experiment)
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2'
    DATASET_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_256/15765738'
    CHECKPOINT_PATH = '/kaggle/working/terramind_v1_base_roads.pth'

    IMG_DIR = os.path.join(DATASET_DIR, 'images_enhanced_png', 'images_enhanced_png')
    MASK_DIR = os.path.join(DATASET_DIR, 'masks_png', 'masks_png')

    # Initialize Weights & Biases
    wandb.init(
        project="terramind_sentinel2_roads",
        config={
            "learning_rate": 0.0001,   # Lower LR suits large pre-trained ViT
            "architecture": "TerraMind",
            "backbone": "terramind_v1_base",
            "decoder": "FCNDecoder",
            "dataset": "Sentinel-2",
            "epochs": 5,
            "batch_size": 8,           # ViT-Base is heavier than ResNet50
            "image_size": 256,
            "loss_function": "DiceLoss",
        }
    )

    # Load the shared train/val/test split
    train_list, val_list, test_list = sentinel2_data_partition(BASE_DIR)

    # Transform — images are already 256x256 but Resize keeps it explicit
    transform = A.Compose([A.Resize(256, 256)])

    # Instantiate datasets
    train_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, train_list, transform=transform)
    val_dataset   = SentinelRoadsDataset(IMG_DIR, MASK_DIR, val_list,   transform=transform)
    test_dataset  = SentinelRoadsDataset(IMG_DIR, MASK_DIR, test_list,  transform=transform)

    batch_size = wandb.config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=True).to(device)

    # DiceLoss for binary road segmentation (consistent with UNet baseline)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=wandb.config.epochs,
        save_path=CHECKPOINT_PATH,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
