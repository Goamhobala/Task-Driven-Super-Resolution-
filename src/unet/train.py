import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp
from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition
from unet.model import build_model, train_model

def main():
    # Input paths
    BASE_DIR = '/kaggle/working/InstaRoadPrototype/dataset/sentinel2/sentinel2_1024'
    CHECKPOINT_PATH = '/kaggle/working/unetplusplus_resnet50_roads.pth'

    IMG_DIR = os.path.join(BASE_DIR, 'images_1024')
    MASK_DIR = os.path.join(BASE_DIR, 'clean_masks')

    # Load the partitions
    train_list, val_list, test_list = sentinel2_data_partition(BASE_DIR)

    # Transform (just in case)
    transform = A.Compose([A.Resize(1024, 1024)])

    # Instantiate datasets
    train_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, train_list, transform=transform)
    val_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, val_list, transform=transform)
    test_dataset = SentinelRoadsDataset(IMG_DIR, MASK_DIR, test_list, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model using model.py
    model = build_model().to(device)

    # Define Loss and Optimizer
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50

    # Execute the training loop
    model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )

    # Define the path and save the weights
    torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Model saved successfully to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
