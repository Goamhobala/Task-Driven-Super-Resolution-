import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class SentinelRoadsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Store the exact list of files assigned to this split
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the filename from the predefined list
        filename = self.file_list[idx] + ".png"  

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Read image (RGB) and mask (Grayscale)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarize the mask (0 or 1)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to PyTorch tensors
        # Image: (C, H, W) normalized to [0, 1]
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        # Mask: Add channel dimension (1, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask, filename


def sentinel2_data_partition(dataset_dir):
    json_path = os.path.join(dataset_dir, 'data_split.json')

    if not os.path.exists(json_path):
         raise FileNotFoundError(f"Cannot find {json_path}. Did you run the generation script first?")

    with open(json_path, 'r') as jf:
        data_list = json.load(jf)

    train_list = data_list['train']
    val_list = data_list['validation']
    test_list = data_list['test']

    return train_list, val_list, test_list