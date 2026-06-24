import os
import albumentations as A
import lightning as L
from torch.utils.data import DataLoader

from unet.dataset import SentinelRoadsDataset, sentinel2_data_partition


class SentinelDataModule(L.LightningDataModule):
    """
    Reproduces the dataset/dataloader setup from train.py's main(),
    parameterized so LightningCLI's `data:` YAML block can drive it.

    Maps 1:1 onto train.yml's data section:
        base_dir         -> BASE_DIR   (passed to sentinel2_data_partition)
        dataset_dir       -> DATASET_DIR (root for images_enhanced_png/masks_png)
        image_size        -> A.Resize(image_size, image_size)
        train_batch_size  -> DataLoader batch_size for train split
        eval_batch_size   -> DataLoader batch_size for val split
        num_workers       -> DataLoader num_workers for both splits
    """

    def __init__(
        self,
        base_dir: str,
        dataset_dir: str,
        image_size: int = 1024,
        train_batch_size: int = 4,
        eval_batch_size: int = 16,
        num_workers: int = 2,
        transform: A = None
    ):
        super().__init__()
        self.base_dir = base_dir
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.img_dir = os.path.join(self.dataset_dir, "images_enhanced_png", "images_enhanced_png")
        self.mask_dir = os.path.join(self.dataset_dir, "masks_png", "masks_png")

        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        train_list, val_list, test_list = sentinel2_data_partition(self.base_dir)

        self.train_dataset = SentinelRoadsDataset(
            self.img_dir, self.mask_dir, train_list, transform=self.transform
        )
        self.val_dataset = SentinelRoadsDataset(
            self.img_dir, self.mask_dir, val_list, transform=self.transform
        )
        self.test_dataset = SentinelRoadsDataset(
            self.img_dir, self.mask_dir, test_list, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )