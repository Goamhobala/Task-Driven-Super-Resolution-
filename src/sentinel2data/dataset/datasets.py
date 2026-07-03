"""Custom Dataset Loader"""
import random
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
from sentinel2data.dataset.bands import DEFAULT_BANDS
from sentinel2data.dataset.reading import apply_norm, read_window


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


class RoadTileDataset(Dataset):
    """Use this dataset for training. Random pixel crops from the 512x512 train tiles.

    Length by default is 10 * len(train_tiles)
    """

    def __init__(self, dataset_dir, bands=DEFAULT_BANDS, image_size=256,
                 length=None, normalize=True, norm_mean=None, norm_std=None):
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, "train").reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = image_size
        self.normalize = normalize
        self.norm_mean = norm_mean  # frozen train stats (or None -> per-image)
        self.norm_std = norm_std
        self.length = length if length is not None else 10 * len(self.df)

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Random crop and normalise"""
        row = self.df.iloc[random.randrange(len(self.df))]
        s = self.image_size
        with rasterio.open(self.dataset_dir / row["image_path"]) as src, \
                rasterio.open(self.dataset_dir / row["mask_path"]) as msrc:
            H, W = src.height, src.width
            top = random.randint(0, max(0, H - s))
            left = random.randint(0, max(0, W - s))
            win = Window(left, top, min(s, W - left), min(s, H - top))
            img = read_window(src, self.bands, win)                 # (C, h, w)
            mask = (msrc.read(1, window=win) > 0).astype("float32")  # (h, w)

        c, h, w = img.shape
        if (h, w) != (s, s):  # short edge tile -> zero-pad (train tiles are 512, rare)
            pad_i = np.zeros((c, s, s), dtype="float32"); pad_i[:, :h, :w] = img
            pad_m = np.zeros((s, s), dtype="float32"); pad_m[:h, :w] = mask
            img, mask = pad_i, pad_m

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_{top}_{left}.png"


class TileCropDataset(Dataset):
    """
    Use this dataset for validation/testing. 

    Deterministic 2x2 crops from the 512x512 tiles - sliding window with no overlaps
    Item ``idx`` -> tile ``idx // 4``, quadrant ``idx % 4`` (row-major)
    """

    def __init__(self, dataset_dir, split, bands=DEFAULT_BANDS,
                 normalize=True, norm_mean=None, norm_std=None):
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, split).reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = 256
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.per_tile = 4  # Assume 512x512 tiles, hence 4 patches.

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def __len__(self):
        return len(self.df) * self.per_tile

    def __getitem__(self, idx):
        """Sliding Window and normalise"""
        row = self.df.iloc[idx // self.per_tile]
        quad = idx % self.per_tile
        s = self.image_size
        top = (quad // 2) * s
        left = (quad % 2) * s
        with rasterio.open(self.dataset_dir / row["image_path"]) as src, \
                rasterio.open(self.dataset_dir / row["mask_path"]) as msrc:
            win = Window(left, top, s, s)
            img = read_window(src, self.bands, win)                  # (C, s, s)
            mask = (msrc.read(1, window=win) > 0).astype("float32")  # (s, s)

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_q{quad}.png"


class RoadDataModule(pl.LightningDataModule):
    """Train - random native crops; val/test - deterministic 2x2 quadrant crops."""

    def __init__(self, dataset_dir: str, bands: tuple[int, ...] = DEFAULT_BANDS,
                 batch_size: int = 16, num_workers: int = 2, image_size: int = 256,
                 length: int | None = None, normalize: bool = True,
                 norm_mean: list[float] | None = None, norm_std: list[float] | None = None):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.bands = tuple(bands)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.length = length
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def train_dataloader(self):
        ds = RoadTileDataset(
            self.dataset_dir, self.bands, self.image_size, self.length, self.normalize,
            self.norm_mean, self.norm_std,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,  # randomness is in __getitem__; DDP adds DistributedSampler
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def _eval_loader(self, split):
        ds = TileCropDataset(
            self.dataset_dir, split, self.bands, self.normalize,
            self.norm_mean, self.norm_std,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self):
        return self._eval_loader("val")

    def test_dataloader(self):
        return self._eval_loader("test")
