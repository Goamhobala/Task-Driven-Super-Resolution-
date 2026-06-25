"""Native-CRS map-style loaders for S2-ROSA-V2 (no torchgeo, no warp).

Each tile/zone is read in its **native CRS, native pixels** via rasterio windows
-- no reprojection (the imagery spans several UTM zones, so a shared CRS would
force a per-patch warp; that is the torchgeo cost we avoid). NaN/inf in the COG
is scrubbed to 0 *before* standardisation. Bands are 1-based COG indices.

  * train -> :class:`RoadTileDataset`: random 256x256 crop from a random 512 tile.
  * val/test -> :class:`ZoneDataset`: one item per whole zone; the stitched,
    overlap-blended scoring happens in ``unet.model``'s ``validation_step``.

Map-style datasets also shard cleanly under DDP (Lightning's DistributedSampler).
"""
from __future__ import annotations

import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

DEFAULT_BANDS = (21, 22, 23)  # 1-based: the appended enhanced-RGB triplet


def read_window(src, bands, window):
    """Read ``(C, h, w)`` float32 from an open rasterio dataset, NaN/inf -> 0.

    The COG contains N/A values; scrub them before they reach standardisation or
    the model (NaN would poison per-image mean/std and the forward pass)."""
    arr = src.read(bands, window=window).astype("float32")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def standardize(img):
    """Per-image, per-channel standardisation of a ``(C, H, W)`` float array."""
    mean = img.mean(axis=(1, 2), keepdims=True)
    std = img.std(axis=(1, 2), keepdims=True) + 1e-6
    return (img - mean) / std


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


class RoadTileDataset(Dataset):
    """Random native-pixel crops from the train tiles. ``__len__`` = patches/epoch."""

    def __init__(self, dataset_dir, bands=DEFAULT_BANDS, image_size=256,
                 length=None, normalize=True):
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, "train").reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = image_size
        self.normalize = normalize
        self.length = length if length is not None else 10 * len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
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
            img = standardize(img)
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_{top}_{left}.png"


class ZoneDataset(Dataset):
    """One item per zone -> ``(image_path, mask_path, zone_name)`` for stitched eval."""

    def __init__(self, dataset_dir, split, bands=DEFAULT_BANDS):
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, split).reset_index(drop=True)
        self.bands = list(bands)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            str(self.dataset_dir / row["image_path"]),
            str(self.dataset_dir / row["mask_path"]),
            str(row["zone_name"]),
        )


def _zone_collate(batch):
    """``batch_size=1`` -> hand the single ``(img, mask, zone)`` tuple straight through."""
    return batch[0]


class RoadDataModule(pl.LightningDataModule):
    """Train = random native crops; val/test = whole zones (stitched in the model)."""

    def __init__(self, dataset_dir, bands=DEFAULT_BANDS, batch_size=16, num_workers=2,
                 image_size=256, length=None, normalize=True):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.bands = tuple(bands)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.length = length
        self.normalize = normalize

    def train_dataloader(self):
        ds = RoadTileDataset(
            self.dataset_dir, self.bands, self.image_size, self.length, self.normalize
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

    def _zone_loader(self, split):
        ds = ZoneDataset(self.dataset_dir, split, self.bands)
        return DataLoader(ds, batch_size=1, num_workers=0, collate_fn=_zone_collate)

    def val_dataloader(self):
        return self._zone_loader("val")

    def test_dataloader(self):
        return self._zone_loader("test")
