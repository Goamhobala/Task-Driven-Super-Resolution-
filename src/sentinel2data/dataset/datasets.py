"""Native-CRS, map-style datasets for S2-ROSA-V2 (shared; no torchgeo, no warp).

Each tile/zone is read in its **native CRS, native pixels** via rasterio windows
-- no reprojection (the imagery spans several UTM zones, so a shared CRS would
force a per-patch warp). NaN scrubbed before standardisation. Bands are 1-based
COG indices.

  * train -> :class:`RoadTileDataset`: random 256x256 crop from a random 512 tile.
  * val/test -> :class:`TileCropDataset`: the 4 deterministic non-overlapping
    256x256 quadrants of each 512 tile (a 2x2 partition -> once-per-pixel coverage,
    so the accumulated IoU/F1 is a stable per-pixel metric, not a per-tile average).

Map-style datasets shard cleanly under DDP (Lightning's DistributedSampler). Any
model imports these from ``sentinel2data.dataset``.
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

from sentinel2data.dataset.bands import DEFAULT_BANDS
from sentinel2data.dataset.reading import apply_norm, read_window


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


class RoadTileDataset(Dataset):
    """Random native-pixel crops from the train tiles. ``__len__`` = patches/epoch."""

    def __init__(self, dataset_dir, bands=DEFAULT_BANDS, image_size=256,
                 length=None, normalize=True, norm_mean=None, norm_std=None):
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, "train").reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = image_size
        self.normalize = normalize
        self.norm_mean = norm_mean  # full-stack frozen train stats (or None -> per-image)
        self.norm_std = norm_std
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
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_{top}_{left}.png"


class TileCropDataset(Dataset):
    """Deterministic eval crops: the 4 non-overlapping ``image_size`` quadrants of
    each 512 tile (2x2 partition -> once-per-pixel coverage, stable val/test metric).

    Item ``idx`` -> tile ``idx // 4``, quadrant ``idx % 4`` (row-major). Tiles are
    exactly ``tile_size`` px (the generator drops partial edge strips), so every
    quadrant read is a full ``image_size`` window -- no padding.
    """

    def __init__(self, dataset_dir, split, bands=DEFAULT_BANDS, image_size=256,
                 normalize=True, norm_mean=None, norm_std=None):
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, split).reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = image_size
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.per_tile = 4  # 2x2 non-overlapping quadrants

    def __len__(self):
        return len(self.df) * self.per_tile

    def __getitem__(self, idx):
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
    """Train = random native crops; val/test = deterministic 2x2 quadrant crops."""

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
        # Frozen per-band train stats (full 23-band stack); None -> per-image standardise.
        self.norm_mean = norm_mean
        self.norm_std = norm_std

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
            self.dataset_dir, split, self.bands, self.image_size, self.normalize,
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
