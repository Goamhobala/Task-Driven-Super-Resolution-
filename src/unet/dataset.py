"""S2-ROSA data loading for the UNet baseline.

The dataset is described entirely by ``metadata.parquet`` at the dataset root.
Each row is one *patch* = one internal block window of a satellite *tile* COG.
Pixels are read lazily from the satellite + mask COGs with a window
reconstructed from ``(patch_row_id, patch_col_id)``; nothing is unpacked to PNG.

Layout expected under ``dataset_dir``::

    metadata.parquet
    imagery/<zone>.tif            # multi-band Sentinel-2 reflectance COG
    masks_raster/<zone>_mask.tif  # single-band uint8 road mask COG (0/1)
"""

from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

# 1-based band indices into the imagery COG. Band order is R, G, B, NIR, ...
# (B4, B3, B2, B8, ...) per processor/graph.py and docs/sentinel2.md.
DEFAULT_BANDS = (1, 2, 3)  # RGB

# metadata.parquet columns this loader depends on.
TILE_PATH_COL = "tile_path"
MASK_PATH_COL = "mask_raster_path"
ROW_COL = "patch_row_id"
COL_COL = "patch_col_id"
ZONE_COL = "zone_name"
SPLIT_COL = "split_set"
_REQUIRED_COLS = [
    TILE_PATH_COL,
    MASK_PATH_COL,
    ROW_COL,
    COL_COL,
    ZONE_COL,
    SPLIT_COL,
]


def read_metadata(dataset_dir):
    """Read the patch catalogue (geometry column skipped, so plain pandas)."""
    path = Path(dataset_dir) / "metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"metadata.parquet not found at {path}. Point --dataset-dir at the "
            "S2-ROSA root produced by sentinel2data."
        )
    return pd.read_parquet(path, columns=_REQUIRED_COLS)


class ROSADataset(Dataset):
    """One sample per metadata row -> ``(image, mask, filename)``.

    ``image`` is ``(C, H, W)`` float32 reflectance in ~[0, 1]; ``mask`` is
    ``(1, H, W)`` float32 binarised to {0, 1}. Both are resized to a fixed
    ``image_size`` (edge block windows are smaller than a full block).
    """

    def __init__(
        self, frame, dataset_dir, bands=DEFAULT_BANDS, image_size=256, transform=None, normalize=True
    ):
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.bands = list(bands)
        self.normalize = normalize
        self.transform = transform or A.Compose([A.Resize(image_size, image_size)])
        # Open rasterio datasets lazily and cache per worker process. The cache
        # starts empty, so it is never pickled across DataLoader workers.
        self._src_cache = {}

    def __len__(self):
        return len(self.frame)

    def _open(self, rel_path):
        abs_path = str(self.dataset_dir / rel_path)
        src = self._src_cache.get(abs_path)
        if src is None:
            src = rasterio.open(abs_path)
            self._src_cache[abs_path] = src
        return src

    @staticmethod
    def _window(src, row_id, col_id):
        """Rebuild a block window from its (row, col) block index."""
        block_h, block_w = src.block_shapes[0]
        col_off = col_id * block_w
        row_off = row_id * block_h
        width = min(block_w, src.width - col_off)
        height = min(block_h, src.height - row_off)
        return Window(col_off, row_off, width, height)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img_src = self._open(row[TILE_PATH_COL])
        mask_src = self._open(row[MASK_PATH_COL])

        win = self._window(img_src, int(row[ROW_COL]), int(row[COL_COL]))

        # (C, H, W) -> (H, W, C) for albumentations
        image = img_src.read(self.bands, window=win).astype(np.float32)
        # Sentinel-2 COGs can store nodata as NaN/inf; left in, these propagate
        # through per-image standardization and make the loss NaN. Replace with 0.
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.transpose(image, (1, 2, 0))
        mask = (mask_src.read(1, window=win) > 0).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        image = np.clip(image, 0.0, 1.0)
        if self.normalize:
            # Per-image, per-channel standardization. Sentinel-2 reflectance is
            # ~5x darker than the ImageNet stats the encoder was pretrained on
            # (mean ~0.09 vs ~0.485); without this the encoder sees out-of-
            # distribution inputs and the decoder collapses to all-background.
            mean = image.mean(axis=(0, 1), keepdims=True)
            std = image.std(axis=(0, 1), keepdims=True) + 1e-6
            image = (image - mean) / std
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)

        filename = f"{row[ZONE_COL]}_{int(row[ROW_COL])}_{int(row[COL_COL])}.png"
        return image, mask, filename


class ROSADataModule(pl.LightningDataModule):
    """Reads metadata.parquet, partitions patches, serves train/val/test loaders.

    Splits come from the ``split_set`` column. The current dataset is entirely
    ``train`` (splits are scaffolded), so when val/test are absent we fall back
    to a deterministic random split controlled by ``val_frac``/``test_frac``.
    """

    def __init__(
        self,
        dataset_dir,
        batch_size=16,
        num_workers=2,
        bands=DEFAULT_BANDS,
        image_size=256,
        val_frac=0.1,
        test_frac=0.1,
        seed=42,
        drop_edge_blocks=True,
        normalize=True,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bands = tuple(bands)
        self.image_size = image_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.drop_edge_blocks = drop_edge_blocks
        self.normalize = normalize
        self.train_ds = self.val_ds = self.test_ds = None

    def _drop_edge_patches(self, df):
        """Keep only full-size block windows; drop partial edge blocks.

        A tile's last block row/col is smaller than a full block when the COG
        dimensions aren't multiples of the block size. Each tile COG is opened
        once to compare every patch's far edge against the raster extent.
        """
        df = df.reset_index(drop=True)
        full = pd.Series(False, index=df.index)
        for tile_path, group in df.groupby(TILE_PATH_COL):
            with rasterio.open(str(self.dataset_dir / tile_path)) as src:
                block_h, block_w = src.block_shapes[0]
                width, height = src.width, src.height
            fits = ((group[COL_COL] + 1) * block_w <= width) & (
                (group[ROW_COL] + 1) * block_h <= height
            )
            full.loc[group.index] = fits
        kept = df[full].reset_index(drop=True)
        print(f"Edge-block filter: kept {len(kept)}/{len(df)} full-size patches")
        return kept

    def _partition(self, df):
        split = df[SPLIT_COL].astype(str).str.lower()
        val = df[split.isin(["val", "validation"])]
        test = df[split.isin(["test"])]
        # No explicit val/test -> deterministic random split over everything.
        if len(val) == 0 and len(test) == 0:
            shuffled = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
            n = len(shuffled)
            n_test = int(n * self.test_frac)
            n_val = int(n * self.val_frac)
            test = shuffled.iloc[:n_test]
            val = shuffled.iloc[n_test : n_test + n_val]
            train = shuffled.iloc[n_test + n_val :]
            print(
                f"No val/test in split_set; random split -> train {len(train)}, "
                f"val {len(val)}, test {len(test)} (seed={self.seed})"
            )
        else:
            train = df[split.isin(["train"])]
        return train, val, test

    def setup(self, stage=None):
        df = read_metadata(self.dataset_dir)
        if self.drop_edge_blocks:
            df = self._drop_edge_patches(df)
        train, val, test = self._partition(df)

        def make(frame):
            return ROSADataset(
                frame,
                self.dataset_dir,
                bands=self.bands,
                image_size=self.image_size,
                normalize=self.normalize,
            )

        self.train_ds, self.val_ds, self.test_ds = make(train), make(val), make(test)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)
