"""S2-ROSA data loading for the UNet baseline.
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

DEFAULT_BANDS = (1, 2, 3)  # RGB

# TODO: currently not used, we currently pick bands by their index in the COG. 
SATELLITE_BANDS = {
    # Sentinel-2 Multispectral Instrument (MSI)
    'B4': 'Red (Visible)',
    'B3': 'Green (Visible)',
    'B2': 'Blue (Visible)',
    'B8': 'Near Infrared (NIR)',
    'B5': 'Vegetation Red Edge 1',
    'B6': 'Vegetation Red Edge 2',
    'B7': 'Vegetation Red Edge 3',
    'B8A': 'Narrow Near Infrared (NIR)',
    'B11': 'Shortwave Infrared 1 (SWIR 1)',
    'B12': 'Shortwave Infrared 2 (SWIR 2)',

    # Sentinel-1 Synthetic Aperture Radar (SAR)
    'VV_ascending': 'Vertical-transmit, Vertical-receive polarization (Ascending Pass)',
    'VH_ascending': 'Vertical-transmit, Horizontal-receive cross-polarization (Ascending Pass)',
    'VV_descending': 'Vertical-transmit, Vertical-receive polarization (Descending Pass)',
    'VH_descending': 'Vertical-transmit, Horizontal-receive cross-polarization (Descending Pass)',

    # Topography / Terrain Data
    'elevation': 'Elevation (Height above sea level in meters)',
    'slope': 'Slope (Terrain steepness in degrees)',
    'aspect': 'Aspect (Compass direction the terrain faces in degrees)',

    # Urban Masks
    'esa_urban_10m': 'ESA WorldCover 10m Urban Mask 2020',
    'gisa_urban_10m': 'GISLab 10m Urban Mask 2019',
    'wsf_urban_10m': 'World Settlement Footprint 10m Urban Mask 2019 ',
}

# metadata.parquet columns the loader expects
TILE_PATH_COL = "tile_path"
MASK_PATH_COL = "mask_raster_path"
ROW_COL = "patch_row_id"
COL_COL = "patch_col_id"
ZONE_COL = "zone_name"
SPLIT_COL = "split_set"
ROAD_DENSITY_COL = "road_density"  # road_px / total_px, precomputed by MetadataGenerator
_REQUIRED_COLS = [
    TILE_PATH_COL,
    MASK_PATH_COL,
    ROW_COL,
    COL_COL,
    ZONE_COL,
    SPLIT_COL,
    ROAD_DENSITY_COL,
]


def read_metadata(dataset_dir):
    """Read the patch catalogue"""
    path = Path(dataset_dir) / "metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"metadata.parquet not found at {path}. Point --dataset-dir at the "
            "S2-ROSA root produced by sentinel2data."
        )
    return pd.read_parquet(path, columns=_REQUIRED_COLS)


class ROSADataset(Dataset):
    """One sample per metadata row - `(image, mask, filename)`.

    - `image` is `(C, H, W)` 
    - `mask` is `(1, H, W)`
    - Both are resized to a fixed `image_size` (edge block windows are smaller than a full block).
    """

    def __init__(
        self, frame, dataset_dir, bands=DEFAULT_BANDS, image_size=256, transform=None, normalize=True
    ):
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.bands = list(bands)
        self.normalize = normalize
        self.transform = transform or A.Compose([A.Resize(image_size, image_size)])
        self._src_cache = {} # image connection cache

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
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) # TODO: check why there is NA in COG
        image = np.transpose(image, (1, 2, 0))
        mask = (mask_src.read(1, window=win) > 0).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        image = np.clip(image, 0.0, 1.0)
        if self.normalize:
            # Per-image, per-channel standardization.
            mean = image.mean(axis=(0, 1), keepdims=True)
            std = image.std(axis=(0, 1), keepdims=True) + 1e-6
            image = (image - mean) / std
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)

        filename = f"{row[ZONE_COL]}_{int(row[ROW_COL])}_{int(row[COL_COL])}.png"
        return image, mask, filename


class ROSADataModule(pl.LightningDataModule):
    """Reads metadata.parquet, partitions patches, serves train/val/test loaders.
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
        drop_empty_patches=True,
        min_road_density=0.0,
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
        self.drop_empty_patches = drop_empty_patches
        self.min_road_density = min_road_density
        self.normalize = normalize
        self.train_ds = self.val_ds = self.test_ds = None

    def _drop_edge_patches(self, df):
        """Keep only full-size block windows; drop partial edge blocks.
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

    def _drop_empty_patches(self, df, split_name):
        """Drop road-free patches using the precomputed ``road_density`` column.

        Keeps patches with ``road_density > min_road_density`` (default 0.0 drops
        only patches with zero road pixels). Applied to the train split only so
        val/test keep the real road/no-road distribution.
        """
        kept = df[df[ROAD_DENSITY_COL] > self.min_road_density].reset_index(drop=True)
        print(
            f"Empty-patch filter [{split_name}]: kept {len(kept)}/{len(df)} patches "
            f"(road_density > {self.min_road_density})"
        )
        return kept

    def _partition(self, df):
        split = df[SPLIT_COL].astype(str).str.lower()
        val = df[split.isin(["val", "validation"])]
        test = df[split.isin(["test"])]

        # check if val/test are empty
        if len(val) == 0 or len(test) == 0:
            print("Warning: val/test split not found in metadata.parquet")
            raise ValueError("val/test split not found in metadata.parquet")
        else:
            train = df[split.isin(["train"])]
        return train, val, test

    def setup(self, stage=None):
        df = read_metadata(self.dataset_dir)
        if self.drop_edge_blocks:
            df = self._drop_edge_patches(df)
        train, val, test = self._partition(df)
        if self.drop_empty_patches:
            train = self._drop_empty_patches(train, "train")

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
