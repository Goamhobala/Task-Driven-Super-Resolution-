"""Bicubic upscaler dataloader 

Reads a 10m ROSA tiled dataset, upscales 128px image with bicubic interpolation 2.5m to the model

 A `crop_size` native-px window is read and bicubic-upsampled to `crop_size * upscale`.
The mask is rasterised fresh from that tile's road-graph centrelines at the upsampled transform
"""
import random
from functools import lru_cache
from pathlib import Path
import geopandas as gpd
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio import Affine, features
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from torch.utils.data import DataLoader, Dataset
from sentinel2data.dataset.bands import DEFAULT_BANDS
from sentinel2data.dataset.reading import apply_norm
from sentinel2data.generator.helper import window_bounds

# Per-row road buffer half-width (metres), carried in each masks_graph parquet.
_BUFFER_COL = "buffer"


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


@lru_cache(maxsize=256)
def _load_graph(path_str):
    """One tile's road centrelines (native CRS), cached per worker process."""
    return gpd.read_parquet(path_str)


def _read_upsampled(src, bands, window, out_size):
    """Bicubic-read a native ``window`` to ``out_size`` px (NaN/inf -> 0).

    ``out_shape`` forces the returned array to ``out_size`` regardless of edge
    truncation, so image and mask always share the ``(out_size, out_size)`` grid."""
    arr = src.read(
        bands, window=window,
        out_shape=(len(bands), out_size, out_size),
        resampling=Resampling.cubic,
    ).astype("float32")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _graph_mask(graph_path, src, window, out_size, upscale):
    """Rasterise the tile's buffered centrelines over the crop at output resolution.

    Each centreline is buffered by its per-row ``buffer`` half-width (metres) from
    the masks_graph parquet -- geometries are in the tile's metric CRS, so the buffer
    is in metres directly. The upsampled transform keeps the crop's geographic extent
    but 1/upscale-sized pixels, so the mask aligns with the bicubic-upsampled image."""
    out_shape = (out_size, out_size)
    roads = _load_graph(str(graph_path))
    if roads.empty:
        return np.zeros(out_shape, dtype="float32")
    if _BUFFER_COL not in roads.columns:
        raise ValueError(
            f"masks_graph parquet {graph_path} lacks a '{_BUFFER_COL}' column; "
            "regenerate the dataset with the current road-graph labeler."
        )
    minx, miny, maxx, maxy = window_bounds(window, src.transform)
    cand = roads.cx[minx:maxx, miny:maxy]
    if cand.empty:
        return np.zeros(out_shape, dtype="float32")
    up_tf = window_transform(window, src.transform) * Affine.scale(1.0 / upscale)
    buffered = cand.geometry.buffer(cand[_BUFFER_COL].to_numpy(dtype="float64"))
    mask = features.rasterize(
        ((g, 1) for g in buffered), out_shape=out_shape, transform=up_tf,
        fill=0, all_touched=True, dtype="uint8",
    )
    return (mask > 0).astype("float32")


class UpscaleRoadTileDataset(Dataset):
    """Random bicubic-upsampled crops from the train tiles. ``__len__`` = patches/epoch."""

    def __init__(self, dataset_dir, bands=DEFAULT_BANDS, crop_size=128, upscale=4,
                 length=None, normalize=True, norm_mean=None, norm_std=None,
                 min_road_density=0.0):
        self.dataset_dir = Path(dataset_dir)
        df = _read_split_csv(dataset_dir, "train")
        if min_road_density > 0 and "road_density" in df.columns:
            n0 = len(df)
            df = df[df["road_density"] >= min_road_density]
            print(f"[upscale train] road_density >= {min_road_density}: "
                  f"kept {len(df)}/{n0} tiles")
        self.df = df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No train tiles left after the min_road_density filter.")
        self.bands = list(bands)
        self.crop_size = crop_size
        self.upscale = upscale
        self.out_size = crop_size * upscale
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.length = length if length is not None else 10 * len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[random.randrange(len(self.df))]
        cn = self.crop_size
        with rasterio.open(self.dataset_dir / row["image_path"]) as src:
            H, W = src.height, src.width
            top = random.randint(0, max(0, H - cn))
            left = random.randint(0, max(0, W - cn))
            win = Window(left, top, min(cn, W - left), min(cn, H - top))
            img = _read_upsampled(src, self.bands, win, self.out_size)
            mask = _graph_mask(
                self.dataset_dir / row["mask_graph_path"], src, win,
                self.out_size, self.upscale,
            )

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_{top}_{left}.png"


class UpscaleTileCropDataset(Dataset):
    """Deterministic eval crops: the non-overlapping ``crop_size`` native cells of each
    tile (``ceil(H/crop) x ceil(W/crop)`` grid -> once-per-pixel coverage), each
    bicubic-upsampled to ``crop_size * upscale`` with a graph-rasterised mask."""

    def __init__(self, dataset_dir, split, bands=DEFAULT_BANDS, crop_size=128, upscale=4,
                 normalize=True, norm_mean=None, norm_std=None):
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, split).reset_index(drop=True)
        self.bands = list(bands)
        self.crop_size = crop_size
        self.upscale = upscale
        self.out_size = crop_size * upscale
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.grid_w, self.grid_h = self._tile_grid()
        self.per_tile = self.grid_w * self.grid_h

    def _tile_grid(self):
        """Crop grid ``(cols, rows)`` covering one tile (from the first tile's size)."""
        if self.df.empty:
            return 1, 1
        cn = self.crop_size
        with rasterio.open(self.dataset_dir / self.df.iloc[0]["image_path"]) as src:
            h, w = src.height, src.width
        return (w + cn - 1) // cn, (h + cn - 1) // cn

    def __len__(self):
        return len(self.df) * self.per_tile

    def __getitem__(self, idx):
        row = self.df.iloc[idx // self.per_tile]
        cell = idx % self.per_tile
        cn = self.crop_size
        top = (cell // self.grid_w) * cn
        left = (cell % self.grid_w) * cn
        with rasterio.open(self.dataset_dir / row["image_path"]) as src:
            H, W = src.height, src.width
            win = Window(left, top, min(cn, W - left), min(cn, H - top))
            img = _read_upsampled(src, self.bands, win, self.out_size)
            mask = _graph_mask(
                self.dataset_dir / row["mask_graph_path"], src, win,
                self.out_size, self.upscale,
            )

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_c{cell}.png"


class UpscaleRoadDataModule(pl.LightningDataModule):
    """Super-resolution dataloader: bicubic-upsampled crops + graph-rasterised masks.

    Model input patch = ``crop_size * upscale`` (128*4 = 512). ``image_size`` is kept
    only so the shared ``UNetCLI`` link ``data.image_size -> model.image_size`` resolves
    (the model is fully-convolutional and ignores it)."""

    def __init__(self, dataset_dir: str, bands: tuple[int, ...] = DEFAULT_BANDS,
                 batch_size: int = 4, num_workers: int = 2, crop_size: int = 128,
                 upscale: int = 4, image_size: int = 256, length: int | None = None,
                 normalize: bool = True, norm_mean: list[float] | None = None,
                 norm_std: list[float] | None = None, min_road_density: float = 0.0):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.bands = tuple(bands)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.upscale = upscale
        self.image_size = image_size  # vestigial; keeps the CLI link resolvable
        self.length = length
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.min_road_density = min_road_density

    def train_dataloader(self):
        ds = UpscaleRoadTileDataset(
            self.dataset_dir, self.bands, self.crop_size, self.upscale, self.length,
            self.normalize, self.norm_mean, self.norm_std, self.min_road_density,
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0, drop_last=True,
        )

    def _eval_loader(self, split):
        ds = UpscaleTileCropDataset(
            self.dataset_dir, split, self.bands, self.crop_size, self.upscale,
            self.normalize, self.norm_mean, self.norm_std,
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0, drop_last=False,
        )

    def val_dataloader(self):
        return self._eval_loader("val")

    def test_dataloader(self):
        return self._eval_loader("test")
