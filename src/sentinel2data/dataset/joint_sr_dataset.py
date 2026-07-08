"""Joint-SR dataloader: NATIVE 10 m crops paired with HR (2.5 m) masks.

Mirrors :mod:`sentinel2data.dataset.upscale_dataset` (same split CSVs, crop
grids, ``min_road_density`` filter, module layout) with two differences that
define the joint-SR experiment:

  * The image crop is returned at NATIVE resolution, raw DN (nodata -> 0,
    **no bicubic upsampling and no normalisation**) — upsampling is the SR
    network's job inside :class:`sr.model.JointSRUNetLightning`, and
    normalisation happens there *after* super-resolution (SEN2SR consumes raw
    reflectance). ``norm_mean``/``norm_std``/``normalize`` are still declared
    on the datamodule so the shared ``UNetCLI`` links feed them to the model;
    the dataset itself never applies them.
  * The HR mask (``crop_size * upscale`` px) comes from one of two sources:
      - ``mask_source="graph"``  (default): the tile's ``masks_graph`` parquet
        rasterised at the upsampled transform — the pipeline's labels (CDNGI),
        exactly like ``upscale_dataset._graph_mask``.
      - ``mask_source="raster"``: pre-generated HR mask COGs beside the
        imagery (``<split>/<mask_dirname>/{tile}.tif``, dims exactly
        ``upscale`` x the tile's — e.g. the OSM masks written by
        OpenStreetMapTest/dataset_hr_masks.py --scale 4), read at the
        ``upscale``-scaled window. Enables the OSM-vs-CDNGI label comparison
        on the joint-SR model too.
"""
import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from sentinel2data.dataset.reading import read_window
from sentinel2data.dataset.upscale_dataset import _graph_mask, _read_split_csv

# SEN2SR's required input: raw [B4, B3, B2, B8] = R, G, B, NIR (V2 bands 1-4).
SR_INPUT_BANDS = (1, 2, 3, 4)

# Legacy sentinel nodata; V2 COGs use NaN (read_window zeroes it), guard both.
NODATA = -32768

MASK_SOURCES = ("graph", "raster")


def _hr_mask_path(dataset_dir, rel_image_path, mask_dirname):
    """`<split>/imagery/x.tif` -> `<split>/<mask_dirname>/x.tif`."""
    rel = Path(rel_image_path)
    return Path(dataset_dir) / rel.parent.parent / mask_dirname / rel.name


def _check_raster_masks(df, dataset_dir, mask_dirname, upscale):
    """Every tile needs an HR mask with dims exactly ``upscale`` x the tile's."""
    for rel in df["image_path"].drop_duplicates():
        hr = _hr_mask_path(dataset_dir, rel, mask_dirname)
        if not hr.exists():
            raise FileNotFoundError(
                f"No HR mask for {rel} under <split>/{mask_dirname}/. Generate "
                f"with OpenStreetMapTest/dataset_hr_masks.py --scale {upscale} "
                f"--out-dirname {mask_dirname}."
            )
        with rasterio.open(Path(dataset_dir) / rel) as s, rasterio.open(hr) as m:
            if (m.height, m.width) != (s.height * upscale, s.width * upscale):
                raise ValueError(
                    f"{rel}: HR mask is {m.height}x{m.width}, expected exactly "
                    f"{upscale}x the {s.height}x{s.width} tile."
                )


def _read_native(src, bands, window, crop_size):
    """(C, crop, crop) raw-DN float32: window read, nodata -> 0, edge zero-pad."""
    arr = read_window(src, bands, window)          # NaN/inf -> 0
    arr[arr == NODATA] = 0.0
    c, h, w = arr.shape
    if (h, w) != (crop_size, crop_size):
        pad = np.zeros((c, crop_size, crop_size), dtype="float32")
        pad[:, :h, :w] = arr
        arr = pad
    return arr


def _read_raster_hr_mask(dataset_dir, row, mask_dirname, window, out_size, upscale):
    """HR window of the pre-generated mask COG (edge zero-pad to ``out_size``)."""
    hr = _hr_mask_path(dataset_dir, row["image_path"], mask_dirname)
    win = Window(window.col_off * upscale, window.row_off * upscale,
                 int(window.width) * upscale, int(window.height) * upscale)
    with rasterio.open(hr) as src:
        m = (src.read(1, window=win) > 0).astype("float32")
    h, w = m.shape
    if (h, w) != (out_size, out_size):
        pad = np.zeros((out_size, out_size), dtype="float32")
        pad[:h, :w] = m
        m = pad
    return m


class JointSRRoadTileDataset(Dataset):
    """Random NATIVE crops from the train tiles + HR masks. ``__len__`` = patches/epoch."""

    def __init__(self, dataset_dir, bands=SR_INPUT_BANDS, crop_size=128, upscale=4,
                 length=None, min_road_density=0.0,
                 mask_source="graph", mask_dirname="masks_osm_2pt5m"):
        if mask_source not in MASK_SOURCES:
            raise ValueError(f"mask_source must be one of {MASK_SOURCES}, got {mask_source!r}")
        self.dataset_dir = Path(dataset_dir)
        df = _read_split_csv(dataset_dir, "train")
        if min_road_density > 0 and "road_density" in df.columns:
            n0 = len(df)
            df = df[df["road_density"] >= min_road_density]
            print(f"[joint_sr train] road_density >= {min_road_density}: "
                  f"kept {len(df)}/{n0} tiles")
        self.df = df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No train tiles left after the min_road_density filter.")
        self.bands = list(bands)
        self.crop_size = crop_size
        self.upscale = upscale
        self.out_size = crop_size * upscale
        self.mask_source = mask_source
        self.mask_dirname = mask_dirname
        self.length = length if length is not None else 10 * len(self.df)
        if mask_source == "raster":
            _check_raster_masks(self.df, dataset_dir, mask_dirname, upscale)

    def __len__(self):
        return self.length

    def _mask(self, row, src, win):
        if self.mask_source == "graph":
            return _graph_mask(self.dataset_dir / row["mask_graph_path"], src,
                               win, self.out_size, self.upscale)
        return _read_raster_hr_mask(self.dataset_dir, row, self.mask_dirname,
                                    win, self.out_size, self.upscale)

    def __getitem__(self, idx):
        row = self.df.iloc[random.randrange(len(self.df))]
        cn = self.crop_size
        with rasterio.open(self.dataset_dir / row["image_path"]) as src:
            H, W = src.height, src.width
            top = random.randint(0, max(0, H - cn))
            left = random.randint(0, max(0, W - cn))
            win = Window(left, top, min(cn, W - left), min(cn, H - top))
            img = _read_native(src, self.bands, win, cn)
            mask = self._mask(row, src, win)

        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_{top}_{left}.png"


class JointSRTileCropDataset(Dataset):
    """Deterministic eval crops: the non-overlapping NATIVE ``crop_size`` cells of
    each tile, each paired with its HR mask (once-per-pixel coverage)."""

    def __init__(self, dataset_dir, split, bands=SR_INPUT_BANDS, crop_size=128,
                 upscale=4, mask_source="graph", mask_dirname="masks_osm_2pt5m"):
        if mask_source not in MASK_SOURCES:
            raise ValueError(f"mask_source must be one of {MASK_SOURCES}, got {mask_source!r}")
        self.dataset_dir = Path(dataset_dir)
        self.df = _read_split_csv(dataset_dir, split).reset_index(drop=True)
        self.bands = list(bands)
        self.crop_size = crop_size
        self.upscale = upscale
        self.out_size = crop_size * upscale
        self.mask_source = mask_source
        self.mask_dirname = mask_dirname
        self.grid_w, self.grid_h = self._tile_grid()
        self.per_tile = self.grid_w * self.grid_h
        if mask_source == "raster" and not self.df.empty:
            _check_raster_masks(self.df, dataset_dir, mask_dirname, upscale)

    def _tile_grid(self):
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
            img = _read_native(src, self.bands, win, cn)
            if self.mask_source == "graph":
                mask = _graph_mask(self.dataset_dir / row["mask_graph_path"], src,
                                   win, self.out_size, self.upscale)
            else:
                mask = _read_raster_hr_mask(self.dataset_dir, row, self.mask_dirname,
                                            win, self.out_size, self.upscale)

        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_c{cell}.png"


class JointSRDataModule(pl.LightningDataModule):
    """Native-resolution crops + HR masks for joint SR+segmentation training.

    The model's input patch is ``crop_size`` native px; its OUTPUT (and the
    mask) is ``crop_size * upscale``. SEN2SR's shipped FFT mask pins
    ``crop_size`` to 128 (-> 512 HR). ``normalize``/``norm_mean``/``norm_std``
    are NOT applied here — they are declared so ``UNetCLI`` links them into the
    model, which normalises *after* super-resolution. ``image_size`` is
    vestigial (kept so the shared CLI link resolves).

    ``mask_source``: "graph" = pipeline labels (masks_graph parquet, CDNGI)
    rasterised at 2.5 m; "raster" = pre-generated HR masks in
    ``<split>/<mask_dirname>/`` (e.g. OSM, dataset_hr_masks.py --scale 4).
    """

    def __init__(self, dataset_dir: str, bands: tuple[int, ...] = SR_INPUT_BANDS,
                 batch_size: int = 4, num_workers: int = 2, crop_size: int = 128,
                 upscale: int = 4, image_size: int = 256, length: int | None = None,
                 normalize: bool = True, norm_mean: list[float] | None = None,
                 norm_std: list[float] | None = None, min_road_density: float = 0.0,
                 mask_source: str = "graph", mask_dirname: str = "masks_osm_2pt5m"):
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
        self.mask_source = mask_source
        self.mask_dirname = mask_dirname

    def _loader(self, ds, train):
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0, drop_last=train,
        )

    def train_dataloader(self):
        return self._loader(JointSRRoadTileDataset(
            self.dataset_dir, self.bands, self.crop_size, self.upscale,
            self.length, self.min_road_density, self.mask_source, self.mask_dirname,
        ), train=True)

    def _eval_loader(self, split):
        return self._loader(JointSRTileCropDataset(
            self.dataset_dir, split, self.bands, self.crop_size, self.upscale,
            self.mask_source, self.mask_dirname,
        ), train=False)

    def val_dataloader(self):
        return self._eval_loader("val")

    def test_dataloader(self):
        return self._eval_loader("test")
