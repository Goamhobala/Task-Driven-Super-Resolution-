"""torchgeo data loading for the tiled S2-ROSA dataset.

The cut-tiles pipeline (``sentinel2data.generator.make_v2rosa_pipeline``, i.e.
``V2ROSAProcessor``) writes self-contained,
georeferenced 512x512 image + mask COGs under ``images/`` and ``masks/`` and a
per-tile ``metadata.parquet`` with a ``split_set`` column. Here those tiles are
wrapped as torchgeo ``RasterDataset``s, intersected, and sampled into 256x256
patches by a torchgeo ``GeoSampler`` (random for train, grid for val/test).

    from sentinel2data.torchgeo_dataset import get_dataloader
    loader = get_dataloader("/data/S2ROSA_tiled", split="train", batch_size=8)
    batch = next(iter(loader))          # {'image': BxCx256x256, 'mask': Bx256x256, ...}

Because the source zones span several UTM zones, all tiles in one split are
reprojected on the fly to the first tile's CRS (torchgeo default); patches are
therefore not strictly pixel-aligned across UTM-zone boundaries.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader

from torchgeo.datasets import IntersectionDataset, RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

# 20-band COG layout (1-based on disk); see s2rosa dataset memory.
S2_BANDS = (
    "B4", "B3", "B2", "B8", "B5", "B6", "B7", "B8A", "B11", "B12",
    "VV_ascending", "VH_ascending", "VV_descending", "VH_descending",
    "elevation", "slope", "aspect",
    "esa_urban_10m", "gisa_urban_10m", "wsf_urban_10m",
)
RGB_BANDS = ("B4", "B3", "B2")

# S2-ROSA-V2 imagery appends 3 CLAHE+gamma enhanced-RGB bands (B4,B3,B2 -> bands 21-23).
ENHANCED_RGB_BANDS = ("B4_clahe", "B3_clahe", "B2_clahe")
S2_V2_BANDS = S2_BANDS + ENHANCED_RGB_BANDS


class S2RosaImage(RasterDataset):
    """20-band S2-ROSA image tiles (float reflectance / terrain / SAR)."""

    filename_glob = "*.tif"
    is_image = True
    all_bands = S2_BANDS
    rgb_bands = RGB_BANDS


class S2RosaV2Image(RasterDataset):
    """23-band S2-ROSA-V2 image tiles (20 source bands + 3 enhanced-RGB bands)."""

    filename_glob = "*.tif"
    is_image = True
    all_bands = S2_V2_BANDS
    rgb_bands = RGB_BANDS


class S2RosaMask(RasterDataset):
    """Binary road-mask tiles paired with :class:`S2RosaImage` by location."""

    filename_glob = "*.tif"
    is_image = False  # -> sample key 'mask', cast to long


def _split_paths(dataset_dir, split):
    """Resolve (image_paths, mask_paths) for one split from its splits CSV,
    falling back to metadata.parquet."""
    dataset_dir = Path(dataset_dir)
    csv = dataset_dir / "splits" / f"{split}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        df = pd.read_parquet(dataset_dir / "metadata.parquet",
                             columns=["image_path", "mask_path", "split_set"])
        df = df[df["split_set"].astype(str).str.lower() == split.lower()]
    if df.empty:
        raise ValueError(f"No tiles for split {split!r} in {dataset_dir}")
    imgs = [str(dataset_dir / p) for p in df["image_path"]]
    msks = [str(dataset_dir / p) for p in df["mask_path"]]
    return imgs, msks


WGS84 = "EPSG:4326"


def build_dataset(dataset_dir, split="train", bands=RGB_BANDS, crs=WGS84,
                  image_cls=S2RosaImage):
    """Build the intersected image&mask torchgeo dataset for one split.

    ``bands`` selects which of ``image_cls.all_bands`` to read (default RGB).
    ``image_cls`` is the image ``RasterDataset`` (default :class:`S2RosaImage`,
    20-band; pass :class:`S2RosaV2Image` for the 23-band S2-ROSA-V2 imagery).
    ``crs`` is the common working CRS every tile is reprojected to (default WGS84
    so all UTM zones share one grid); ``crs=None`` keeps the first tile's native
    CRS (no warp)."""
    img_paths, msk_paths = _split_paths(dataset_dir, split)
    if crs is None:
        with rasterio.open(img_paths[0]) as src:
            crs = src.crs

    image = image_cls(paths=img_paths, bands=list(bands), crs=crs)
    mask = S2RosaMask(paths=msk_paths, crs=crs)
    return image & mask  # IntersectionDataset -> {'image','mask','bounds','crs'}


def get_dataloader(
    dataset_dir,
    split="train",
    bands=RGB_BANDS,
    patch_size=256,
    batch_size=8,
    num_workers=0,
    length=None,
    stride=None,
    crs=WGS84,
):
    """DataLoader of 256x256 patches sampled from the split's tiles.

    train -> ``RandomGeoSampler`` (``length`` patches/epoch, default 100*n_tiles);
    val/test -> ``GridGeoSampler`` (dense, ``stride`` defaults to ``patch_size``).
    ``crs`` defaults to EPSG:4326 (all tiles reprojected to WGS84); ``crs=None``
    keeps the first tile's native CRS.
    """
    dataset = build_dataset(dataset_dir, split=split, bands=bands, crs=crs)

    if split == "train":
        if length is None:
            length = 100 * len(dataset.datasets[0].index)
        sampler = RandomGeoSampler(dataset, size=patch_size, length=length)
        shuffle = False  # sampler already randomises
    else:
        sampler = GridGeoSampler(
            dataset, size=patch_size, stride=stride or patch_size
        )
        shuffle = False

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=stack_samples,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the torchgeo loader.")
    ap.add_argument("dataset_dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--all-bands", action="store_true", help="read all 20 bands")
    args = ap.parse_args()

    bands = S2_BANDS if args.all_bands else RGB_BANDS
    loader = get_dataloader(
        args.dataset_dir, split=args.split, bands=bands, batch_size=4, length=8
    )
    batch = next(iter(loader))
    print("image", tuple(batch["image"].shape), batch["image"].dtype)
    print("mask ", tuple(batch["mask"].shape), batch["mask"].dtype,
          "road frac", float((batch["mask"] > 0).float().mean()))
