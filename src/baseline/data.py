"""Dataset plumbing for the baseline.

Reuses `sentinel2data.processor.dataset.RoadSegDataset` (windowed reads over the
combined COGs, per-band z-score with the frozen Data.npz stats, site-level
split) so training, validation and benchmarking all share one normalisation and
one split definition. Adds only what the baseline needs on top:

  * `resolve_mask_suffix` — discover how the masks in `mask_10m/` are named
    (`{site}_mask.tif`, `{site}.tif`, ...) instead of hard-coding it.
  * `compute_pos_weight` — BCE positive weight from the actual road frequency.
  * `BenchDataset` — also yields the `chip_id`/`tile_id` the benchmarking store
    keys on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from sentinel2data.processor.dataset import (
    CHANNEL_GROUPS,
    NODATA,
    RoadSegDataset,
    list_sites,
    split_sites,
)

__all__ = [
    "CHANNEL_GROUPS",
    "resolve_mask_suffix",
    "build_splits",
    "make_dataset",
    "compute_pos_weight",
    "AugmentedRoadSegDataset",
    "BenchDataset",
]


class AugmentedRoadSegDataset(RoadSegDataset):
    """RoadSegDataset that applies an Albumentations transform per patch.

    The base class returns the normalised image `(C, H, W)` and binary mask
    `(1, H, W)` as tensors; we hand the same spatial transform to both so they
    stay pixel-aligned, then convert back. With `transform=None` it is exactly
    `RoadSegDataset`, so the same class can serve augmented (train) and plain
    (val/test) splits. Albumentations wants channel-last numpy, hence the
    permutes around the call.
    """

    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if self.transform is None:
            return x, y
        img = x.permute(1, 2, 0).numpy()   # (H, W, C)
        mask = y[0].numpy()                 # (H, W)
        out = self.transform(image=img, mask=mask)
        x = torch.from_numpy(np.ascontiguousarray(out["image"].transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(out["mask"]))[None, ...]
        return x, y


def resolve_mask_suffix(masks_dir, sites, default="_mask.tif") -> str:
    """Infer the mask filename suffix from the first site that has a mask.

    The mask generator lets the caller name the output, so `mask_10m/` may hold
    `{site}_mask.tif` or `{site}.tif`. Pick the suffix off a real file rather
    than guessing; fall back to `default` if nothing matches.
    """
    masks_dir = Path(masks_dir)
    for s in sites:
        for p in sorted(masks_dir.glob(f"{s}*.tif")):
            return p.name[len(s):]
    return default


def build_splits(imagery_dir):
    """All sites discovered under `imagery_dir`, partitioned train/val/test."""
    return split_sites(list_sites(imagery_dir))


def make_dataset(imagery_dir, masks_dir, sites, stats_path, config,
                 patch_size=256, stride=256, mask_suffix="_mask.tif",
                 transform=None):
    """Build the patch dataset, optionally with a per-patch augmentation.

    Pass `transform` (an Albumentations Compose, see `baseline.augment`) for the
    training split; leave it None for val/test so they stay deterministic.
    """
    return AugmentedRoadSegDataset(
        imagery_dir, masks_dir, sites, stats_path,
        config=config, patch_size=patch_size, stride=stride,
        mask_suffix=mask_suffix, transform=transform,
    )


def compute_pos_weight(masks_dir, sites, mask_suffix, cap=50.0) -> torch.Tensor:
    """BCE `pos_weight` = (#background / #road) over the training masks.

    Reads each training mask once (whole tile, uint8 — cheap) and pools the
    pixel counts. Capped so an almost-empty tile set can't blow the weight up.
    """
    masks_dir = Path(masks_dir)
    pos = neg = 0
    for s in sites:
        mp = masks_dir / f"{s}{mask_suffix}"
        if not mp.exists():
            continue
        with rasterio.open(mp) as src:
            m = src.read(1)
        p = int((m > 0).sum())
        pos += p
        neg += m.size - p
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(min(neg / pos, cap), dtype=torch.float32)


class BenchDataset(RoadSegDataset):
    """RoadSegDataset that also returns the chip/tile identity per patch.

    `chip_id = {site}_r{row_off}_c{col_off}` is the benchmarking unit; `tile_id`
    is the parent site. With stride == patch_size the chips tile the site
    without overlap, so each is scored exactly once.
    """

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        img_path, _mask_path, r, c = self.items[idx]
        tile_id = img_path.stem
        chip_id = f"{tile_id}_r{r}_c{c}"
        return x, y, chip_id, tile_id
