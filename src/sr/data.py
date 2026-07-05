"""Dataset plumbing for the R0/R1/R2 resolution-enhancement experiments.

Same on-disk layout and site-level split as the baseline (`baseline.data` /
`sentinel2data.processor.dataset`), with two deliberate differences:

  * The image patch is returned as RAW surface reflectance (DN / 10000) in
    SEN2SR's band order [B4, B3, B2, B8] = R, G, B, NIR — NOT z-scored. SEN2SR
    was trained on reflectance, so normalisation for the segmentation encoder
    happens *after* super-resolution, inside `JointSRSegModule` (a
    differentiable affine using the same frozen Data.npz stats the baseline
    uses). Our combined COGs already store bands in this order (BAND_NAMES
    starts B4, B3, B2, B8), so the M0 channel slice needs no permutation.
  * The mask comes from the 2.5 m directory (`mask_2pt5m/`) and the window is
    read at `scale`× the imagery offsets, so a 128×128 @10 m patch pairs with
    a 512×512 @2.5 m mask. Grid alignment (HR dims == scale × LR dims) is
    asserted per site at init rather than assumed.

Augmentation is the baseline's lossless default (D4 flips/rotations) only,
applied to LR image and HR mask with the same group element via torch ops —
Albumentations can't jointly transform an image/mask pair of different sizes.
The photometric extras from `baseline.augment` are tuned for z-scored input
and are deliberately not carried over to reflectance space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from baseline.data import compute_pos_weight, resolve_mask_suffix  # noqa: F401  (re-exported)
from sentinel2data.processor.dataset import NODATA, _offsets, list_sites, split_sites

__all__ = [
    "SRRoadSegDataset",
    "build_splits",
    "resolve_mask_suffix",
    "compute_pos_weight",
    "loader_kwargs",
    "REFLECTANCE_SCALE",
    "SR_CHANNELS",
]

# DN -> surface reflectance divisor expected by SEN2SR.
REFLECTANCE_SCALE = 10000.0

# Band indices of [B4, B3, B2, B8] in the combined 14-band COG (== the M0
# channel group). Already SEN2SR's [R, G, B, NIR] order — see module docstring.
SR_CHANNELS = [0, 1, 2, 3]


def build_splits(imagery_dir):
    """All sites discovered under `imagery_dir`, partitioned train/val/test."""
    return split_sites(list_sites(imagery_dir))


def loader_kwargs(num_workers: int) -> dict:
    """DataLoader kwargs matching baseline.train: spawn workers (CUDA + GDAL
    are not fork-safe), persistent workers, pinned memory."""
    kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
    else:
        kwargs["pin_memory"] = False
    return kwargs


def _apply_d4(x: torch.Tensor, y: torch.Tensor, k: int):
    """Apply the k-th element of the D4 dihedral group (k in [0, 8)) to both
    tensors. Works across resolutions because rot90/flip act on relative axes."""
    if k % 4:
        x = torch.rot90(x, k % 4, dims=(-2, -1))
        y = torch.rot90(y, k % 4, dims=(-2, -1))
    if k >= 4:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))
    return x.contiguous(), y.contiguous()


class SRRoadSegDataset(Dataset):
    """10 m RGB+NIR reflectance patches paired with 2.5 m road-mask patches.

    Yields `(x, y)`: x float32 (4, P, P) reflectance in [B4, B3, B2, B8] order,
    y float32 (1, scale·P, scale·P) binary. `d4=True` adds random D4
    flips/rotations (train split only; leave off for val/test).
    """

    def __init__(self, imagery_dir, hr_masks_dir, sites, patch_size=128,
                 stride=128, mask_suffix="_mask.tif", scale=4, d4=False):
        self.imagery_dir = Path(imagery_dir)
        self.hr_masks_dir = Path(hr_masks_dir)
        self.patch = patch_size
        self.scale = scale
        self.d4 = d4

        self.items = []  # (img_path, hr_mask_path, row_off, col_off) at 10 m
        for s in sites:
            img = self.imagery_dir / f"{s}.tif"
            mask = self.hr_masks_dir / f"{s}{mask_suffix}"
            if not img.exists() or not mask.exists():
                print(f"[sr.data] skip {s}: missing image or 2.5m mask")
                continue
            with rasterio.open(img) as src:
                H, W = src.height, src.width
            with rasterio.open(mask) as src:
                mH, mW = src.height, src.width
            # The 2.5 m masks must be rasterised on the imagery grid upsampled
            # by exactly `scale` (road_mask.py rasterises onto the reference
            # raster's transform/shape, so this holds when the 2.5 m reference
            # grid was derived from the same tile). Fail loudly otherwise.
            if (mH, mW) != (H * self.scale, W * self.scale):
                raise ValueError(
                    f"{s}: 2.5m mask is {mH}x{mW} but imagery is {H}x{W} "
                    f"(expected exactly {self.scale}x). Re-check mask generation."
                )
            for r in _offsets(H, patch_size, stride):
                for c in _offsets(W, patch_size, stride):
                    self.items.append((img, mask, r, c))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, r, c = self.items[idx]

        with rasterio.open(img_path) as src:
            x = src.read([ch + 1 for ch in SR_CHANNELS],
                         window=Window(c, r, self.patch, self.patch)).astype("float32")
        hr = self.patch * self.scale
        with rasterio.open(mask_path) as src:
            m = src.read(1, window=Window(c * self.scale, r * self.scale, hr, hr))

        x[x == NODATA] = 0.0            # nodata -> 0 BEFORE scaling (== zero reflectance)
        x /= REFLECTANCE_SCALE          # DN -> surface reflectance, SEN2SR's input space
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        y = (m > 0).astype("float32")[None, ...]  # (1, sP, sP) binary

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.d4:
            # Per-worker RNG (seeded by Lightning's seed_everything(workers=True)).
            x, y = _apply_d4(x, y, int(np.random.randint(8)))
        return x, y
