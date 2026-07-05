"""Dataset plumbing for the R0/R1/R2 resolution-enhancement experiments.

Reads the V2 tiled layout written by `sentinel2data generate` (split CSVs
under `<root>/splits/`, 512x512 tile COGs under `<root>/<split>/imagery/`),
with two deliberate differences from the friend-side loaders
(`sentinel2data.dataset.datasets` / `upscale_dataset`):

  * The image patch is returned as RAW surface reflectance (DN / 10000) in
    SEN2SR's band order [B4, B3, B2, B8] = R, G, B, NIR — NOT z-scored. SEN2SR
    was trained on reflectance, so normalisation for the segmentation encoder
    happens *after* super-resolution, inside `JointSRSegModule` (a
    differentiable affine using the frozen norm-stats the baseline uses).
    Bands 1-4 of the V2 COGs are exactly this order
    (`sentinel2data.dataset.bands.S2_10M`), so no permutation is needed.
  * The mask does NOT come from the pipeline's `masks_raster/` (10 m) or the
    `masks_graph` parquets. It comes from the pre-generated OSM HR masks
    (`OpenStreetMapTest/dataset_hr_masks.py`) that live beside the imagery:
    `<split>/masks_osm_2pt5m/{tile}.tif`, rasterised on the tile grid
    upsampled by exactly `scale` — so a 128x128 @10 m patch pairs with a
    512x512 @2.5 m mask. Grid alignment (HR dims == scale x LR dims) is
    asserted per tile at init rather than assumed. The only contract between
    the two repos is this on-disk layout.

Patch enumeration is the deterministic non-overlapping grid (`stride` =
`patch_size` -> 16 patches per 512 tile), matching the R-series design of
identical patch sets across R0/R1/R2, rather than the random-crop scheme of
`RoadTileDataset`.

Augmentation is the lossless D4 (flips/rotations) only, applied to LR image
and HR mask with the same group element via torch ops — Albumentations can't
jointly transform an image/mask pair of different sizes. Photometric extras
are tuned for z-scored input and are deliberately not applied to reflectance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from sentinel2data.dataset.bands import S2_10M

__all__ = [
    "SRRoadSegDataset",
    "compute_pos_weight",
    "loader_kwargs",
    "REFLECTANCE_SCALE",
    "SR_BANDS",
    "HR_MASKS_DIRNAME",
]

# DN -> surface reflectance divisor expected by SEN2SR.
REFLECTANCE_SCALE = 10000.0

# Legacy sentinel nodata; V2 COGs use NaN, but guard both.
NODATA = -32768

# 1-based rasterio band ids of [B4, B3, B2, B8] in the V2 COG (== the M0
# channel group). Already SEN2SR's [R, G, B, NIR] order — see module docstring.
SR_BANDS = tuple(S2_10M)

# Where dataset_hr_masks.py writes the OSM HR masks, relative to each split dir.
HR_MASKS_DIRNAME = "masks_osm_2pt5m"


def read_split_csv(dataset_dir, split) -> pd.DataFrame:
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


def hr_mask_path(dataset_dir, rel_image_path, dirname=HR_MASKS_DIRNAME) -> Path:
    """`<split>/imagery/x.tif` -> `<split>/<dirname>/x.tif` (dataset_hr_masks.py's layout)."""
    rel = Path(rel_image_path)
    return Path(dataset_dir) / rel.parent.parent / dirname / rel.name


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


def _offsets(size: int, patch: int, stride: int) -> list[int]:
    """Window offsets covering `size` px: 0, stride, ... with the last window
    clamped inside the raster (so every pixel is covered exactly once when
    stride == patch and size % patch == 0, e.g. 512/128 -> [0,128,256,384])."""
    if size <= patch:
        return [0]
    offs = list(range(0, size - patch + 1, stride))
    if offs[-1] != size - patch:
        offs.append(size - patch)
    return offs


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


def compute_pos_weight(dataset_dir, split="train",
                       hr_masks_dirname=HR_MASKS_DIRNAME) -> torch.Tensor:
    """BCE pos_weight = background/road pixel ratio over the split's HR masks."""
    df = read_split_csv(dataset_dir, split)
    pos = neg = 0
    for rel in df["image_path"].drop_duplicates():
        mask = hr_mask_path(dataset_dir, rel, hr_masks_dirname)
        if not mask.exists():
            continue
        with rasterio.open(mask) as src:
            m = src.read(1)
        p = int(np.count_nonzero(m > 0))
        pos += p
        neg += m.size - p
    if pos == 0:
        raise ValueError(
            f"No road pixels found in any {split} HR mask under "
            f"{Path(dataset_dir)}/<split>/{hr_masks_dirname}/ — wrong dir?"
        )
    return torch.tensor(neg / pos, dtype=torch.float32)


class SRRoadSegDataset(Dataset):
    """10 m RGB+NIR reflectance patches paired with 2.5 m OSM road-mask patches.

    Yields `(x, y)`: x float32 (4, P, P) reflectance in [B4, B3, B2, B8] order,
    y float32 (1, scale·P, scale·P) binary. `d4=True` adds random D4
    flips/rotations (train split only; leave off for val/test).
    `min_road_density` drops tiles below that road fraction (use on train only;
    mirrors `UpscaleRoadDataModule`'s filter for comparability with the R0'
    bicubic runs — val/test must stay unfiltered).
    """

    def __init__(self, dataset_dir, split, hr_masks_dirname=HR_MASKS_DIRNAME,
                 patch_size=128, stride=128, scale=4, d4=False,
                 min_road_density=0.0):
        self.dataset_dir = Path(dataset_dir)
        self.patch = patch_size
        self.scale = scale
        self.d4 = d4

        df = read_split_csv(dataset_dir, split)
        if min_road_density > 0 and "road_density" in df.columns:
            n0 = len(df)
            df = df[df["road_density"] >= min_road_density]
            print(f"[sr.data:{split}] road_density >= {min_road_density}: "
                  f"kept {len(df)}/{n0} tiles")
        if df.empty:
            raise ValueError(f"No {split} tiles (after filtering) in {dataset_dir}")

        self.items = []  # (img_path, hr_mask_path, row_off, col_off) at 10 m
        skipped = 0
        for rel in df["image_path"].drop_duplicates():
            img = self.dataset_dir / rel
            mask = hr_mask_path(self.dataset_dir, rel, hr_masks_dirname)
            if not img.exists() or not mask.exists():
                skipped += 1
                continue
            with rasterio.open(img) as src:
                H, W = src.height, src.width
            with rasterio.open(mask) as src:
                mH, mW = src.height, src.width
            # dataset_hr_masks.py rasterises onto the tile's transform scaled
            # by exactly `scale`, so this must hold. Fail loudly otherwise.
            if (mH, mW) != (H * self.scale, W * self.scale):
                raise ValueError(
                    f"{rel}: HR mask is {mH}x{mW} but imagery is {H}x{W} "
                    f"(expected exactly {self.scale}x). Re-run dataset_hr_masks.py "
                    f"with --scale {self.scale}."
                )
            for r in _offsets(H, patch_size, stride):
                for c in _offsets(W, patch_size, stride):
                    self.items.append((img, mask, r, c))
        if skipped:
            print(f"[sr.data:{split}] skipped {skipped} tiles with no image or "
                  f"HR mask ({hr_masks_dirname}/) — run dataset_hr_masks.py?")
        if not self.items:
            raise ValueError(
                f"No usable {split} tiles: no HR masks found under "
                f"<split>/{hr_masks_dirname}/. Generate them with "
                f"OpenStreetMapTest/dataset_hr_masks.py first."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, r, c = self.items[idx]

        with rasterio.open(img_path) as src:
            x = src.read(list(SR_BANDS),
                         window=Window(c, r, self.patch, self.patch)).astype("float32")
        hr = self.patch * self.scale
        with rasterio.open(mask_path) as src:
            m = src.read(1, window=Window(c * self.scale, r * self.scale, hr, hr))

        x[x == NODATA] = 0.0            # legacy nodata -> 0 BEFORE scaling
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # V2 nodata is NaN
        x /= REFLECTANCE_SCALE          # DN -> surface reflectance, SEN2SR's input space

        y = (m > 0).astype("float32")[None, ...]  # (1, sP, sP) binary

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.d4:
            # Per-worker RNG (seeded by Lightning's seed_everything(workers=True)).
            x, y = _apply_d4(x, y, int(np.random.randint(8)))
        return x, y
