"""
dataset.py — torch Dataset over the pre-built COGs + pre-generated masks.

No mask generation at runtime: it just reads files, so the whole thing can be
zipped and uploaded as a Kaggle dataset.

Key choices:
  * SITE-LEVEL split (whole sites to train/val/test) — patches from one site
    never straddle splits, so no spatial-autocorrelation leakage.
  * Channel selection via CHANNEL_GROUPS picks the M0-M3 modality config.
  * Per-band z-score using FROZEN training stats (norm_stats.npz). Same stats
    for val/test.
  * Windowed reads (rasterio) so we never load a whole 2600x2600 tile per patch.
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

NODATA = -32768

# Full 14-band layout written by build_cog.py.
BAND_NAMES = ["B4", "B3", "B2", "B8",
              "B5", "B6", "B7", "B8A", "B11", "B12",
              "VHA", "VVA", "VHD", "VVD"]

# Modality ablation -> band indices into the 14-band stack.
CHANNEL_GROUPS = {
    "M0": [0, 1, 2, 3],                          # RGB + NIR
    "M1": list(range(10)),                       # + S2 20 m bands
    "M2": [0, 1, 2, 3, 10, 11, 12, 13],          # RGB+NIR + S1
    "M3": list(range(14)),                       # everything
}

# Default site-level split (20 ChatGPT-suggested sites). Edit freely.
# Held out for val/test were picked to span density regimes / biomes.
VAL_SITES = ["Worcester", "Mtubatuba"]
TEST_SITES = ["Makhanda", "Nongoma", "Upington"]


def split_sites(all_sites, val_sites=VAL_SITES, test_sites=TEST_SITES):
    val = [s for s in all_sites if s in val_sites]
    test = [s for s in all_sites if s in test_sites]
    train = [s for s in all_sites if s not in val_sites and s not in test_sites]
    return {"train": train, "val": val, "test": test}


def list_sites(imagery_dir):
    return sorted(p.stem for p in Path(imagery_dir).glob("*.tif"))


def _offsets(n, patch, stride):
    """Top-left offsets that tile [0, n) with full `patch`-sized windows."""
    if n <= patch:
        return [0]
    offs = list(range(0, n - patch + 1, stride))
    if offs[-1] != n - patch:
        offs.append(n - patch)  # snap the last window flush to the edge
    return offs


class RoadSegDataset(Dataset):
    def __init__(self, imagery_dir, masks_dir, sites, stats_path,
                 config="M3", patch_size=256, stride=256,
                 mask_suffix="_mask.tif", min_road_frac=0.0):
        self.imagery_dir = Path(imagery_dir)
        self.masks_dir = Path(masks_dir)
        self.channels = CHANNEL_GROUPS[config]
        self.patch = patch_size
        self.mask_suffix = mask_suffix

        stats = np.load(stats_path)
        self.mean = stats["mean"][self.channels].astype("float32")[:, None, None]
        self.std = stats["std"][self.channels].astype("float32")[:, None, None]
        self.std[self.std == 0] = 1.0  # guard constant bands

        # Build the patch index up front (cheap: only reads tile dimensions).
        self.items = []  # (img_path, mask_path, row_off, col_off)
        for s in sites:
            img = self.imagery_dir / f"{s}.tif"
            mask = self.masks_dir / f"{s}{mask_suffix}"
            if not img.exists() or not mask.exists():
                print(f"[dataset] skip {s}: missing image or mask")
                continue
            with rasterio.open(img) as src:
                H, W = src.height, src.width
            for r in _offsets(H, patch_size, stride):
                for c in _offsets(W, patch_size, stride):
                    self.items.append((img, mask, r, c))

        self.min_road_frac = min_road_frac  # optionally drop near-empty patches (training only)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, r, c = self.items[idx]
        win = Window(c, r, self.patch, self.patch)

        with rasterio.open(img_path) as src:
            x = src.read([ch + 1 for ch in self.channels], window=win).astype("float32")
        with rasterio.open(mask_path) as src:
            m = src.read(1, window=win)

        valid = x != NODATA
        x = (x - self.mean) / self.std
        x[~valid] = 0.0  # nodata -> 0 (== band mean after z-score)

        y = (m > 0).astype("float32")[None, ...]  # (1, H, W) binary

        return torch.from_numpy(x), torch.from_numpy(y)
