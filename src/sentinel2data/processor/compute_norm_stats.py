"""
compute_norm_stats.py — per-band mean/std over the TRAINING sites only.

Run locally after build_cog.py, then ship the resulting norm_stats.npz inside the
Kaggle dataset. Stats are frozen and reused for val/test (no leakage).

  * nodata (and any non-finite) excluded.
  * SAR bands clipped to a sane dB range before stats so speckle outliers don't
    blow up the std. Stored values are dB*100, so [-30, +5] dB -> [-3000, 500].
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

from dataset import BAND_NAMES, NODATA, list_sites, split_sites

# Indices of the SAR bands within the 14-band stack.
SAR_IDX = [BAND_NAMES.index(b) for b in ("VVA", "VHA", "VVD", "VHD")]
SAR_CLIP_STORED = (-3000, 500)  # = [-30, +5] dB at dB*100


def compute(imagery_dir, train_sites, clip_sar=True):
    n_bands = len(BAND_NAMES)
    s = np.zeros(n_bands, dtype="float64")
    ss = np.zeros(n_bands, dtype="float64")
    cnt = np.zeros(n_bands, dtype="float64")

    for site in train_sites:
        path = Path(imagery_dir) / f"{site}.tif"
        if not path.exists():
            print(f"[stats] skip {site}: not found")
            continue
        with rasterio.open(path) as src:
            arr = src.read().astype("float64")  # (C, H, W)
        for b in range(n_bands):
            band = arr[b]
            valid = np.isfinite(band) & (band != NODATA)
            v = band[valid]
            if clip_sar and b in SAR_IDX:
                v = np.clip(v, *SAR_CLIP_STORED)
            s[b] += v.sum()
            ss[b] += (v * v).sum()
            cnt[b] += v.size
        print(f"[stats] accumulated {site}")

    cnt[cnt == 0] = 1.0
    mean = s / cnt
    var = np.maximum(ss / cnt - mean ** 2, 0.0)
    std = np.sqrt(var)
    return mean.astype("float32"), std.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagery", required=True, help="<dataset>/imagery dir of combined COGs")
    ap.add_argument("--out", default=None, help="output .npz (default: <imagery>/../norm_stats.npz)")
    ap.add_argument("--no-clip-sar", action="store_true")
    args = ap.parse_args()

    imagery = Path(args.imagery)
    all_sites = list_sites(imagery)
    train = split_sites(all_sites)["train"]
    print(f"Computing stats over {len(train)} training sites: {train}")

    mean, std = compute(imagery, train, clip_sar=not args.no_clip_sar)
    out = Path(args.out) if args.out else imagery.parent / "norm_stats.npz"
    np.savez(out, mean=mean, std=std, bands=np.array(BAND_NAMES))

    print(f"\nSaved {out}")
    for b, m, sd in zip(BAND_NAMES, mean, std):
        print(f"  {b:>4}: mean={m:10.2f}  std={sd:10.2f}")


if __name__ == "__main__":
    main()
