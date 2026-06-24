"""
build_cog.py — combine the per-modality GeoTIFFs from GEE into ONE analysis-ready
multi-band COG per site, aligned to the 10 m S2 grid.

This is the adapter at the `imagery/` seam: it produces what a
DatasetManager / mask generator expects (one COG per tile)

Why int16: every band fits losslessly — S2 reflectance 0..10000, S1 stored as
dB*100 (~ -3000..+100), topo metres (0..~3500). int16 halves the file vs float32
for Kaggle upload. The Dataset casts to float and normalises at load.

Channel order written (14 bands, no topo by default):
    0-3   S2 10 m : B4 B3 B2 B8           (R, G, B, NIR)
    4-9   S2 20 m : B5 B6 B7 B8A B11 B12   (bicubic-upsampled to 10 m)
    10-13 S1       : VHA VVA VHD VVD        (asc VH/VV, desc VH/VV; dB*100)

Scale factors to apply downstream: S2 / 10000  ->  reflectance 0..1
                                   S1 / 100    ->  dB
                                   topo        ->  metres (no scaling)

Run locally (NOT on Kaggle) before packaging the dataset:
    python build_cog.py --base /Volumes/MAC_KIOXIA/Data --out /Volumes/MAC_KIOXIA/Data
"""
import argparse
from pathlib import Path
import os, pyproj
os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
os.environ.pop("PROJ_LIB", None) 
import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.enums import Resampling

# Band layout (no topo). Append topo names here if you enable --topo.
BAND_NAMES = ["B4", "B3", "B2", "B8",
              "B5", "B6", "B7", "B8A", "B11", "B12",
              "VHA", "VVA", "VHD", "VVD"]
NODATA = -32768  # int16 sentinel; safe — valid S1 dB*100 bottoms out around -3000


def _load(path):
    # float32 up front so cubic ringing can't wrap/clip during resampling.
    return rxr.open_rasterio(path, masked=True).astype("float32")


def build_site_cog(s2_10m_path, s2_20m_path, s1_path, out_path, topo_path=None):
    ref = _load(s2_10m_path)  # reference grid (4, H, W)

    s2_20m = _load(s2_20m_path).rio.reproject_match(ref, resampling=Resampling.cubic) if s2_20m_path is not None else None
    s1 = _load(s1_path).rio.reproject_match(ref, resampling=Resampling.bilinear) if s1_path is not None else None

    layers = [ref]
    if s2_20m is not None:
        layers.append(s2_20m)
    if s1 is not None:
        layers.append(s1)
    if topo_path is not None:
        layers.append(_load(topo_path).rio.reproject_match(ref, resampling=Resampling.bilinear))

    arr = np.concatenate([lyr.values for lyr in layers], axis=0)  # (C, H, W) float32, NaN=nodata

    # NaN -> sentinel, round to nearest int, cast int16.
    arr = np.where(np.isfinite(arr), np.rint(arr), NODATA).astype("int16")

    n_bands = arr.shape[0]
    if n_bands != len(BAND_NAMES):
        print(f"  ! {out_path.name}: {n_bands} bands but {len(BAND_NAMES)} names "
              f"(add topo names to BAND_NAMES if you enabled --topo)")

    profile = {
        "driver": "COG",
        "dtype": "int16",
        "count": n_bands,
        "height": arr.shape[1],
        "width": arr.shape[2],
        "crs": ref.rio.crs,
        "transform": ref.rio.transform(),
        "nodata": NODATA,
        "compress": "DEFLATE",
        "predictor": 2,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        for i in range(n_bands):
            name = BAND_NAMES[i] if i < len(BAND_NAMES) else f"band_{i+1}"
            dst.set_band_description(i + 1, name)
    return out_path


def find_sites(base):
    """Sites for which all required per-modality exports exist."""
    s2_10m_dir = base / "S2_10m"
    sites = []
    for p in sorted(s2_10m_dir.glob("*_S2_10m.tif")):
        sites.append(p.name.replace("_S2_10m.tif", ""))
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir containing S2_10m/, (S2_20m/), (S1/), (Sen12_topo/)")
    ap.add_argument("--out", required=True, help="output dataset dir; COGs go to <out>/imagery/")
    ap.add_argument("--topo", action="store_true", help="include topo as extra bands", default=False)
    ap.add_argument("--s2_20m", action="store_true", help="include S2 20m bands", default=False)
    ap.add_argument("--s1", action="store_true", help="include S1 bands", default=False)
    args = ap.parse_args()

    base = Path(args.base)
    imagery_dir = Path(args.out) / "imagery"

    sites = find_sites(base)
    print(f"Found {len(sites)} sites with S2 10 m exports.")
    for site in sites:
        s2_10m = base / "S2_10m" / f"{site}_S2_10m.tif"
        s2_20m = base / "S2_20m" / f"{site}_S2_20m.tif" if args.s2_20m else None
        s1 = base / "S1" / f"{site}_S1.tif" if args.s1 else None
        topo = base / "Sen12_topo" / f"{site}_topo.tif" if args.topo else None
        missing = [str(p) for p in (s2_20m, s1) if not p.exists()]
        if topo is not None and not topo.exists():
            missing.append(str(topo))
        if missing:
            print(f"[skip] {site}: missing {missing}")
            continue

        out = imagery_dir / f"{site}.tif"
        build_site_cog(s2_10m, s2_20m, s1, out, topo_path=topo)
        print(f"[ok]   {site} -> {out}")


if __name__ == "__main__":
    main()
