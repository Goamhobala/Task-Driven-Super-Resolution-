"""
Align InstaRoad multimodal tiles onto one common 10 m grid and stack to (C, H, W).

Uses rioxarray/rasterio (geo-aware, GDAL under the hood) — NOT cv2/PIL/scipy — so
the geotransform is respected and every modality lands on exactly the same pixel
grid as the reference. Run: pip install rioxarray rasterio

Reference grid = the S2 10 m tile. Everything else is reproject_match'd onto it:
  - S2 20 m  -> upsampled to 10 m with bicubic (cubic convolution); matches M1.
  - S1       -> already 10 m, so this is a near-identity that just guarantees the
                grid/origin line up exactly (bilinear; use nearest for bit-exact).
  - topo     -> bilinear (continuous elevation).

Output channel order:
  [B4,B3,B2,B8]                      S2 10 m
  [B5,B6,B7,B8A,B11,B12]             S2 20 m -> 10 m
  [VHA,VVA,VHD,VVD]                  S1
  (+ topo bands if provided)

Values are returned as float32, UNSCALED (still in stored units):
  S2 = 0..10000 reflectance  -> divide by 10000 for 0..1
  S1 = dB * 100 (int16)      -> divide by 100 for dB
Do that scaling in your normalisation step, not here.
"""
import numpy as np
import rioxarray as rxr
from rasterio.enums import Resampling


def _load(path):
    # float32 up front so bicubic ringing can't wrap a uint16 / clip an int16.
    return rxr.open_rasterio(path, masked=True).astype("float32")


def align_site(s2_10m_path, s2_20m_path, s1_path):
    ref = _load(s2_10m_path)  # (4, H, W), e.g. 4 x 2500 x 2500 — the reference grid
    if s2_20m_path is not None:
        s2_20m = _load(s2_20m_path).rio.reproject_match(ref, resampling=Resampling.cubic)
    if s1_path is not None:
        s1     = _load(s1_path).rio.reproject_match(ref, resampling=Resampling.bilinear)

    layers = [ref]
    if s2_20m_path is not None:
        layers.append(s2_20m)
    if s1_path is not None:
        layers.append(s1)
    # if topo_path is not None:
    #     layers.append(_load(topo_path).rio.reproject_match(ref, resampling=Resampling.bilinear))

    arr = np.concatenate([lyr.values for lyr in layers], axis=0)  # (C, H, W)
    return arr, ref.rio.transform(), ref.rio.crs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=".")
    p.add_argument("--site", default="Thohoyandou")
    p.add_argument("--with_20m", default=None)
    p.add_argument("--with_s1", default=None)
    args = p.parse_args()
    base, site = args.base, args.site

    arr, transform, crs = align_site(
        f"{base}/{site}_S2_10m.tif",
        f"{base}/{site}_S2_20m.tif" if args.with_20m else None,
        f"{base}/{site}_S1.tif" if args.with_s1 else None,
        # topo_path=f"{base}/{site}_topo.tif",  # set None to skip
    )
    print(f"{site}: stacked shape {arr.shape}  dtype {arr.dtype}")
    print(f"  CRS {crs}")
    print(f"  all channels now share one {arr.shape[1]}x{arr.shape[2]} grid")