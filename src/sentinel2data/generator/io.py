"""COG and catalogue writers"""
from pathlib import Path
import numpy as np
import rasterio
from sentinel2data.generator.config import RASTER_EXTS
from sentinel2data.generator.helper import stretch_bands

# Loading
def load_image_rgb(path, bands=(1, 2, 3), percentile_range=(2, 98)):
    """Load an RGB image as a BGR uint8 array (OpenCV convention).

    Rasters are read via rasterio and percentile-stretched to 8-bit if not
    already uint8; PNG/JPG are read directly via OpenCV.
    """
    import cv2 # lazy import

    if not is_raster(path):
        return cv2.imread(str(path))  # already BGR uint8

    with rasterio.open(path) as src:
        arr = src.read(bands)  # (3, H, W), band order R, G, B

    if arr.dtype == np.uint8:
        rgb = np.transpose(arr, (1, 2, 0))
    else:
        rgb = np.transpose(stretch_bands(arr.astype(np.float32), percentile_range), (1, 2, 0))

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_binary_mask(path, band=1):
    """Load a road mask as a 0/255 uint8 array (any non-zero pixel is road)."""
    if is_raster(path):
        with rasterio.open(path) as src:
            arr = src.read(band)
    else:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if arr is None:
        raise ValueError(f"Failed to load mask from {path}. Arr is None")

    return np.where(arr > 0, 255, 0).astype(np.uint8)

def is_raster(path):
    return Path(path).suffix.lower() in RASTER_EXTS


# Writing
def _set_tiling(profile, tiled, blockxsize, blockysize):
    """Configure internal tiling. ``tiled=False`` -> plain (striped) GTiff."""
    if tiled:
        profile.update(tiled=True, blockxsize=blockxsize, blockysize=blockysize)
    else:
        profile["tiled"] = False
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)


def write_image_cog(path, arr, profile, *, transform, blockxsize=None,
                    blockysize=None, tiled=True):
    """Write a multi-band image array ``(C, H, W)``, deflate-compressed.

    ``count`` and ``dtype`` are taken from ``arr`` (so appended enhanced bands
    are handled); ``profile`` supplies crs etc. ``tiled=False`` writes a plain
    (non-COG) GTiff -- used for the raw 512px V2 train tiles.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(profile)
    profile.update(
        count=arr.shape[0],
        dtype=arr.dtype,
        height=arr.shape[1],
        width=arr.shape[2],
        transform=transform,
        compress="deflate",
        predictor=3 if np.issubdtype(arr.dtype, np.floating) else 2,
    )
    _set_tiling(profile, tiled, blockxsize, blockysize)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
    return path


def write_mask_cog(path, mask, profile, *, transform=None, blockxsize=None,
                   blockysize=None, tiled=True):
    """Write a 2-D binary mask as a single-band, uint8, LZW-compressed GTiff.

    Pass ``transform`` to re-georeference a cut tile; omit it (``None``) to keep
    the transform already present in ``profile`` (full-zone masks). ``tiled=False``
    writes a plain GTiff (raw V2 train mask tiles).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(profile)
    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        height=mask.shape[0],
        width=mask.shape[1],
        compress="lzw",
    )
    if transform is not None:
        profile["transform"] = transform
    _set_tiling(profile, tiled, blockxsize, blockysize)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)
    return path


def write_geoparquet(gdf, path):
    """Write a GeoDataFrame catalogue to a GeoParquet, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    return path


def write_split_csvs(gdf, splits_dir, columns):
    """Write one ``<split>.csv`` per ``split_set`` value with the given columns.

    Returns the list of written paths (empty if the catalogue is empty).
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    written = []
    if gdf.empty:
        return written
    for split_name, group in gdf.groupby("split_set"):
        out_path = splits_dir / f"{split_name}.csv"
        group[columns].to_csv(out_path, index=False)
        print(f"Wrote {len(group)} rows to {out_path}")
        written.append(out_path)
    return written
