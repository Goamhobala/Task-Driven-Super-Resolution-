"""Shared raster / vector geometry helpers.
  - window bounds calculation
  - percentile contrast stretch
  - imagery directory scanning 
"""
from pathlib import Path
import numpy as np
from rasterio.warp import transform_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from sentinel2data.generator.config import IMAGE_EXTS, RGBEnhanceConfig

# image directory scanning
def scan_rasters(directory, exts=IMAGE_EXTS, skip_mask_suffix="_mask.tif"):
    """Return sorted source rasters in the given directory.

    Skips macOS `._` dotfiles and previously-generated `*_mask.tif` - so a
    dataset dir can be re-scanned in place without re-ingesting its own masks.
    """
    directory = Path(directory)
    paths = []
    for ext in exts:
        paths.extend(directory.glob(f"*{ext}"))

    out = []
    for p in sorted(paths):
        if p.name.startswith("._"):
            continue
        if skip_mask_suffix and p.name.endswith(skip_mask_suffix):
            continue
        out.append(p)
    return out


# window helpers
def window_bounds(window, transform):
    """Map a rasterio ``Window`` to ``(minx, miny, maxx, maxy)`` in CRS units."""
    ptf = window_transform(window, transform)
    minx = ptf.c
    maxy = ptf.f
    maxx = minx + window.width * ptf.a
    miny = maxy + window.height * ptf.e
    return minx, miny, maxx, maxy


def window_box(window, transform):
    """Shapely ``box`` footprint of a window (see :func:`window_bounds`)."""
    return box(*window_bounds(window, transform))


def reproject_bounds(bounds, src_crs, dst_crs):
    """Reproject ``(minx, miny, maxx, maxy)`` -> ``(west, south, east, north)``."""
    minx, miny, maxx, maxy = bounds
    return transform_bounds(src_crs, dst_crs, minx, miny, maxx, maxy)


# visualisation helper
def stretch_to_uint8(band, percentile_range=(2, 98)):
    """Percentile contrast stretch of a single float band to 0-255.

    Pixels <= 0 are treated as nodata and excluded from the percentiles; an
    all-nodata or degenerate band returns zeros.
    """
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)

    lo, hi = np.percentile(valid, percentile_range)
    if hi <= lo:
        return np.zeros_like(band, dtype=np.uint8)

    stretched = np.clip(band, lo, hi)
    stretched = (stretched - lo) / (hi - lo)
    return (stretched * 255).astype(np.uint8)


def stretch_bands(img, percentile_range=(2, 98)):
    """Stretch a ``(3, H, W)`` float array to a ``(3, H, W)`` uint8 array."""
    return np.stack(
        [stretch_to_uint8(img[i], percentile_range) for i in range(img.shape[0])]
    )


# RGB enhancement (appended as extra imagery bands)
def enhance_rgb(rgb, cfg=None):
    """Per-channel CLAHE + gamma enhancement -> ``(3, H, W)`` float32 in [0,1].

    ``rgb`` is the imagery's R,G,B as a ``(3, H, W)`` float array. Each channel
    is min-max normalised to [0,1], gamma-corrected (brightening darks), then
    CLAHE-equalised (local contrast) via scikit-image. Non-finite pixels
    (NaN/inf nodata) are excluded from the min/max and mapped to 0; a flat or
    all-nodata channel maps to zeros. The result is meant to be *appended* to
    the imagery as 3 extra bands, keeping every original band.
    """
    from skimage.exposure import adjust_gamma, equalize_adapthist, rescale_intensity

    if cfg is None:
        cfg = RGBEnhanceConfig()

    out = []
    for i in range(rgb.shape[0]):
        ch = rgb[i].astype("float32")
        finite = np.isfinite(ch)
        if not finite.any():  # all NaN/inf -> nothing to enhance
            out.append(np.zeros_like(ch, dtype="float32"))
            continue
        lo = float(ch[finite].min())
        hi = float(ch[finite].max())
        if hi <= lo:  # flat channel -> nothing to enhance
            out.append(np.zeros_like(ch, dtype="float32"))
            continue
        # Nodata (NaN/inf) -> lo
        ch = np.where(finite, ch, lo)
        norm = rescale_intensity(ch, in_range=(lo, hi), out_range=(0.0, 1.0))
        gammad = adjust_gamma(norm, gamma=cfg.gamma)
        clahe = equalize_adapthist(
            gammad,
            kernel_size=cfg.clahe_kernel_size,
            clip_limit=cfg.clahe_clip_limit,
        )
        out.append(clahe.astype("float32"))
    return np.stack(out)