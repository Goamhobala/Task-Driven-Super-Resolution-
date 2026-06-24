from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from matplotlib import pyplot as plt
import geopandas as gpd
from rasterio.plot import show

from sentinel2data.generator.helper import stretch_bands


def tiff_to_png_cumulative_stretch(input_tif, output_png, bands=[1, 2, 3], percentile_range=(2, 98)):
    with rasterio.open(input_tif) as src:
        tiff_image = src.read(bands).astype(np.float32)
    img_8bit = stretch_bands(tiff_image, percentile_range)
    final_img = np.transpose(img_8bit, (1, 2, 0))
    Image.fromarray(final_img).save(output_png)


def visualize_classification(
    metadata_path,
    out_plot_path,
    zone_name=None,
    tile_id=None,
    dataset_dir=None,
    backdrop="satellite",
):
    """Plot one tile's patches coloured by urbanisation classification.
    """
    print("Loading metadata for visualization...")
    gdf = gpd.read_parquet(metadata_path)

    if zone_name is not None:
        patches = gdf[gdf["zone_name"] == zone_name]
    elif tile_id is not None:
        patches = gdf[gdf["tile_id"] == tile_id]
    else:
        raise ValueError("Provide either zone_name or tile_id.")

    if patches.empty:
        raise ValueError(f"No patches found for zone_name={zone_name!r} / tile_id={tile_id!r}.")

    zone = patches["zone_name"].iloc[0]
    backdrop = (backdrop or "none").lower()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    if backdrop in ("satellite", "mask"):
        if dataset_dir is None:
            raise ValueError(f"dataset_dir is required for backdrop={backdrop!r}.")

        if backdrop == "satellite":
            raster_path = Path(dataset_dir) / patches["tile_path"].iloc[0]
            print(f"Drawing satellite backdrop from {raster_path}...")
            with rasterio.open(raster_path) as src:
                data = stretch_bands(src.read([1, 2, 3]).astype(np.float32))
                transform = src.transform
                raster_crs = src.crs
            show(data, transform=transform, ax=ax)
        else:  # mask
            raster_path = Path(dataset_dir) / patches["mask_raster_path"].iloc[0]
            print(f"Drawing binary-mask backdrop from {raster_path}...")
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                transform = src.transform
                raster_crs = src.crs
            show(data, transform=transform, ax=ax, cmap="gray")

        patches = patches.to_crs(raster_crs)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    print("Plotting patch classifications...")
    patches.plot(
        column="urbanisation_classification",
        categorical=True,
        cmap="viridis",
        legend=True,
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_title(f"Patch classification over {zone} ({backdrop})")
    Path(out_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {out_plot_path}")


# --------------------------------------------------------------------------- #
# S2-ROSA-V2 triptych: RGB | RGB enhanced | road mask
# --------------------------------------------------------------------------- #
RGB_BAND_IDX = (1, 2, 3)  # source B4, B3, B2 (1-based)


def _enhanced_band_idx(count):
    """1-based indices of the 3 appended enhanced-RGB bands (the last three)."""
    return [count - 2, count - 1, count]


def _render_v2_triptych(dataset_dir, row, out_dir, percentile_range):
    """One image -> a 3-panel PNG under ``out_dir/<split>/<stem>.png``."""
    dataset_dir = Path(dataset_dir)
    img_path = dataset_dir / row["image_path"]
    mask_path = dataset_dir / row["mask_path"]
    split = row["split_set"]
    stem = Path(row["image_path"]).stem

    with rasterio.open(img_path) as src:
        rgb = stretch_bands(src.read(list(RGB_BAND_IDX)).astype(np.float32), percentile_range)
        enhanced = src.read(_enhanced_band_idx(src.count)).astype(np.float32)  # already [0,1]

    with rasterio.open(mask_path) as msk:
        mask = msk.read(1)

    rgb_img = np.transpose(rgb, (1, 2, 0))
    enh_img = np.transpose(np.clip(enhanced * 255.0, 0, 255).astype(np.uint8), (1, 2, 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB")
    axes[1].imshow(enh_img)
    axes[1].set_title("RGB enhanced (CLAHE + gamma)")
    axes[2].imshow(mask > 0, cmap="gray")
    axes[2].set_title("Road mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{stem}  [{split}]")

    out_png = Path(out_dir) / split / f"{stem}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def visualize_rosav2(dataset_dir, out_dir=None, limit=None, percentile_range=(0, 100)):
    """Render a side-by-side RGB | RGB-enhanced | road-mask PNG for every image in
    a ROSAV2 dataset.

    Reads ``<dataset_dir>/metadata.parquet`` and writes PNGs under
    ``out_dir`` (default ``<dataset_dir>/visualisation``), mirroring the
    train/val/test split folders. ``limit`` caps the images rendered per split.
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir) if out_dir is not None else dataset_dir / "visualisation"
    print(f"Loading metadata from {dataset_dir / 'metadata.parquet'}...")
    gdf = gpd.read_parquet(dataset_dir / "metadata.parquet")

    n = 0
    for split, group in gdf.groupby("split_set"):
        if limit is not None:
            group = group.head(limit)
        print(f"Rendering {len(group)} {split} image(s)...")
        for _, row in group.iterrows():
            _render_v2_triptych(dataset_dir, row, out_dir, percentile_range)
            n += 1
    print(f"Wrote {n} visualisations to {out_dir}")
    return out_dir