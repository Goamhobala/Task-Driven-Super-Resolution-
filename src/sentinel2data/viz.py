from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from matplotlib import pyplot as plt
import geopandas as gpd
from rasterio.plot import show

def tiff_to_png_cumulative_stretch(input_tif, output_png, bands=[1, 2, 3], percentile_range=(2, 98)):
    with rasterio.open(input_tif) as src:
        tiff_image = src.read(bands).astype(np.float32)
        png_image = np.zeros_like(tiff_image)

        for i in range(3):
            band = tiff_image[i]
            # TODO: check if [band>0] is nessary
            lower_pctl, upper_pctl = np.percentile(band[band > 0], [percentile_range[0], percentile_range[1]])

            # clip and stretch to png range (0-255)
            stretched = np.clip(band, lower_pctl, upper_pctl)
            stretched = (stretched - lower_pctl) / (upper_pctl - lower_pctl)
            png_image[i] = stretched * 255

        img_8bit = png_image.astype(np.uint8)

        final_img = np.transpose(img_8bit, (1, 2, 0))
        Image.fromarray(final_img).save(output_png)

def _stretch_rgb(img, percentile_range=(2, 98)):
    """Percentile contrast-stretch a (3, H, W) float array to uint8."""
    out = np.zeros_like(img)
    for i in range(3):
        band = img[i]
        valid = band[band > 0]
        if valid.size:
            lo, hi = np.percentile(valid, list(percentile_range))
            if hi > lo:
                out[i] = np.clip((np.clip(band, lo, hi) - lo) / (hi - lo) * 255, 0, 255)
    return out.astype(np.uint8)


def visualize_classification(
    metadata_path,
    out_plot_path,
    zone_name=None,
    tile_id=None,
    dataset_dir=None,
):
    """Plot one tile's patches coloured by urbanisation classification.

    Selects the patches for a single tile by ``zone_name`` or ``tile_id``
    (one-to-one) and plots their bounding polygons coloured by class. When
    ``dataset_dir`` is given, the tile's satellite image is drawn underneath
    and the patches are reprojected onto it.
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Optional satellite backdrop: reproject patches to the image's native CRS.
    if dataset_dir is not None:
        sat_path = Path(dataset_dir) / patches["tile_path"].iloc[0]
        print(f"Drawing satellite backdrop from {sat_path}...")
        with rasterio.open(sat_path) as src:
            img = src.read([1, 2, 3]).astype(np.float32)
            transform = src.transform
            sat_crs = src.crs
        show(_stretch_rgb(img), transform=transform, ax=ax)
        patches = patches.to_crs(sat_crs)
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

    ax.set_title(f"Patch classification over {zone}")
    plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {out_plot_path}")