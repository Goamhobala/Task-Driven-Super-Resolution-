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

def visualize_classification(sat_cog_path, out_plot_path, metadata_path):
    """Visualizes the metadata classifications over the satellite imagery."""
    print("Loading metadata and satellite imagery for visualization...")
    gdf = gpd.read_parquet(metadata_path)
    
    with rasterio.open(sat_cog_path) as src:
        img = src.read([1, 2, 3]).astype(np.float32)
        transform = src.transform

    print("Applying stretch for visualization...")
    stretched_img = np.zeros_like(img)
    for i in range(3):
        band = img[i]
        valid_pixels = band[band > 0]
        if len(valid_pixels) > 0:
            p2, p98 = np.percentile(valid_pixels, [2, 98])
            if p98 > p2:
                stretched = np.clip(band, p2, p98)
                stretched = (stretched - p2) / (p98 - p2)
                stretched_img[i] = stretched * 255
        else:
            stretched_img[i] = 0

    img_8bit = stretched_img.astype(np.uint8)

    print("Plotting map...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    show(img_8bit, transform=transform, ax=ax)

    gdf.plot(
        column='class',
        cmap='viridis',
        legend=True,
        alpha=0.4,
        edgecolor='white',
        linewidth=0.5,
        ax=ax
    )

    ax.set_title("Tile Classification over Sentinel-2 Imagery")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")

    plt.savefig(out_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {out_plot_path}")