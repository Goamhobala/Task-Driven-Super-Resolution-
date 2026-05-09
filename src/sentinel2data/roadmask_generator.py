import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import box
import matplotlib.pyplot as plt
import jenkspy

def main():
    # 1. Define your paths
    raster_file = "/Volumes/FILES/SouthAfricanS2/CapeTown.tif"
    parquet_file = "/Volumes/FILES/RoadVectorData/OvertureSARoadData/south_africa_overture_roads.parquet"
    output_folder = "/Volumes/FILES/PrototypeDataset/CapeTown"

    # 2. Initialize the Builder
    builder = InstaRoadDatasetBuilder(
        raster_path=raster_file,
        vector_path=parquet_file,
        base_out_dir=output_folder
    )

    # 3. Run the pipeline steps
    builder.prepare_data()
    builder.generate_mask(buffer_m=10)
    builder.classify_grid()
    # builder.plot_classification()
    builder.plot_classification_on_image()
    # builder.extract_patches()

class InstaRoadDatasetBuilder:
    """
    A pipeline to process Sentinel-2 imagery and Overture maps into
    classified 256x256 PyTorch-ready patches.
    """

    def __init__(self, raster_path, vector_path, base_out_dir, patch_size_px=256, resolution_m=10):
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.base_out_dir = base_out_dir

        # Physical math
        self.patch_size_px = patch_size_px
        self.resolution_m = resolution_m
        self.block_size_m = patch_size_px * resolution_m

        # State variables populated during execution
        self.raster_meta = None
        self.raster_crs = None
        self.raster_bounds = None
        self.footprint = None
        self.local_roads = None
        self.grid = None
        self.mask_path = None

    def prepare_data(self):
        """Loads raster metadata, aligns CRSs, and clips vectors to the raster footprint."""
        print(f"[{os.path.basename(self.raster_path)}] Extracting metadata...")

        with rasterio.open(self.raster_path) as src:
            self.raster_meta = src.meta.copy()
            self.raster_crs = src.crs
            self.raster_bounds = src.bounds
            self.footprint = box(*self.raster_bounds)

        print("Loading and clipping Overture Parquet data...")
        roads = gpd.read_parquet(self.vector_path)
        roads = roads.to_crs(self.raster_crs)
        self.local_roads = gpd.clip(roads, self.footprint)

        print(f"Found {len(self.local_roads)} road segments in this image footprint.")

    def generate_mask(self, buffer_m=10, mask_filename="temp_mask.tif"):
        """Buffers roads and rasterizes them into a binary mask."""
        print(f"Rasterizing road mask (Buffer: {buffer_m}m)...")
        self.mask_path = os.path.join(self.base_out_dir, mask_filename)
        os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)

        if self.local_roads.empty:
            print("Warning: No roads found. Creating an empty mask.")
            mask = np.zeros((self.raster_meta['height'], self.raster_meta['width']), dtype='uint8')
        else:
            buffered_roads = self.local_roads.geometry.buffer(buffer_m)
            shapes = ((geom, 1) for geom in buffered_roads)
            mask = features.rasterize(
                shapes=shapes,
                out_shape=(self.raster_meta['height'], self.raster_meta['width']),
                transform=self.raster_meta['transform'],
                fill=0,
                all_touched=True,
                dtype='uint8'
            )

        mask_meta = self.raster_meta.copy()
        mask_meta.update(dtype='uint8', count=1, nodata=0)

        with rasterio.open(self.mask_path, "w", **mask_meta) as dest:
            dest.write(mask, 1)

        print("Local road mask generated successfully.")

    def classify_grid(self):
        """Creates physical grid, calculates road density, and applies Jenks Natural Breaks."""
        print(f"Generating {self.block_size_m}m x {self.block_size_m}m patch grid...")
        minx, miny, maxx, maxy = self.raster_bounds

        grid_cells = [
            box(x0, y0, x0 + self.block_size_m, y0 + self.block_size_m)
            for x0 in np.arange(minx, maxx, self.block_size_m)
            for y0 in np.arange(miny, maxy, self.block_size_m)
        ]

        self.grid = gpd.GeoDataFrame(geometry=grid_cells, crs=self.raster_crs)
        self.grid['patch_id'] = range(len(self.grid))

        print("Calculating road density per patch...")
        roads_in_grid = gpd.overlay(self.local_roads, self.grid, how='intersection')
        roads_in_grid['road_length'] = roads_in_grid.geometry.length
        lengths_per_patch = roads_in_grid.groupby('patch_id')['road_length'].sum()

        self.grid['total_road_length'] = self.grid['patch_id'].map(lengths_per_patch).fillna(0)

        # Classification Logic
        labels = ['Rural', 'Peri-Urban', 'Urban']
        self.grid['class'] = 'Empty'

        has_roads_mask = self.grid['total_road_length'] > 0
        road_values = self.grid.loc[has_roads_mask, 'total_road_length']

        if road_values.nunique() >= 3:
            breaks = jenkspy.jenks_breaks(road_values, n_classes=3)
            self.grid.loc[has_roads_mask, 'class'] = pd.cut(
                road_values, bins=breaks, labels=labels, include_lowest=True
            ).astype(str)
            print(f"Thresholds -> Rural: < {breaks[1]:.0f}m | Peri-Urban: < {breaks[2]:.0f}m | Urban: > {breaks[2]:.0f}m")
        elif not road_values.empty:
            self.grid.loc[has_roads_mask, 'class'] = 'Rural'
            print("Minimal road variance found. Labeling non-zero patches as Rural.")
        else:
            print("No roads found. Entire tile is labeled Empty.")

    def plot_classification(self, out_name="classification.png"):
        """Saves a plot of the classified patches and overlaid roads."""
        if self.grid is None:
            raise ValueError("Grid not generated. Run classify_grid() first.")

        print("Saving Classification Map plot...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        self.grid.plot(column='class', cmap='viridis', legend=True, alpha=0.5, edgecolor='black', ax=ax)
        self.local_roads.plot(ax=ax, color='black', linewidth=0.5)

        ax.set_title("Overture Road Density Classification")
        plt.savefig(os.path.join(self.base_out_dir, out_name))
        plt.close(fig)

    def extract_patches(self):
        """Slices the image and mask into 256x256 arrays and sorts into class folders."""
        if self.grid is None or self.mask_path is None:
            raise ValueError("Run generate_mask() and classify_grid() before extracting.")

        print("Setting up dataset directories...")
        classes = ['Empty', 'Rural', 'Peri-Urban', 'Urban']
        for cls in classes:
            os.makedirs(os.path.join(self.base_out_dir, "images", cls), exist_ok=True)
            os.makedirs(os.path.join(self.base_out_dir, "masks", cls), exist_ok=True)

        print("Slicing rasters into patches...")
        valid_patch_count = 0
        tile_name = os.path.splitext(os.path.basename(self.raster_path))[0]

        with rasterio.open(self.raster_path) as src_img, rasterio.open(self.mask_path) as src_mask:
            img_profile = src_img.profile.copy()
            mask_profile = src_mask.profile.copy()

            for idx, row in self.grid.iterrows():
                minx, miny, maxx, maxy = row.geometry.bounds
                window = from_bounds(minx, miny, maxx, maxy, src_img.transform)

                img_patch = src_img.read(window=window)
                mask_patch = src_mask.read(window=window)

                # Skip edges to ensure perfect 256x256 squares
                if img_patch.shape[1:] != (self.patch_size_px, self.patch_size_px):
                    continue

                valid_patch_count += 1
                patch_transform = rasterio.windows.transform(window, src_img.transform)

                # Update profiles
                for profile in [img_profile, mask_profile]:
                    profile.update({'height': self.patch_size_px, 'width': self.patch_size_px, 'transform': patch_transform})

                # Unique filenames to prevent overwrites when running multiple tiles
                filename = f"{tile_name}_patch_{row['patch_id']:04d}.tif"

                # Save Image
                img_out_path = os.path.join(self.base_out_dir, "images", row['class'], filename)
                with rasterio.open(img_out_path, 'w', **img_profile) as dest_img:
                    dest_img.write(img_patch)

                # Save Mask
                mask_out_path = os.path.join(self.base_out_dir, "masks", row['class'], filename)
                with rasterio.open(mask_out_path, 'w', **mask_profile) as dest_mask:
                    dest_mask.write(mask_patch)

        print(f"Successfully exported {valid_patch_count} patches for {tile_name}!")

        # Optional Cleanup: Remove the large temporary mask if you don't need it
        # os.remove(self.mask_path)


    def plot_classification_on_image(self, out_name="classification_on_image.png"):
        """Saves a plot of the classified patches overlaid on the stretched satellite image."""
        if self.grid is None:
            raise ValueError("Grid not generated. Run classify_grid() first.")

        from rasterio.plot import show

        print("Generating stretched satellite background for plotting...")
        with rasterio.open(self.raster_path) as src:
            # Read RGB bands (1, 2, 3)
            img = src.read([1, 2, 3]).astype(np.float32)
            transform = src.transform

        # Apply 2% - 98% cumulative stretch for visualization
        stretched_img = np.zeros_like(img)
        for i in range(3):
            band = img[i]
            # Avoid calculating percentiles on pure 0 background padding
            valid_pixels = band[band > 0]

            if len(valid_pixels) > 0:
                p2, p98 = np.percentile(valid_pixels, [2, 98])
                # Prevent division by zero if an entire band is uniform
                if p98 > p2:
                    stretched = np.clip(band, p2, p98)
                    stretched = (stretched - p2) / (p98 - p2)
                    stretched_img[i] = stretched * 255
            else:
                stretched_img[i] = 0

        # Convert to 8-bit unsigned integer
        img_8bit = stretched_img.astype(np.uint8)

        print("Saving Classification on Image plot...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # 1. Plot the stretched satellite image background
        show(img_8bit, transform=transform, ax=ax)

        # 2. Overlay the classified grid
        # alpha=0.4 makes the grid semi-transparent so you can see the city below
        self.grid.plot(
            column='class',
            cmap='viridis',
            legend=True,
            alpha=0.4,
            edgecolor='white',
            linewidth=0.5,
            ax=ax
        )

        ax.set_title("Overture Road Classification over Sentinel-2 Imagery")
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        # Save with a high DPI so zooming in looks crisp
        plt.savefig(os.path.join(self.base_out_dir, out_name), dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()