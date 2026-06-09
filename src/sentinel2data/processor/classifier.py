import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import transform as window_transform
from shapely.geometry import box
import jenkspy

# This should really rather be a parquet metadata generator. 
class TileClassifier:
    """Reads a road mask COG, classifies internal tiles, and generates metadata."""
    
    def __init__(self, mask_cog_path, out_metadata_path):
        self.mask_cog_path = mask_cog_path
        self.out_metadata_path = out_metadata_path

    def classify(self):
        print("Scanning mask COG internal blocks...")
        records = []

        with rasterio.open(self.mask_cog_path) as src:
            crs = src.crs
            # Iterate through the internal block windows natively
            for ji, window in src.block_windows(1):
                mask_patch = src.read(1, window=window)
                
                # Skip edge tiles if they aren't perfectly square
                if mask_patch.shape[0] != window.height or mask_patch.shape[1] != window.width:
                    print("Warning: Skipping non-square tile. Normally the edge tiles")
                    continue

                # Calculate road density purely from pixels (faster than vector math)
                road_pixels = np.sum(mask_patch > 0)
                
                # Get geospatial bounds of this specific tile
                patch_transform = window_transform(window, src.transform)
                minx = patch_transform.c
                maxy = patch_transform.f
                maxx = minx + (window.width * patch_transform.a)
                miny = maxy + (window.height * patch_transform.e)

                records.append({
                    'window_row': ji[0],
                    'window_col': ji[1],
                    'minx': minx,
                    'miny': miny,
                    'maxx': maxx,
                    'maxy': maxy,
                    'road_pixels': road_pixels,
                    'geometry': box(minx, miny, maxx, maxy)
                })

        gdf = gpd.GeoDataFrame(records, crs=crs)
        
        print("Applying Jenks Natural Breaks classification...")
        labels = ['Rural', 'Peri-Urban', 'Urban']
        gdf['class'] = 'Empty'

        has_roads = gdf['road_pixels'] > 0
        road_values = gdf.loc[has_roads, 'road_pixels']

        if road_values.nunique() >= 3:
            breaks = jenkspy.jenks_breaks(road_values, n_classes=3)
            gdf.loc[has_roads, 'class'] = pd.cut(
                road_values, bins=breaks, labels=labels, include_lowest=True
            ).astype(str)
            print(f"Thresholds (Pixels) -> Rural: < {breaks[1]:.0f} | Peri-Urban: < {breaks[2]:.0f} | Urban: > {breaks[2]:.0f}")
        elif not road_values.empty:
            gdf.loc[has_roads, 'class'] = 'Rural'
        
        # Save to Parquet
        os.makedirs(os.path.dirname(self.out_metadata_path), exist_ok=True)
        gdf.to_parquet(self.out_metadata_path)
        print(f"Metadata saved to {self.out_metadata_path} with {len(gdf)} tiles.")