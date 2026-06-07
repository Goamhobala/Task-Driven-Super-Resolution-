import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box

class RoadMaskGenerator:
    """Creates a binary road mask COG matching the satellite imagery's internal tiling."""

    def __init__(self, sat_cog_path, vector_parquet_path, out_mask_path, out_graph_path=None, buffer_m=10):
        self.sat_cog_path = sat_cog_path
        self.vector_parquet_path = vector_parquet_path
        self.out_mask_path = out_mask_path
        self.out_graph_path = out_graph_path
        self.buffer_m = buffer_m

    def generate(self):
        print(f"Reading metadata from: {os.path.basename(self.sat_cog_path)}")
        with rasterio.open(self.sat_cog_path) as src:
            sat_meta = src.meta.copy()
            sat_crs = src.crs
            sat_bounds = src.bounds

            # internal tiling
            blockxsize = src.profile.get('blockxsize', 256)
            blockysize = src.profile.get('blockysize', 256)

        footprint = box(*sat_bounds)
        footprint_gdf = gpd.GeoDataFrame({'geometry': [footprint]}, crs=sat_crs)
        bbox_4326 = footprint_gdf.to_crs("EPSG:4326").total_bounds

        print("Filtering Overture Parquet data...")
        roads = gpd.read_parquet(self.vector_parquet_path, bbox=tuple(bbox_4326))

        if roads.empty:
            print("Warning: No roads found. Creating an empty mask.")
            mask = np.zeros((sat_meta['height'], sat_meta['width']), dtype='uint8')
            local_roads = gpd.GeoDataFrame({'geometry': []}, crs=sat_crs)
        else:
            print(f"Reprojecting and clipping {len(roads)} road segments...")
            roads = roads.to_crs(sat_crs)
            local_roads = gpd.clip(roads, footprint)

            print(f"Rasterizing roads with a {self.buffer_m}m buffer...")
            buffered_roads = local_roads.geometry.buffer(self.buffer_m)
            shapes = ((geom, 1) for geom in buffered_roads)
            mask = features.rasterize(
                shapes=shapes,
                out_shape=(sat_meta['height'], sat_meta['width']),
                transform=sat_meta['transform'],
                fill=0,
                all_touched=True,
                dtype='uint8'
            )

        # Update metadata for an internally tiled, compressed mask
        sat_meta.update(
            dtype='uint8', 
            count=1, 
            nodata=0,
            compress='lzw',
            tiled=True,
            blockxsize=blockxsize,
            blockysize=blockysize
        )

        os.makedirs(os.path.dirname(self.out_mask_path), exist_ok=True)
        print(f"Writing tiled road mask COG to {self.out_mask_path}...")
        with rasterio.open(self.out_mask_path, "w", **sat_meta) as dest:
            dest.write(mask, 1)

        # Write the clipped road vector (line geometry, sat CRS) as a graph parquet
        if self.out_graph_path:
            os.makedirs(os.path.dirname(self.out_graph_path), exist_ok=True)
            print(f"Writing road vector parquet to {self.out_graph_path}...")
            local_roads.to_parquet(self.out_graph_path)

        print("Mask generation complete.")