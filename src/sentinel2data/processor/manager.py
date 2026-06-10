from pathlib import Path
import pandas as pd
import geopandas as gpd
import shapely.geometry

from sentinel2data.processor.mask_generator import RoadMaskGenerator
from sentinel2data.processor.metadata_generator import MetadataGenerator

METADATA_COLUMNS = [
    # indexing
    "tile_id",
    "patch_row_id",
    "patch_col_id"
    "zone_name",
    # paths relative to the dataset dir
    "tile_path",
    "mask_raster_path",
    "mask_graph_path",

    "spatial_resolution", # default 10m currently, will add 20m bands later
    "urbanisation_classification",
    "road_density",              
    "split_set",

    "satellite_image_dates",     # TODO: multiple dates needed due to satellite imagery composite
    "crs", 
    "patch_bounding_geometry"
]

IMAGE_EXTS = (".tif", ".tiff")
COMMON_CRS = "EPSG:4326"


class DatasetManager:
    """Orchestrates S2-ROSA dataset generation.
    
    General Flow: 
        1. scan `imagery` directory for each satellite COG
        2. generate a matching road binary mask COG + road graph data
        3. build per-tile metadata
    """

    def __init__(self, dataset_dir, overture_parquet_path, buffer_m=10):
        self.dataset_dir = Path(dataset_dir)
        self.overture_parquet_path = Path(overture_parquet_path)
        self.buffer_m = buffer_m    # TODO: make the buffer dependent on the hierachy of road sizes

        self.imagery_dir = self.dataset_dir / "imagery"
        self.masks_raster_dir = self.dataset_dir / "masks_raster"
        self.masks_graph_dir = self.dataset_dir / "masks_graph"
        self.splits_dir = self.dataset_dir / "splits"
        self.metadata_path = self.dataset_dir / "metadata.parquet"

    def scan_imagery(self):
        """Return sorted satellite COG paths in imagery directory"""
        paths = []
        for ext in IMAGE_EXTS:
            paths.extend(self.imagery_dir.glob(f"*{ext}"))

        result = []
        for p in sorted(paths):
            if p.name.startswith("._") or p.name.endswith("_mask.tif"):
                continue
            result.append(p)
        return result

    # Testing purposes
    def create_test_geoparquet(self):
        """Write a test metadata.parquet with the defined column schema."""
        
        dummy_data = [{
            "tile_id": 1,
            "patch_id": 1,
            "zone_name": "Cape Town",
            "tile_path": "imagery/CapeTown.tif",
            "mask_raster_path": "masks_raster/CapeTown_mask.tif",
            "mask_graph_path": "masks_graph/CapeTown_graph.parquet",
            "spatial_resolution": 10.0,
            "urbanisation_classification": "urban",
            "road_density": 0.45,
            "split_set": "train",
            "satellite_image_dates": ["2023-01-01,2023-01-15"],
            "crs": "EPSG:32734",
            "patch_bounding_geometry": shapely.geometry.box(
                18.401382926206566, -34.09027262472663, 18.678261199345727, -33.85959951456962
            )
        }]

        df = pd.DataFrame(dummy_data, columns=METADATA_COLUMNS)

        gdf = gpd.GeoDataFrame(
            df, 
            geometry="patch_bounding_geometry", 
            crs=COMMON_CRS
        )

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(self.metadata_path)

        print(f"Wrote test metadata scaffold to {self.metadata_path}")
        return self.metadata_path


    def build_products(self):
        images = self.scan_imagery()
        if not images:
            print(f"Aborting. No satellite imagery found in {self.imagery_dir}")
            return

        print(f"Found {len(images)} satellite images.")

        self.masks_raster_dir.parent.mkdir(parents=True, exist_ok=True)
        self.masks_graph_dir.parent.mkdir(parents=True, exist_ok=True)
        self.splits_dir.parent.mkdir(parents=True, exist_ok=True)

        # TODO: move this inside the metadata generator. 
        # The API should be like you pass in the relavent information and when you are ready to get the whole list you call "get_root_metadata"
        metadata_list = [] 
        for image_index, sat_path in enumerate(images):
            mask_path = self.masks_raster_dir / f"{sat_path.stem}_mask.tif"
            graph_path = self.masks_graph_dir / f"{sat_path.stem}_graphs.parquet"

            print("-" * 50)
            print(f"[{image_index}] assigned to {sat_path.stem}")

            mask_gen = RoadMaskGenerator(
                sat_cog_path=sat_path,
                overture_parquet_path=self.overture_parquet_path,
                out_mask_path=mask_path,
                out_graph_path=graph_path,
                buffer_m=self.buffer_m,
            )
            mask_gen.generate_raster_mask()
            road_graph_path = mask_gen.generate_road_graph()

            meta_gen = MetadataGenerator(
                mask_cog_path=mask_path,
                image_index=image_index,
                image_name=sat_path.stem,
                image_path=sat_path.relative_to(self.dataset_dir),
                mask_raster_path=mask_path.relative_to(self.dataset_dir),
                road_graph_path=road_graph_path,
            )
            gdf = meta_gen.build_records()
            gdf = gdf.to_crs(COMMON_CRS)
            metadata_list.append(gdf)

        metadata_gdf = gpd.GeoDataFrame(
            pd.concat(metadata_list, ignore_index=True), geometry="patch_bounding_geometry", crs=COMMON_CRS
        )[METADATA_COLUMNS]

        self._write_geoparquet(metadata_gdf)
        print("-" * 50)
        print(f"Wrote {len(metadata_gdf)} tiles to {self.metadata_path}")

        return self.metadata_path

    def _write_geoparquet(self, gdf):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(self.metadata_path)