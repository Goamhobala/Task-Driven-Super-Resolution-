from pathlib import Path
import pandas as pd
import geopandas as gpd
import shapely.geometry

from sentinel2data.processor.mask_generator import RoadMaskGenerator
from sentinel2data.processor.metadata_generator import MetadataGenerator, METADATA_COLUMNS

IMAGE_EXTS = (".tif", ".tiff")
COMMON_CRS = "EPSG:4326"


class DatasetManager:
    """Orchestrates S2-ROSA dataset generation.
    
    General Flow: 
        1. scan `imagery` directory for each satellite COG
        2. generate a matching road binary mask COG + road graph data
        3. build per-tile metadata
    """

    def __init__(self, dataset_dir, roads_parquet_path, buffer_m=5,
                 val_frac=0.1, test_frac=0.1, split_seed=42):
        self.dataset_dir = Path(dataset_dir)
        self.roads_parquet_path = Path(roads_parquet_path)

        self.buffer_m = buffer_m # Fallback

        self.val_frac = val_frac
        self.test_frac = test_frac
        self.split_seed = split_seed

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
            "patch_row_id": 0,
            "patch_col_id": 0,
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

        self.masks_raster_dir.mkdir(parents=True, exist_ok=True)
        self.masks_graph_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # MetadataGenerator accumulates every tile's patches; the combined
        # catalogue is retrieved once at the end via get_root_metadata().
        meta_gen = MetadataGenerator(dataset_dir=self.dataset_dir, common_crs=COMMON_CRS)
        for tile_id, sat_path in enumerate(images):
            mask_path = self.masks_raster_dir / f"{sat_path.stem}_mask.tif"
            graph_path = self.masks_graph_dir / f"{sat_path.stem}_graphs.parquet"

            print("-" * 50)
            print(f"[{tile_id}] assigned to {sat_path.stem}")

            mask_gen = RoadMaskGenerator(
                sat_cog_path=sat_path,
                roads_parquet_path=self.roads_parquet_path,
                out_mask_path=mask_path,
                out_graph_path=graph_path,
                default_buffer_m=self.buffer_m,
            )
            mask_gen.generate_raster_mask()
            road_graph_path = mask_gen.generate_road_graph()

            meta_gen.add_tile(
                tile_id=tile_id,
                mask_cog_path=mask_path,
                sat_cog_path=sat_path,
                mask_graph_path=road_graph_path,
            )

        meta_gen.assign_random_split(
            val_frac=self.val_frac, test_frac=self.test_frac, seed=self.split_seed
        )
        metadata_gdf = meta_gen.get_root_metadata()
        self._write_geoparquet(metadata_gdf)
        meta_gen.write_splits(self.splits_dir)
        print("-" * 50)
        print(f"Wrote {len(metadata_gdf)} patches to {self.metadata_path}")

        return self.metadata_path

    def _write_geoparquet(self, gdf):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(self.metadata_path)