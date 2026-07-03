"""The dataset pipeline"""
from pathlib import Path
import geopandas as gpd
from sentinel2data.generator.config import (
    WGS84,
    DatasetPaths,
    TileSpec,
)
from sentinel2data.generator.helper import scan_rasters
from sentinel2data.generator.io import write_geoparquet, write_split_csvs
from sentinel2data.generator.labels import RasterMaskLabeler
from sentinel2data.generator.splitting import splitset_map
from sentinel2data.generator.tagging import BiomeTagger, UrbanisationClassifier
from sentinel2data.generator.processor import ROSAProcessor

class ImageryScan:
    """Wrapper for scanning rasters with helper function"""

    def __init__(self, directory):
        self.directory = Path(directory)

    def scan(self):
        return scan_rasters(self.directory)


class RosaPipeline:
    """Pipeline for ROSA Dataset Creation

    General Flow:
        1. Scan imagery directory for COGs (one per zone).
        2. Split zones into train/val/test (holdout).
        3. For each zone, process it with V2ROSAProcessor to create tiles and metadata
        4. Tag each tile with biome and urbanisation class
        5. Write the final GeoDataFrame to metadata.parquet and split CSVs
    """

    def __init__(self, *, source, processor, paths, biome_tagger: BiomeTagger, val_frac=0.1, test_frac=0.1,
                 split_seed=42):
        self.source = source
        self.processor = processor
        self.paths = paths
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.split_seed = split_seed
        self.biome_tagger = biome_tagger

    def run(self):
        # Scan imagery
        images = self.source.scan()
        if not images:
            print(f"Aborting. No satellite imagery found in {self.source.directory}")
            return None
        print(f"Found {len(images)} source COG(s).")

        # Split zones into train/val/test
        zones = sorted({p.stem for p in images})
        split_map = splitset_map(zones, self.val_frac, self.test_frac, self.split_seed)
        counts = {s: list(split_map.values()).count(s) for s in ("train", "val", "test")}
        print(f"Zone split (seed={self.split_seed}): {counts} zones")

        # Process each zone and collect rows of metadata
        rows = []
        for zone_id, sat in enumerate(images):
            print("-" * 50)
            rows.extend(self.processor.process(zone_id, sat, split_map[sat.stem], self.paths))

        if not rows:
            print("Warning: No images produced; nothing written.")
            return None

        # Tagging: Biome and Urbanisation classification
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)
        gdf["image_id"] = range(len(gdf))
        if self.biome_tagger is not None:
            gdf[self.biome_tagger.column] = self.biome_tagger.tag(gdf)

        # Urbanisation class per split (Jenks breaks over each split's densities).
        urban = UrbanisationClassifier()
        gdf[urban.column] = urban.tag(gdf)

        schema = self.processor.schema
        gdf = gdf[schema.columns]
        write_geoparquet(gdf, self.paths.metadata_path)
        write_split_csvs(gdf, self.paths.splits_dir, schema.split_csv_columns)

        print("-" * 50)
        print(f"Wrote {len(gdf)} images to {self.paths.metadata_path}")
        return self.paths.metadata_path


def make_rosa_pipeline(
    imagery_dir,
    output_dir,
    roads_parquet_path,
    biome_parquet_path=None,
    tile_size=512,
    patch_size=256,
    val_frac=0.1,
    test_frac=0.1,
    split_seed=42,
    empty_keep_ratio=1.0,
    tile_seed=42,
    enhance_cfg=None,
):
    processor = ROSAProcessor(
        roads_parquet_path,
        tile_spec=TileSpec(tile_size=tile_size, patch_size=patch_size),
        mask_labeler=RasterMaskLabeler(),
        enhance_cfg=enhance_cfg,
        empty_keep_ratio=empty_keep_ratio,
        tile_seed=tile_seed,
    )
    biome_tagger = (
        BiomeTagger(biome_parquet_path) if biome_parquet_path is not None else None
    )
    if biome_tagger is None:
        raise ValueError("No biome parquet given")

    return RosaPipeline(
        source=ImageryScan(imagery_dir),
        processor=processor,
        paths=DatasetPaths(output_dir),
        val_frac=val_frac,
        test_frac=test_frac,
        split_seed=split_seed,
        biome_tagger=biome_tagger,
    )