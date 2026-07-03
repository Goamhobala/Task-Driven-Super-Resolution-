"""The dataset pipeline

A :class:`DatasetPipeline` is assembled from independent strategies:

    scan imagery -> ZoneProcessor (per zone) -> catalogue
                 -> Tagger(s) -> SplitStrategy -> write parquet + split CSVs

Presets:
  - :func:`make_v2rosa_pipeline` -- V2ROSA split-first (all splits -> raw 512px tiles,
    +enhanced-RGB bands, empty-tile subsampling): ``generate --variant V2ROSA``.
  - :func:`make_v1rosa_pipeline` -- V1ROSA (patch index): ``generate --variant V1ROSA``.

``DatasetPipeline`` (scan -> process -> tag -> split -> write) backs V1ROSA;
:class:`V2RosaPipeline` (split-first) backs V2ROSA.
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd
from sentinel2data.generator.config import (
    WGS84,
    DatasetPaths,
    TileSpec,
)
from sentinel2data.generator.helper import scan_rasters
from sentinel2data.generator.io import write_geoparquet, write_split_csvs
from sentinel2data.generator.labels import RasterMaskLabeler, RoadGraphLabeler
from sentinel2data.generator.splitting import RandomTileSplit, _holdout_map
from sentinel2data.generator.tagging import BiomeTagger, UrbanisationClassifier
from sentinel2data.generator.processor import V2ROSAProcessor


class ImageryScan:
    """Wrapper for scanning rasters with helper function"""

    def __init__(self, directory):
        self.directory = Path(directory)

    def scan(self):
        return scan_rasters(self.directory)


def build_catalogue(zone_gdfs, schema):
    """Concatenate per-zone GeoDataFrames into one catalogue (common CRS).

    Returns ``None`` if there are no rows. Reindexes ``tile_id`` globally when
    the schema asks for it (cut-tiles).
    """
    zone_gdfs = [g for g in zone_gdfs if g is not None and not g.empty]
    if not zone_gdfs:
        return None
    gdf = gpd.GeoDataFrame(
        pd.concat(zone_gdfs, ignore_index=True),
        geometry=schema.geometry_col,
        crs=WGS84,
    )
    if schema.reindex_tile_id:
        gdf["tile_id"] = range(len(gdf))
    return gdf


class DatasetPipeline:
    """Runs the scan -> process -> tag -> split -> write sequence."""

    def __init__(self, *, source, processor, taggers, split, paths):
        self.source = source
        self.processor = processor
        self.taggers = list(taggers)
        self.split = split
        self.paths = paths

    def run(self):
        images = self.source.scan()
        if not images:
            print(f"Aborting. No satellite imagery found in {self.source.directory}")
            return None
        print(f"Found {len(images)} source COG(s).")

        zone_gdfs = []
        for zone_id, sat_path in enumerate(images):
            print("-" * 50)
            zone_gdfs.append(self.processor.process(zone_id, sat_path, self.paths))

        schema = self.processor.schema
        gdf = build_catalogue(zone_gdfs, schema)
        if gdf is None:
            print("No road-bearing tiles produced; nothing written.")
            return None

        for tagger in self.taggers:
            gdf[tagger.column] = tagger.tag(gdf)
        self.split.assign(gdf)

        gdf = gdf[schema.columns]
        write_geoparquet(gdf, self.paths.metadata_path)
        write_split_csvs(gdf, self.paths.splits_dir, schema.split_csv_columns)

        print("-" * 50)
        print(f"Wrote {len(gdf)} rows to {self.paths.metadata_path}")
        return self.paths.metadata_path



class V2RosaPipeline:
    """Split-first runner for S2-ROSA-V2.

    Each zone's split is decided BEFORE processing (so its output dir is known),
    then :class:`V2ROSAProcessor` tiles it into that split. Rows from all zones are
    concatenated, ``image_id``-stamped, biome-tagged (optional), urbanisation-
    classified per split, and written to ``metadata.parquet`` + ``splits/<split>.csv``.
    """

    def __init__(self, *, source, processor, paths, val_frac=0.1, test_frac=0.1,
                 split_seed=42, biome_tagger=None):
        self.source = source
        self.processor = processor
        self.paths = paths
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.split_seed = split_seed
        self.biome_tagger = biome_tagger

    def run(self):
        images = self.source.scan()
        if not images:
            print(f"Aborting. No satellite imagery found in {self.source.directory}")
            return None
        print(f"Found {len(images)} source COG(s).")

        zones = sorted({p.stem for p in images})
        split_map = _holdout_map(zones, self.val_frac, self.test_frac, self.split_seed)
        counts = {s: list(split_map.values()).count(s) for s in ("train", "val", "test")}
        print(f"Zone split (seed={self.split_seed}): {counts} zones")

        rows = []
        for zone_id, sat in enumerate(images):
            print("-" * 50)
            rows.extend(self.processor.process(zone_id, sat, split_map[sat.stem], self.paths))

        if not rows:
            print("No images produced; nothing written.")
            return None

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


def make_v2rosa_pipeline(
    imagery_dir,
    output_dir,
    roads_parquet_path,
    biome_parquet_path=None,
    tile_size=512,
    patch_size=256,
    buffer_m=DEFAULT_BUFFER_M,
    val_frac=0.1,
    test_frac=0.1,
    split_seed=42,
    empty_keep_ratio=1.0,
    tile_seed=42,
    enhance_cfg=None,
):
    """S2-ROSA-V2 (split-first): every split cut into raw 512px tiles (+3 enhanced-RGB
    bands), empty tiles subsampled by ``empty_keep_ratio``; optional BiomeTagger."""
    processor = V2ROSAProcessor(
        roads_parquet_path,
        tile_spec=TileSpec(tile_size=tile_size, patch_size=patch_size),
        mask_labeler=RasterMaskLabeler(default_buffer_m=buffer_m),
        enhance_cfg=enhance_cfg,
        empty_keep_ratio=empty_keep_ratio,
        tile_seed=tile_seed,
    )
    biome_tagger = (
        BiomeTagger(biome_parquet_path) if biome_parquet_path is not None else None
    )
    if biome_tagger is None:
        print("No biome parquet given; biome left as 'Unknown'.")

    return V2RosaPipeline(
        source=ImageryScan(imagery_dir),
        processor=processor,
        paths=DatasetPaths(output_dir),
        val_frac=val_frac,
        test_frac=test_frac,
        split_seed=split_seed,
        biome_tagger=biome_tagger,
    )