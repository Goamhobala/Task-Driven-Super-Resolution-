import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import transform as window_transform
from shapely.geometry import box
import jenkspy
from pathlib import Path

CLASS_LABELS = ["Rural", "Peri-Urban", "Urban"]
EMPTY_LABEL = "Empty"

# Scaffolded values
SCAFFOLD_DATES = ["2023-01-01"]   # TODO: real composite dates per tile
SCAFFOLD_SPLIT = "train"          # TODO: real train/val/test assignment
SCAFFOLD_BIOME = "Unknown"       # TODO: real biome/zone names per tile

METADATA_COLUMNS = [
    # indexing
    "tile_id",
    "patch_row_id",
    "patch_col_id",
    "zone_name",
    # paths relative to the dataset dir
    "tile_path",
    "mask_raster_path",
    "mask_graph_path",
    # classification
    "spatial_resolution",
    "urbanisation_classification",
    "biome",
    "road_density",
    "split_set",
    # additional metadata
    "satellite_image_dates",
    "crs",
    "patch_bounding_geometry",
]

# Columns written to each splits/<split>.csv (indexing + paths only).
SPLIT_CSV_COLUMNS = [
    "tile_id",
    "patch_row_id",
    "patch_col_id",
    "zone_name",
    "tile_path",
    "mask_raster_path",
    "mask_graph_path",
]


class MetadataGenerator:
    """Accumulates per-patch metadata across tiles into one root catalogue.
    """

    def __init__(self, dataset_dir, common_crs="EPSG:4326"):
        self.dataset_dir = Path(dataset_dir)
        self.common_crs = common_crs
        self._tiles = []  # list of per-tile GeoDataFrames

    def add_tile(self, tile_id, mask_cog_path, sat_cog_path, mask_graph_path):
        """Scan one tile's patches and stage them into the catalogue."""
        zone_name = Path(sat_cog_path).stem
        paths = {
            "tile_path": self._rel(sat_cog_path),
            "mask_raster_path": self._rel(mask_cog_path),
            "mask_graph_path": self._rel(mask_graph_path),
        }

        records, native_crs = self._scan_patches(tile_id, zone_name, paths, mask_cog_path)
        self._classify(records)

        tile_gdf = gpd.GeoDataFrame(
            records, geometry="patch_bounding_geometry", crs=native_crs
        ).to_crs(self.common_crs)
        self._tiles.append(tile_gdf)
        return tile_gdf

    def get_root_metadata(self):
        """Return the combined per-patch catalogue (common CRS, ordered cols)."""
        if not self._tiles:
            empty = {
                c: pd.Series(dtype="object")
                for c in METADATA_COLUMNS
                if c != "patch_bounding_geometry"
            }
            return gpd.GeoDataFrame(
                empty,
                geometry=gpd.GeoSeries([], dtype="geometry"),
                crs=self.common_crs,
            )[METADATA_COLUMNS]

        return gpd.GeoDataFrame(
            pd.concat(self._tiles, ignore_index=True),
            geometry="patch_bounding_geometry",
            crs=self.common_crs,
        )[METADATA_COLUMNS]

    def write_splits(self, splits_dir):
        """Write one ``splits/<split>.csv`` per split_set value.
        """
        gdf = self.get_root_metadata()
        splits_dir = Path(splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)

        written = []
        if gdf.empty:
            return written
        for split_name, group in gdf.groupby("split_set"):
            out_path = splits_dir / f"{split_name}.csv"
            group[SPLIT_CSV_COLUMNS].to_csv(out_path, index=False)
            print(f"Wrote {len(group)} patches to {out_path}")
            written.append(out_path)
        return written

    def _scan_patches(self, tile_id, zone_name, paths, mask_cog_path):
        print(f"Scanning mask blocks: {zone_name}")
        records = []
        with rasterio.open(mask_cog_path) as src:
            native_crs = src.crs
            crs_str = src.crs.to_string()  # native satellite imagery CRS
            spatial_resolution = abs(src.transform.a)
            for ji, window in src.block_windows(1):
                patch = src.read(1, window=window)

                road_pixels = int(np.count_nonzero(patch > 0))
                total_pixels = int(patch.size)
                road_density = road_pixels / total_pixels if total_pixels else 0.0

                ptf = window_transform(window, src.transform)
                minx = ptf.c
                maxy = ptf.f
                maxx = minx + window.width * ptf.a
                miny = maxy + window.height * ptf.e
                patch_box = box(minx, miny, maxx, maxy)

                records.append(
                    {
                        "tile_id": tile_id,
                        "patch_row_id": ji[0],
                        "patch_col_id": ji[1],
                        "zone_name": zone_name,
                        "tile_path": paths["tile_path"],
                        "mask_raster_path": paths["mask_raster_path"],
                        "mask_graph_path": paths["mask_graph_path"],
                        "spatial_resolution": spatial_resolution,
                        "urbanisation_classification": EMPTY_LABEL,
                        "biome": SCAFFOLD_BIOME,
                        "road_density": road_density,
                        "split_set": SCAFFOLD_SPLIT,
                        "satellite_image_dates": list(SCAFFOLD_DATES),
                        "crs": crs_str,
                        "patch_bounding_geometry": patch_box,
                    }
                )
        return records, native_crs

    def _rel(self, path):
        """Path relative to the dataset dir (stored in the parquet)."""
        path = Path(path)
        try:
            return str(path.relative_to(self.dataset_dir))
        except ValueError:
            return str(path)

    # Jenks Natural Breaks
    # TODO: understand more deeply 
    def _classify(self, records):
        densities = np.array([r["road_density"] for r in records])
        values = densities[densities > 0]
        if values.size == 0:
            return

        # Jenks needs >= 3 distinct values for 3 classes; otherwise span the
        # range directly. Dedup break edges: degenerate inputs (few distinct
        # densities) make Jenks repeat edges, which pd.cut rejects, so the
        # number of usable bins -- and labels -- shrinks accordingly.
        if np.unique(values).size >= 3:
            breaks = sorted(set(jenkspy.jenks_breaks(values, n_classes=3)))
        else:
            breaks = sorted({float(values.min()), float(values.max())})

        n_bins = len(breaks) - 1
        if n_bins < 1:
            per_value = np.full(values.size, CLASS_LABELS[0])
        else:
            bin_labels = CLASS_LABELS[:n_bins]
            per_value = pd.cut(
                values, bins=breaks, labels=bin_labels, include_lowest=True
            ).astype(str)
            print("Density breaks:", [round(b, 5) for b in breaks], "->", bin_labels)

        li = 0
        for r in records:
            if r["road_density"] > 0:
                r["urbanisation_classification"] = per_value[li]
                li += 1
