Intial test code

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from shapely import wkb
import jenkspy

CLASS_LABELS = ["Rural", "Peri-Urban", "Urban"]
EMPTY_LABEL = "Empty"


class MetadataGenerator:
    """Turns one road-mask COG into per-tile geoparquet records.

    Each internal block window of the mask COG becomes one tile row holding:
    indexing (image/tile), paths, road density + Jenks classification, a
    per-tile slice of the road-network graph (WKB), and a BBox geometry.

    split_set and satellite_image_date are emitted as None placeholders; they
    are decided/obtained later in the pipeline.
    """

    def __init__(
        self,
        mask_cog_path,
        image_index,
        image_name,
        image_path,
        mask_raster_path,
        road_graph=None,
    ):
        self.mask_cog_path = mask_cog_path
        self.image_index = image_index
        self.image_name = image_name
        self.image_path = image_path            # relative path stored in parquet
        self.mask_raster_path = mask_raster_path  # relative path stored in parquet
        self.road_graph = road_graph            # GeoDataFrame (sat CRS) or None
        self.crs = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def build_records(self):
        """Return a GeoDataFrame of tile records for this image."""
        records = self._scan_tiles()
        self._classify(records)
        return gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs)

    # ------------------------------------------------------------------ #
    # Tile scan
    # ------------------------------------------------------------------ #
    def _scan_tiles(self):
        print(f"Scanning mask blocks: {self.image_name}")
        roads = self.road_graph
        sindex = roads.sindex if roads is not None and not roads.empty else None

        records = []
        tile_index = 0
        with rasterio.open(self.mask_cog_path) as src:
            self.crs = src.crs
            crs_str = src.crs.to_string()  # native UTM; road_network stays in it
            for ji, window in src.block_windows(1):
                patch = src.read(1, window=window)

                # Skip ragged edge tiles that are not a full block.
                if patch.shape[0] != window.height or patch.shape[1] != window.width:
                    print(f"Warning: skipping non-square edge tile {ji}.")
                    continue

                road_pixels = int(np.count_nonzero(patch > 0))
                total_pixels = int(patch.size)
                road_density = road_pixels / total_pixels if total_pixels else 0.0

                ptf = window_transform(window, src.transform)
                minx = ptf.c
                maxy = ptf.f
                maxx = minx + window.width * ptf.a
                miny = maxy + window.height * ptf.e
                tile_box = box(minx, miny, maxx, maxy)

                records.append(
                    {
                        "image_index": self.image_index,
                        "tile_index": tile_index,
                        "image_name": self.image_name,
                        "image_path": self.image_path,
                        "mask_raster_path": self.mask_raster_path,
                        "classification": EMPTY_LABEL,
                        "road_density": road_density,
                        "split_set": None,
                        "satellite_image_date": None,
                        "road_network": self._tile_road_network(tile_box, sindex),
                        "crs": crs_str,
                        "window_row": ji[0],
                        "window_col": ji[1],
                        "geometry": tile_box,
                    }
                )
                tile_index += 1
        return records

    def _tile_road_network(self, tile_box, sindex):
        """WKB of the road centrelines intersected with this tile, or None."""
        if sindex is None:
            return None
        candidates = self.road_graph.iloc[list(sindex.query(tile_box))]
        if candidates.empty:
            return None
        clipped = candidates.intersection(tile_box)
        clipped = clipped[~clipped.is_empty]
        if clipped.empty:
            return None
        return wkb.dumps(clipped.union_all())

    # ------------------------------------------------------------------ #
    # Jenks classification over the image's tile densities
    # ------------------------------------------------------------------ #
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
            # All road tiles share one density value.
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
                r["classification"] = per_value[li]
                li += 1
