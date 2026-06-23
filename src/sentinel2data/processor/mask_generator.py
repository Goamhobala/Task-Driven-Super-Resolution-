import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from pathlib import Path

# Overture Buffer Value Classes (RoadMaskGenerator).
ROAD_CLASS_BUFFER_M = {
    "motorway": 15.0,
    "trunk": 12.0,
    "primary": 10.0,
    "secondary": 8.0,
    "tertiary": 6.0,
    "unclassified": 5.0,
    "residential": 5.0,
    "living_street": 4.0,
    "service": 3.0,
    "track": 3.0,
    "pedestrian": 3.0,
    "cycleway": 2.0,
    "footway": 2.0,
    "path": 2.0,
    "steps": 2.0,
    "bridleway": 2.0,
}

ROAD_TIER_BUFFER_M = {"major": 12.0, "medium": 7.0}
DEFAULT_BUFFER_M = 5.0

CDNGI_ROADS_LAYER = "TRAN_ROADS_EXP"
CDNGI_CLASS_MAP = {
    "National Freeway": "major",
    "On/OffRamp": "major",
    "National Road": "major",
    "Arterial Road": "major",
    "Main Road": "major",
    "Secondary Road": "medium",
}

OVERTURE_MAJOR_MEDIUM = ("motorway", "trunk", "primary", "secondary")
OVERTURE_CLASS_MAP = {
    "motorway": "major",
    "trunk": "major",
    "primary": "major",
    "secondary": "medium",
}

ROAD_VECTOR_COLUMNS = ("source", "source_class", "class", "province", "geometry")


class RoadVectorExtractor:
    """Extract major + medium scale roads from CDNGI (GeoPackage) and/or Overture
    (GeoParquet) into one normalized GeoParquet (or GeoPackage).

    Both sources are filtered to their large/medium classes and reprojected to
    EPSG:4326. Each source's native class is remapped onto two simplified,
    intuitive tiers -- major, medium (see ``CDNGI_CLASS_MAP`` /
    ``OVERTURE_CLASS_MAP``) -- carried in the ``class`` column; buffer widths
    per tier live in ``ROAD_TIER_BUFFER_M``.

    This is a plain stacked union (rows tagged by ``source``); it does NOT
    deduplicate roads that both datasets digitize. Overlap conflation is a
    separate step.
    """

    def __init__(
        self,
        out_path,
        cdngi_path=None,
        overture_path=None,
        cdngi_layer=CDNGI_ROADS_LAYER,
    ):
        if cdngi_path is None and overture_path is None:
            raise ValueError("Provide at least one of cdngi_path or overture_path.")
        self.out_path = Path(out_path)
        self.cdngi_path = Path(cdngi_path) if cdngi_path is not None else None
        self.overture_path = Path(overture_path) if overture_path is not None else None
        self.cdngi_layer = cdngi_layer

    def _cdngi_gpkg_paths(self):
        """Resolve cdngi_path to a sorted list of .gpkg files (file or directory)."""
        if self.cdngi_path.is_dir():
            return sorted(self.cdngi_path.rglob("*.gpkg"))
        return [self.cdngi_path]

    def _load_cdngi(self):
        """Read major+medium roads from one or many CDNGI GeoPackages, normalized."""
        keys = list(CDNGI_CLASS_MAP)
        where = "FEAT_TYPE IN ({})".format(", ".join(f"'{k}'" for k in keys))

        parts = []
        for gpkg in self._cdngi_gpkg_paths():
            province = gpkg.stem.split("_")[0]
            print(f"Reading CDNGI {gpkg.name} (province {province})...")
            gdf = gpd.read_file(
                gpkg, layer=self.cdngi_layer, columns=["FEAT_TYPE"], where=where
            )
            if gdf.empty:
                continue
            gdf = gdf.to_crs("EPSG:4326")
            out = gpd.GeoDataFrame(
                {
                    "source": "cdngi",
                    "source_class": gdf["FEAT_TYPE"].to_numpy(),
                    "class": gdf["FEAT_TYPE"].map(CDNGI_CLASS_MAP).to_numpy(),
                    "province": province,
                    "geometry": gdf.geometry.to_numpy(),
                },
                crs="EPSG:4326",
            )
            parts.append(out)

        if not parts:
            return None
        combined = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
        )
        print(f"CDNGI: {len(combined)} major+medium road segments.")
        return combined

    def _load_overture(self):
        """Read major+medium roads from an Overture roads GeoParquet, normalized.

        Uses Arrow predicate pushdown so only matching rows are materialized."""
        print(f"Reading Overture {self.overture_path.name} (predicate pushdown)...")
        filters = [
            ("subtype", "==", "road"),
            ("class", "in", list(OVERTURE_MAJOR_MEDIUM)),
        ]
        gdf = gpd.read_parquet(
            self.overture_path,
            columns=["subtype", "class", "subclass", "geometry"],
            filters=filters,
        )
        if gdf.empty:
            return None
        gdf = gdf.to_crs("EPSG:4326")
        out = gpd.GeoDataFrame(
            {
                "source": "overture",
                "source_class": gdf["class"].to_numpy(),
                "class": gdf["class"].map(OVERTURE_CLASS_MAP).to_numpy(),
                "province": None,
                "geometry": gdf.geometry.to_numpy(),
            },
            crs="EPSG:4326",
        )
        print(f"Overture: {len(out)} major+medium road segments.")
        return out

    def build(self):
        """Load the requested sources, stack, and write the combined layer.
        Returns the output path."""
        parts = []
        if self.cdngi_path is not None:
            cdngi = self._load_cdngi()
            if cdngi is not None:
                parts.append(cdngi)
        if self.overture_path is not None:
            overture = self._load_overture()
            if overture is not None:
                parts.append(overture)

        if not parts:
            raise ValueError("No road features extracted from the given sources.")

        combined = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )[list(ROAD_VECTOR_COLUMNS)]

        by_class = combined.groupby("class").size().to_dict()
        print(f"Combined: {len(combined)} segments by class: {by_class}")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing combined roads to {self.out_path}...")
        if self.out_path.suffix.lower() == ".gpkg":
            combined.to_file(self.out_path, driver="GPKG", layer="roads_major_medium")
        else:
            # Per-row covering bbox so downstream readers can spatially filter
            # (RoadMaskGenerator reads a per-COG bbox window).
            combined.to_parquet(self.out_path, write_covering_bbox=True)
        print("Road vector extraction complete.")
        return self.out_path


class RoadMaskGenerator:
    """Derives road artifacts for one satellite COG from a combined road-vector
    GeoParquet (see ``RoadVectorExtractor``; ``class`` holds major/medium tiers).
    """

    def __init__(
        self,
        sat_cog_path,
        roads_parquet_path,
        out_mask_path,
        out_graph_path,
        class_buffer_m=None,
        default_buffer_m=DEFAULT_BUFFER_M,
    ):
        self.sat_cog_path = Path(sat_cog_path)
        self.roads_parquet_path = Path(roads_parquet_path)
        self.out_mask_path = Path(out_mask_path)
        self.out_graph_path = Path(out_graph_path)

        self.class_buffer_m = dict(
            ROAD_TIER_BUFFER_M if class_buffer_m is None else class_buffer_m
        )
        self.default_buffer_m = default_buffer_m

        # store metadata and road graphs
        self.sat_meta = self._read_sat_meta()
        self.roads = self._load_roads()

    def _read_sat_meta(self):
        with rasterio.open(self.sat_cog_path) as src:
            sat_meta = {
                "meta": src.meta.copy(),
                "crs": src.crs,
                "bounds": src.bounds,
                "blockxsize": src.profile.get("blockxsize", 256),
                "blockysize": src.profile.get("blockysize", 256),
            }
        return sat_meta

    def _load_roads(self):
        sat_crs = self.sat_meta["crs"]
        footprint = box(*self.sat_meta["bounds"])

        footprint_gdf = gpd.GeoDataFrame({"geometry": [footprint]}, crs=sat_crs)
        minx, miny, maxx, maxy = footprint_gdf.to_crs("EPSG:4326").total_bounds

        print("Filtering road parquet to COG footprint...")
        try:
            roads = gpd.read_parquet(
                self.roads_parquet_path, bbox=(minx, miny, maxx, maxy)
            )
        except (ValueError, KeyError):
            # Parquet lacks a covering bbox column: read all, filter in memory.
            roads = gpd.read_parquet(self.roads_parquet_path)
            roads = roads.cx[minx:maxx, miny:maxy]

        if roads.empty:
            print("Warning: no roads found in footprint.")
            return gpd.GeoDataFrame({"geometry": []}, crs=sat_crs)

        print(f"Reprojecting and clipping {len(roads)} road segments...")
        roads = roads.to_crs(sat_crs)
        return gpd.clip(roads, footprint)

    def _clip_roads(self, patch_box, sindex):
        """Road centrelines intersected with one patch; empty geometry if none."""
        empty = self.roads.iloc[0:0]
        if sindex is None:
            return empty
        candidates = self.roads.iloc[list(sindex.query(patch_box))]
        if candidates.empty:
            return empty

        clipped_geom = candidates.geometry.intersection(patch_box)
        # Drop empties and point-only touches (length 0); keep line content.
        keep = ~clipped_geom.is_empty & (clipped_geom.length > 0)
        if not keep.any():
            return empty

        out = candidates.loc[keep].copy()
        out.geometry = clipped_geom.loc[keep]
        return out


    def _road_buffer_distances(self, roads):
        """Per-row buffer radius (metres) from each road's ``class`` tier
        (major/medium), falling back to ``default_buffer_m`` for unmapped or
        missing classes."""
        if "class" in roads.columns:
            dist = roads["class"].map(self.class_buffer_m).fillna(self.default_buffer_m)
        else:
            dist = pd.Series(self.default_buffer_m, index=roads.index)
        return dist.to_numpy(dtype="float64")

    def generate_raster_mask(self):
        """Rasterize the buffered roads into a binary, internally-tiled,
        LZW-compressed mask COG aligned to the satellite imagery. Each road is
        buffered by a width that depends on its ``class`` tier."""
        sat_meta = self.sat_meta["meta"]
        roads = self.roads

        if roads.empty:
            print("No roads to rasterize. Creating an empty mask.")
            mask = np.zeros((sat_meta["height"], sat_meta["width"]), dtype="uint8")
        else:
            distances = self._road_buffer_distances(roads)
            print(
                f"Rasterizing {len(roads)} roads with per-class buffers "
                f"({distances.min():.1f}-{distances.max():.1f} m half-width)..."
            )
            buffered = roads.geometry.buffer(distances)
            shapes = ((geom, 1) for geom in buffered)
            mask = features.rasterize(
                shapes=shapes,
                out_shape=(sat_meta["height"], sat_meta["width"]),
                transform=sat_meta["transform"],
                fill=0,
                all_touched=True,
                dtype="uint8",
            )

        sat_meta.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="lzw",
            tiled=True,
            blockxsize=self.sat_meta["blockxsize"],
            blockysize=self.sat_meta["blockysize"],
        )

        self.out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing tiled road-mask COG to {self.out_mask_path}...")
        with rasterio.open(self.out_mask_path, "w", **sat_meta) as dest:
            dest.write(mask, 1)
        print("Raster mask complete.")
        return self.out_mask_path

    def generate_road_graph(self):
        """Split the road network into patches aligned to the COG's internal
        tiling. Writes one parquet row per (road segment ∩ patch).
        Returns the path of the written parquet.
        """
        crs = self.sat_meta["crs"]
        block_w = self.sat_meta["blockxsize"]
        block_h = self.sat_meta["blockysize"]
        sindex = self.roads.sindex if not self.roads.empty else None

        print(f"Extracting patch-aligned road graphs (block {block_w}x{block_h})...")
        parts = []
        with rasterio.open(self.sat_cog_path) as src:
            for ji, window in src.block_windows(1):
                ptf = window_transform(window, src.transform)
                minx = ptf.c
                maxy = ptf.f
                maxx = minx + window.width * ptf.a
                miny = maxy + window.height * ptf.e
                patch_box = box(minx, miny, maxx, maxy)

                patch_roads = self._clip_roads(patch_box, sindex)
                if patch_roads.empty:
                    continue
                patch_roads = patch_roads.copy()
                patch_roads["patch_row_id"] = ji[0]
                patch_roads["patch_col_id"] = ji[1]
                parts.append(patch_roads)

        if parts:
            gdf = gpd.GeoDataFrame(
                pd.concat(parts, ignore_index=True), geometry="geometry", crs=crs
            )
        else:
            # No road hit any patch: keep the full attribute schema, zero rows.
            gdf = self.roads.iloc[0:0].copy()
            gdf["patch_row_id"] = pd.Series(dtype="int64")
            gdf["patch_col_id"] = pd.Series(dtype="int64")

        n_patches = (
            0 if gdf.empty else gdf.groupby(["patch_row_id", "patch_col_id"]).ngroups
        )
        self.out_graph_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Writing {len(gdf)} road segments across {n_patches} patches "
            f"to {self.out_graph_path}..."
        )
        gdf.to_parquet(self.out_graph_path)
        print("Road graph complete.")
        return self.out_graph_path