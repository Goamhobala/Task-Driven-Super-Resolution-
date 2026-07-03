"""Per-zone (Large COG tiff) label generation from a combined road-vector GeoParquet.

  - 'RasterMaskLabeler` class -- buffered binary raster mask aligned to imagery.
  - `RoadGraphLabeler` class  -- patch-aligned road-segment vector parquet.

`load_zone_roads` function performs read sat meta, spatially filter + reproject + clip the road parquet to the COG footprint.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import box
from sentinel2data.generator.config import WGS84
from sentinel2data.generator.helper import window_box
from sentinel2data.generator.io import write_mask_cog


@dataclass
class ZoneRoads:
    """Roads clipped + reprojected to one satellite COG, plus that COG's meta."""

    roads: gpd.GeoDataFrame
    sat_meta: dict  # {meta, crs, bounds, blockxsize, blockysize}

    @property
    def crs(self):
        return self.sat_meta["crs"]

    @property
    def is_empty(self) -> bool:
        return self.roads.empty


def _read_sat_meta(sat_cog_path) -> dict:
    with rasterio.open(sat_cog_path) as src:
        return {
            "meta": src.meta.copy(),
            "crs": src.crs,
            "bounds": src.bounds,
            "blockxsize": src.profile.get("blockxsize", 256),
            "blockysize": src.profile.get("blockysize", 256),
        }


def load_zone_roads(sat_cog_path, roads_parquet_path) -> ZoneRoads:
    """Read the COG's metadata and the roads intersecting its footprint."""
    sat_meta = _read_sat_meta(sat_cog_path)
    sat_crs = sat_meta["crs"]
    footprint = box(*sat_meta["bounds"])

    footprint_gdf = gpd.GeoDataFrame({"geometry": [footprint]}, crs=sat_crs)
    minx, miny, maxx, maxy = footprint_gdf.to_crs(WGS84).total_bounds

    print("Filtering road parquet to COG footprint...")
    try:
        roads = gpd.read_parquet(roads_parquet_path, bbox=(minx, miny, maxx, maxy))
    except (ValueError, KeyError):
        # Parquet lacks a covering bbox column: read all, filter in memory.
        roads = gpd.read_parquet(roads_parquet_path)
        roads = roads.cx[minx:maxx, miny:maxy]

    if roads.empty:
        print("Warning: no roads found in footprint.")
        return ZoneRoads(gpd.GeoDataFrame({"geometry": []}, crs=sat_crs), sat_meta)

    print(f"Reprojecting and clipping {len(roads)} road segments...")
    roads = roads.to_crs(sat_crs)
    return ZoneRoads(gpd.clip(roads, footprint), sat_meta)


@runtime_checkable
class LabelGenerator(Protocol):
    """Produce one label artifact for a zone, returning its written path."""

    name: str

    def generate(self, sat_cog_path: Path, zone: ZoneRoads, out_path: Path) -> Path:
        ...


class RasterMaskLabeler:
    """Rasterize buffered roads into a binary, tiled, LZW mask COG.

    Each road is buffered by a half-width that depends on its ``class`` tier
    (major/medium), falling back to ``default_buffer_m`` for unmapped classes.
    """

    name = "mask_raster"

    def __init__(self, tier_buffer_m=None, default_buffer_m=DEFAULT_BUFFER_M):
        self.tier_buffer_m = dict(
            ROAD_TIER_BUFFER_M if tier_buffer_m is None else tier_buffer_m
        )
        self.default_buffer_m = default_buffer_m

    def buffer_distances(self, roads):
        """Per-row buffer radius (metres) from each road's ``class`` tier."""
        if "class" in roads.columns:
            dist = roads["class"].map(self.tier_buffer_m).fillna(self.default_buffer_m)
        else:
            dist = pd.Series(self.default_buffer_m, index=roads.index)
        return dist.to_numpy(dtype="float64")

    def rasterize(self, zone: ZoneRoads) -> np.ndarray:
        """Build the binary mask array (no disk IO) -- reusable by tilers."""
        meta = zone.sat_meta["meta"]
        roads = zone.roads
        if roads.empty:
            print("No roads to rasterize. Creating an empty mask.")
            return np.zeros((meta["height"], meta["width"]), dtype="uint8")

        distances = self.buffer_distances(roads)
        print(
            f"Rasterizing {len(roads)} roads with per-class buffers "
            f"({distances.min():.1f}-{distances.max():.1f} m half-width)..."
        )
        buffered = roads.geometry.buffer(distances)
        return features.rasterize(
            shapes=((geom, 1) for geom in buffered),
            out_shape=(meta["height"], meta["width"]),
            transform=meta["transform"],
            fill=0,
            all_touched=True,
            dtype="uint8",
        )

    def generate(self, sat_cog_path, zone: ZoneRoads, out_path) -> Path:
        mask = self.rasterize(zone)
        print(f"Writing tiled road-mask COG to {out_path}...")
        write_mask_cog(
            out_path,
            mask,
            zone.sat_meta["meta"],
            blockxsize=zone.sat_meta["blockxsize"],
            blockysize=zone.sat_meta["blockysize"],
        )
        print("Raster mask complete.")
        return Path(out_path)


class RoadGraphLabeler:
    """Split the road network into patches aligned to the COG's internal tiling.

    Writes one parquet row per (road segment ∩ patch).
    """

    name = "mask_graph"

    def generate(self, sat_cog_path, zone: ZoneRoads, out_path) -> Path:
        crs = zone.crs
        block_w = zone.sat_meta["blockxsize"]
        block_h = zone.sat_meta["blockysize"]
        roads = zone.roads
        sindex = roads.sindex if not roads.empty else None

        print(f"Extracting patch-aligned road graphs (block {block_w}x{block_h})...")
        parts = []
        with rasterio.open(sat_cog_path) as src:
            for ji, window in src.block_windows(1):
                patch_box = window_box(window, src.transform)
                patch_roads = self._clip_roads(roads, patch_box, sindex)
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
            gdf = roads.iloc[0:0].copy()
            gdf["patch_row_id"] = pd.Series(dtype="int64")
            gdf["patch_col_id"] = pd.Series(dtype="int64")

        n_patches = (
            0 if gdf.empty else gdf.groupby(["patch_row_id", "patch_col_id"]).ngroups
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Writing {len(gdf)} road segments across {n_patches} patches "
            f"to {out_path}..."
        )
        gdf.to_parquet(out_path)
        print("Road graph complete.")
        return out_path

    @staticmethod
    def _clip_roads(roads, patch_box, sindex):
        """Road centrelines intersected with one patch; empty if none."""
        empty = roads.iloc[0:0]
        if sindex is None:
            return empty
        candidates = roads.iloc[list(sindex.query(patch_box))]
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
