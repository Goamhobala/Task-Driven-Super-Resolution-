"""Zone processors: how one source COG becomes catalogue rows + tile artifacts.

This is the main axis a new dataset *variant* swaps. Each :class:`ZoneProcessor`
owns its output layout and row schema and composes :mod:`labels` generators:

  * :class:`V2ROSAProcessor` (V2ROSA) -- split-aware: train zones -> raw 512px
    GTiff tiles, val/test zones -> whole-zone COG; each image gets appended
    enhanced-RGB bands + a road-graph parquet, under ``<root>/<split>/``.
  * :class:`V1ROSAProcessor` (V1ROSA) -- write an in-place full-zone raster mask +
    road-graph parquet, one row per internal mask block-window.

``V1ROSAProcessor.process(zone_id, sat, paths)`` returns a per-zone GeoDataFrame for
the generic pipeline; ``V2ROSAProcessor.process(zone_id, sat, split, paths)`` returns
row dicts for the split-first pipeline (which assigns ``image_id``).
"""
import zlib
from pathlib import Path
from typing import Protocol, runtime_checkable
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from sentinel2data.generator.config import (
    SCAFFOLD_DATES,
    V2_SCHEMA,
    UNKNOWN_BIOME,
    CatalogueSchema,
    DatasetPaths,
    EMPTY_LABEL,
    RGBEnhanceConfig,
    TileSpec,
    WGS84,
    split_layout,
)
from sentinel2data.generator.helper import (
    enhance_rgb,
    reproject_bounds,
    window_bounds,
    window_box,
)
from sentinel2data.dataset.bands import S2_V2_BANDS
from sentinel2data.generator.io import write_image_cog, write_mask_cog
from sentinel2data.generator.labels import (
    RasterMaskLabeler,
    RoadGraphLabeler,
    load_zone_roads,
)


@runtime_checkable
class ZoneProcessor(Protocol):
    """Turn one zone COG into per-zone catalogue rows (+ side-effect artifacts)."""

    schema: CatalogueSchema

    def process(
        self, zone_id: int, sat_cog_path: Path, paths: DatasetPaths
    ) -> "gpd.GeoDataFrame | None":
        ...


def _rel(root, path):
    """Path relative to the dataset root (stored in the parquet)."""
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


class V2ROSAProcessor:
    """S2-ROSA-V2 processor: cut every zone into non-overlapping 512px tiles.

    All splits are tiled identically (train/val/test) -- the pipeline decides a
    zone's split first, then this processor writes its tiles under
    ``<root>/<split>/{imagery,masks_raster,masks_graph}``, one catalogue row per
    tile. Non-empty tiles are always kept; empty (no-road) tiles are kept with
    probability ``empty_keep_ratio`` (seeded per zone, same for every split), so
    ``1.0`` drops nothing and ``0.0`` keeps only road-bearing tiles.

    Every tile gets 3 appended CLAHE+gamma enhanced-RGB bands (:func:`enhance_rgb`),
    a raster mask, and a road-graph parquet. ``process`` returns a list of row dicts
    (the split-first pipeline concatenates + assigns ``image_id``).
    """

    schema = V2_SCHEMA

    def __init__(self, roads_parquet_path, tile_spec=None, mask_labeler=None,
                 enhance_cfg=None, empty_keep_ratio=1.0, tile_seed=42):
        self.roads_parquet_path = Path(roads_parquet_path)
        self.tile_spec = tile_spec or TileSpec()
        self.mask_labeler = mask_labeler or RasterMaskLabeler()
        self.enhance_cfg = enhance_cfg or RGBEnhanceConfig()
        self.empty_keep_ratio = float(empty_keep_ratio)
        self.tile_seed = int(tile_seed)

    # -- entry -------------------------------------------------------------
    def process(self, zone_id, sat_cog_path, split, paths):
        sat_cog_path = Path(sat_cog_path)
        zone_name = sat_cog_path.stem
        print(f"[{zone_id}] {zone_name} -> {split}")

        layout = split_layout(paths.root, split)
        zone = load_zone_roads(sat_cog_path, self.roads_parquet_path)
        full_mask = self.mask_labeler.rasterize(zone)  # (H, W) uint8
        sindex = zone.roads.sindex if not zone.roads.empty else None

        with rasterio.open(sat_cog_path) as img:
            return self._tile_zone(
                zone, sindex, img, full_mask, zone_name, split, layout, paths
            )

    # -- tile every split into raw 512px tiles ----------------------------
    def _tile_zone(self, zone, sindex, img, full_mask, zone_name, split, layout, paths):
        ts = self.tile_spec.tile_size
        patch = self.tile_spec.patch_size
        rgb_idx = [b - 1 for b in self.enhance_cfg.rgb_bands]
        n_rows = img.height // ts  # full tiles only -> edge remainder dropped
        n_cols = img.width // ts

        keep_empty = self._empty_keep_mask(zone_name, full_mask, ts, n_rows, n_cols)
        rows = []
        kept = kept_road = dropped = 0
        for r in range(n_rows):
            for c in range(n_cols):
                win = Window(c * ts, r * ts, ts, ts)
                mask_arr = full_mask[r * ts : (r + 1) * ts, c * ts : (c + 1) * ts]
                road_px = int(np.count_nonzero(mask_arr > 0))
                if road_px == 0 and not keep_empty[r, c]:
                    dropped += 1
                    continue

                img_arr = img.read(window=win).astype("float32")  # (C, ts, ts)
                enhanced = enhance_rgb(img_arr[rgb_idx], self.enhance_cfg)
                out = np.concatenate([img_arr, enhanced], axis=0)  # (C+3, ts, ts)
                win_tf = window_transform(win, img.transform)

                stem = f"{zone_name}_r{r}_c{c}"
                img_path = layout["imagery"] / f"{stem}.tif"
                msk_path = layout["masks_raster"] / f"{stem}.tif"
                gph_path = layout["masks_graph"] / f"{stem}.parquet"
                write_image_cog(img_path, out, img.profile, transform=win_tf,
                                tiled=True, blockxsize=patch, blockysize=patch,
                                interleave="band",
                                band_names=list(S2_V2_BANDS))
                write_mask_cog(msk_path, mask_arr, img.meta, transform=win_tf,
                               tiled=True, blockxsize=patch, blockysize=patch)
                self._write_graph(zone, window_box(win, img.transform), sindex, gph_path)

                rows.append(self._row(
                    paths=paths, zone_name=zone_name, split=split,
                    img_path=img_path, msk_path=msk_path, gph_path=gph_path,
                    bounds=window_bounds(win, img.transform), src_crs=img.crs,
                    res=abs(win_tf.a), total_px=ts * ts, road_px=road_px,
                    tile_size=ts, band_count=out.shape[0],
                ))
                kept += 1
                kept_road += road_px > 0
        partial = n_cols * (img.width % ts > 0) + n_rows * (img.height % ts > 0)
        print(
            f"  {split} grid {n_rows}x{n_cols}: kept {kept} "
            f"({kept_road} road, {kept - kept_road} empty), dropped {dropped} "
            f"empty, {partial} partial edge strips ignored"
        )
        return rows

    def _empty_keep_mask(self, zone_name, full_mask, ts, n_rows, n_cols):
        """Boolean ``(n_rows, n_cols)``: which empty tiles to keep (seeded per zone).

        Non-empty tiles are left ``True`` (the caller keeps them regardless). An RNG
        seeded by ``(tile_seed, crc32(zone_name))`` -- stable across runs, unlike
        Python ``hash`` -- draws one uniform per empty tile and keeps it when the
        draw is below ``empty_keep_ratio``. ``ratio >= 1.0`` short-circuits to keep
        all; ``ratio <= 0`` drops every empty tile.
        """
        ratio = self.empty_keep_ratio
        keep = np.ones((n_rows, n_cols), dtype=bool)
        if ratio >= 1.0:
            return keep
        rng = np.random.default_rng([self.tile_seed, zlib.crc32(zone_name.encode())])
        for r in range(n_rows):
            for c in range(n_cols):
                block = full_mask[r * ts : (r + 1) * ts, c * ts : (c + 1) * ts]
                if np.count_nonzero(block > 0) == 0:  # empty -> sample
                    keep[r, c] = rng.random() < ratio
        return keep

    # -- shared ------------------------------------------------------------
    def _write_graph(self, zone, clip_box, sindex, out_path):
        """Roads clipped to an image's extent (tile) or the whole zone -> parquet."""
        if clip_box is not None:
            roads = RoadGraphLabeler._clip_roads(zone.roads, clip_box, sindex)
        else:
            roads = zone.roads
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        roads.to_parquet(out_path)
        return out_path

    def _row(self, *, paths, zone_name, split, img_path, msk_path, gph_path,
             bounds, src_crs, res, total_px, road_px, tile_size, band_count):
        west, south, east, north = reproject_bounds(bounds, src_crs, WGS84)
        return {
            "image_id": -1,  # assigned by the split-first pipeline
            "zone_name": zone_name,
            "split_set": split,
            "image_path": _rel(paths.root, img_path),
            "mask_path": _rel(paths.root, msk_path),
            "mask_graph_path": _rel(paths.root, gph_path),
            "tile_size": tile_size,
            "band_count": band_count,
            "spatial_resolution": res,
            "road_pixels": road_px,
            "road_density": road_px / total_px,
            "urbanisation_classification": EMPTY_LABEL,  # set per split by the tagger
            "biome": UNKNOWN_BIOME,
            "satellite_image_dates": list(SCAFFOLD_DATES),
            "crs": src_crs.to_string(),
            "geometry": box(west, south, east, north),
        }


class V1ROSAProcessor:
    """Index a zone COG's internal mask block-windows as patches (v1).

    Writes a full-zone raster mask + patch-aligned road-graph parquet in place
    (``masks_raster/``, ``masks_graph/``), then records one row per mask block.
    """

    schema = V1_SCHEMA

    def __init__(self, roads_parquet_path, mask_labeler=None, graph_labeler=None):
        self.roads_parquet_path = Path(roads_parquet_path)
        self.mask_labeler = mask_labeler or RasterMaskLabeler()
        self.graph_labeler = graph_labeler or RoadGraphLabeler()

    def process(self, zone_id, sat_cog_path, paths):
        sat_cog_path = Path(sat_cog_path)
        zone_name = sat_cog_path.stem
        print(f"[{zone_id}] {zone_name}")

        mask_path = paths.masks_raster_dir / f"{zone_name}_mask.tif"
        graph_path = paths.masks_graph_dir / f"{zone_name}_graphs.parquet"

        zone = load_zone_roads(sat_cog_path, self.roads_parquet_path)
        self.mask_labeler.generate(sat_cog_path, zone, mask_path)
        self.graph_labeler.generate(sat_cog_path, zone, graph_path)

        rel_paths = {
            "tile_path": _rel(paths.root, sat_cog_path),
            "mask_raster_path": _rel(paths.root, mask_path),
            "mask_graph_path": _rel(paths.root, graph_path),
        }
        records, native_crs = self._scan_patches(zone_id, zone_name, rel_paths, mask_path)
        if not records:
            return None
        return gpd.GeoDataFrame(
            records, geometry="patch_bounding_geometry", crs=native_crs
        ).to_crs(WGS84)

    def _scan_patches(self, zone_id, zone_name, rel_paths, mask_path):
        print(f"Scanning mask blocks: {zone_name}")
        records = []
        with rasterio.open(mask_path) as src:
            native_crs = src.crs
            crs_str = src.crs.to_string()
            spatial_resolution = abs(src.transform.a)
            for ji, window in src.block_windows(1):
                patch = src.read(1, window=window)
                road_pixels = int(np.count_nonzero(patch > 0))
                total_pixels = int(patch.size)
                road_density = road_pixels / total_pixels if total_pixels else 0.0
                records.append(
                    {
                        "tile_id": zone_id,
                        "patch_row_id": ji[0],
                        "patch_col_id": ji[1],
                        "zone_name": zone_name,
                        **rel_paths,
                        "spatial_resolution": spatial_resolution,
                        "urbanisation_classification": EMPTY_LABEL,
                        "biome": SCAFFOLD_BIOME,
                        "road_density": road_density,
                        "split_set": None,  # assigned by the SplitStrategy
                        "satellite_image_dates": list(SCAFFOLD_DATES),
                        "crs": crs_str,
                        "patch_bounding_geometry": window_box(window, src.transform),
                    }
                )
        return records, native_crs
