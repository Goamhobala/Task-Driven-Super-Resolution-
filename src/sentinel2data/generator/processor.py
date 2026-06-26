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
from pathlib import Path
from typing import Protocol, runtime_checkable
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from sentinel2data.generator.config import (
    V1_SCHEMA,
    SCAFFOLD_BIOME,
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
    """S2-ROSA-V2 split-aware processor.

    The output shape depends on the zone's split (decided by the pipeline first):
      * ``train`` -> road-bearing 512px **band-interleaved tiled COG** tiles (drop
        empty), one row/tile.
      * ``val`` / ``test`` -> the **whole zone as a COG**, one row/zone.

    Every output image gets 3 appended CLAHE+gamma enhanced-RGB bands
    (:func:`enhance_rgb`), a raster mask, and a road-graph parquet, written under
    ``<root>/<split>/{imagery,masks_raster,masks_graph}``. ``process`` returns a
    list of row dicts (the split-first pipeline concatenates + assigns ``image_id``).
    """

    schema = V2_SCHEMA

    def __init__(self, roads_parquet_path, tile_spec=None, mask_labeler=None,
                 enhance_cfg=None):
        self.roads_parquet_path = Path(roads_parquet_path)
        self.tile_spec = tile_spec or TileSpec()
        self.mask_labeler = mask_labeler or RasterMaskLabeler()
        self.enhance_cfg = enhance_cfg or RGBEnhanceConfig()

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
            if split == "train":
                return self._train_tiles(
                    zone, sindex, img, full_mask, zone_name, layout, paths
                )
            return self._whole_zone(
                zone, img, full_mask, zone_name, split, layout, paths
            )

    # -- train: raw 512px tiles -------------------------------------------
    def _train_tiles(self, zone, sindex, img, full_mask, zone_name, layout, paths):
        ts = self.tile_spec.tile_size
        patch = self.tile_spec.patch_size
        rgb_idx = [b - 1 for b in self.enhance_cfg.rgb_bands]
        rows = []
        n_rows = img.height // ts  # full tiles only -> edge remainder dropped
        n_cols = img.width // ts
        kept = dropped = 0
        for r in range(n_rows):
            for c in range(n_cols):
                win = Window(c * ts, r * ts, ts, ts)
                mask_arr = full_mask[r * ts : (r + 1) * ts, c * ts : (c + 1) * ts]
                road_px = int(np.count_nonzero(mask_arr > 0))
                if road_px == 0:
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
                    paths=paths, zone_name=zone_name, split="train", is_tile=True,
                    tile_row=r, tile_col=c, img_path=img_path, msk_path=msk_path,
                    gph_path=gph_path, bounds=window_bounds(win, img.transform),
                    src_crs=img.crs, res=abs(win_tf.a), total_px=ts * ts,
                    road_px=road_px, tile_size=ts, band_count=out.shape[0],
                ))
                kept += 1
        partial = n_cols * (img.width % ts > 0) + n_rows * (img.height % ts > 0)
        print(
            f"  train grid {n_rows}x{n_cols}: kept {kept}, dropped {dropped} "
            f"empty, {partial} partial edge strips ignored"
        )
        return rows

    # -- val/test: whole-zone COG -----------------------------------------
    def _whole_zone(self, zone, img, full_mask, zone_name, split, layout, paths):
        block = self.tile_spec.patch_size
        rgb = img.read(self.enhance_cfg.rgb_bands).astype("float32")  # (3, H, W)
        enhanced = enhance_rgb(rgb, self.enhance_cfg)
        del rgb

        img_path = layout["imagery"] / f"{zone_name}.tif"
        msk_path = layout["masks_raster"] / f"{zone_name}.tif"
        gph_path = layout["masks_graph"] / f"{zone_name}.parquet"
        self._write_zone_imagery(img, enhanced, img_path, block)
        write_mask_cog(
            msk_path, full_mask, img.meta, tiled=True, blockxsize=block, blockysize=block
        )
        self._write_graph(zone, None, None, gph_path)

        road_px = int(np.count_nonzero(full_mask > 0))
        left, bottom, right, top = img.bounds
        row = self._row(
            paths=paths, zone_name=zone_name, split=split, is_tile=False,
            tile_row=None, tile_col=None, img_path=img_path, msk_path=msk_path,
            gph_path=gph_path, bounds=(left, bottom, right, top), src_crs=img.crs,
            res=abs(img.transform.a), total_px=img.width * img.height,
            road_px=road_px, tile_size=None, band_count=img.count + 3,
        )
        return [row]

    def _write_zone_imagery(self, src, enhanced, out_path, block):
        """Stream the source bands + appended enhanced bands into one COG (keeps
        peak memory to one source band + the enhanced triplet, not the whole stack)."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile = dict(src.profile)
        profile.update(
            count=src.count + enhanced.shape[0],
            dtype="float32",
            transform=src.transform,
            compress="deflate",
            predictor=1,  # float predictor (3) hurts noisy S2 reflectance
            tiled=True,
            blockxsize=block,
            blockysize=block,
            interleave="band",
        )
        names = list(S2_V2_BANDS)
        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                dst.write(src.read(b).astype("float32"), b)
            for j in range(enhanced.shape[0]):
                dst.write(enhanced[j], src.count + 1 + j)
            for i, name in enumerate(names, start=1):
                dst.set_band_description(i, name)
        return out_path

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

    def _row(self, *, paths, zone_name, split, is_tile, tile_row, tile_col,
             img_path, msk_path, gph_path, bounds, src_crs, res, total_px,
             road_px, tile_size, band_count):
        west, south, east, north = reproject_bounds(bounds, src_crs, WGS84)
        return {
            "image_id": -1,  # assigned by the split-first pipeline
            "zone_name": zone_name,
            "split_set": split,
            "is_tile": is_tile,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "image_path": _rel(paths.root, img_path),
            "mask_path": _rel(paths.root, msk_path),
            "mask_graph_path": _rel(paths.root, gph_path),
            "tile_size": tile_size,
            "band_count": band_count,
            "spatial_resolution": res,
            "road_pixels": road_px,
            "road_density": road_px / total_px,
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
