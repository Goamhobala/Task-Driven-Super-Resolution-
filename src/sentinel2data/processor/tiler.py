"""Tile big satellite COGs into small, road-bearing 512x512 COG tiles.

Pipeline per source COG (one zone):
  1. rasterise the road vector into a full-zone binary mask (RoadMaskGenerator).
  2. walk a grid of ``tile_size`` x ``tile_size`` windows, **dropping partial
     edge windows** (only full tiles are kept) so every tile divides cleanly
     into ``patch_size`` patches for a torchgeo sampler.
  3. **drop any tile whose mask is all-zero** (no roads).
  4. write each surviving tile as an internally-tiled (``patch_size`` blocks)
     image COG + mask COG, and record one metadata row per tile.

Metadata is now **one row per tile** (not per internal patch): the torchgeo
``GridGeoSampler`` / ``RandomGeoSampler`` carves 256x256 patches at read time,
so the catalogue only needs to describe whole tiles. Rows are augmented with
the NVM2024 biome (see :mod:`sentinel2data.processor.biome`).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box

from sentinel2data.processor.biome import BiomeTagger, UNKNOWN_BIOME
from sentinel2data.processor.mask_generator import RoadMaskGenerator

IMAGE_EXTS = (".tif", ".tiff")
COMMON_CRS = "EPSG:4326"
SCAFFOLD_DATES = ["2023-01-01"]

# One row per kept tile.
TILE_METADATA_COLUMNS = [
    "tile_id",
    "zone_name",
    "tile_row",
    "tile_col",
    "image_path",
    "mask_path",
    "tile_size",
    "patch_size",
    "spatial_resolution",
    "road_pixels",
    "road_density",
    "biome",
    "split_set",
    "satellite_image_dates",
    "crs",
    "geometry",
]

SPLIT_CSV_COLUMNS = ["tile_id", "zone_name", "image_path", "mask_path", "split_set"]


class DatasetTiler:
    """Cut a directory of satellite COGs into a torchgeo-ready tiled dataset."""

    def __init__(
        self,
        imagery_dir,
        output_dir,
        roads_parquet_path,
        biome_parquet_path=None,
        tile_size=512,
        patch_size=256,
        buffer_m=5,
        val_frac=0.1,
        test_frac=0.1,
        split_seed=42,
    ):
        self.imagery_dir = Path(imagery_dir)
        self.output_dir = Path(output_dir)
        self.roads_parquet_path = Path(roads_parquet_path)
        self.biome_parquet_path = (
            Path(biome_parquet_path) if biome_parquet_path is not None else None
        )

        if tile_size % patch_size != 0:
            raise ValueError(
                f"tile_size ({tile_size}) must be a whole multiple of "
                f"patch_size ({patch_size}) so tiles divide into clean patches."
            )
        self.tile_size = tile_size
        self.patch_size = patch_size
        self.buffer_m = buffer_m

        self.val_frac = val_frac
        self.test_frac = test_frac
        self.split_seed = split_seed

        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.splits_dir = self.output_dir / "splits"
        self.metadata_path = self.output_dir / "metadata.parquet"

    # -- scanning ----------------------------------------------------------
    def scan_imagery(self):
        """Return sorted source COG paths, skipping macOS dotfiles and masks."""
        paths = []
        for ext in IMAGE_EXTS:
            paths.extend(self.imagery_dir.glob(f"*{ext}"))
        out = []
        for p in sorted(paths):
            if p.name.startswith("._") or p.name.endswith("_mask.tif"):
                continue
            out.append(p)
        return out

    # -- orchestration -----------------------------------------------------
    def build(self):
        images = self.scan_imagery()
        if not images:
            print(f"Aborting. No satellite imagery found in {self.imagery_dir}")
            return None

        print(f"Found {len(images)} source COG(s). Tile {self.tile_size}px, "
              f"patch {self.patch_size}px.")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for zone_id, sat_path in enumerate(images):
            print("-" * 50)
            print(f"[{zone_id}] {sat_path.stem}")
            records.extend(self._tile_zone(zone_id, sat_path))

        if not records:
            print("No road-bearing tiles produced; nothing written.")
            return None

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=COMMON_CRS)
        gdf["tile_id"] = range(len(gdf))

        self._tag_biomes(gdf)
        self._assign_split(gdf)

        gdf = gdf[TILE_METADATA_COLUMNS]
        gdf.to_parquet(self.metadata_path)
        self._write_splits(gdf)

        print("-" * 50)
        print(f"Wrote {len(gdf)} tiles to {self.metadata_path}")
        return self.metadata_path

    # -- per-zone tiling ---------------------------------------------------
    def _tile_zone(self, zone_id, sat_path):
        """Generate a full-zone mask, then emit every kept tile's COGs + rows."""
        zone_name = sat_path.stem
        with tempfile.TemporaryDirectory() as tmp:
            full_mask = Path(tmp) / f"{zone_name}_mask.tif"
            graph_stub = Path(tmp) / f"{zone_name}_graph.parquet"  # unused here
            mask_gen = RoadMaskGenerator(
                sat_cog_path=sat_path,
                roads_parquet_path=self.roads_parquet_path,
                out_mask_path=full_mask,
                out_graph_path=graph_stub,
                default_buffer_m=self.buffer_m,
            )
            mask_gen.generate_raster_mask()
            return self._cut_tiles(zone_id, zone_name, sat_path, full_mask)

    def _cut_tiles(self, zone_id, zone_name, sat_path, full_mask):
        records = []
        ts = self.tile_size
        with rasterio.open(sat_path) as img, rasterio.open(full_mask) as msk:
            n_rows = img.height // ts  # full tiles only -> edge remainder dropped
            n_cols = img.width // ts
            kept = dropped_empty = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    win = Window(c * ts, r * ts, ts, ts)
                    mask_arr = msk.read(1, window=win)
                    road_px = int(np.count_nonzero(mask_arr > 0))
                    if road_px == 0:
                        dropped_empty += 1
                        continue

                    img_arr = img.read(window=win)
                    win_tf = window_transform(win, img.transform)
                    name = f"{zone_name}_r{r}_c{c}.tif"
                    img_rel = Path("images") / name
                    msk_rel = Path("masks") / name
                    self._write_cog(self.output_dir / img_rel, img_arr, img, win_tf)
                    self._write_mask_cog(
                        self.output_dir / msk_rel, mask_arr, msk, win_tf
                    )

                    records.append(
                        self._tile_record(
                            zone_id, zone_name, r, c, img_rel, msk_rel,
                            img, win, win_tf, road_px,
                        )
                    )
                    kept += 1
            print(f"  grid {n_rows}x{n_cols}: kept {kept}, dropped {dropped_empty} "
                  f"empty, {n_cols * (img.width % ts > 0) + n_rows * (img.height % ts > 0)} "
                  "partial edge strips ignored")
        return records

    def _tile_record(self, zone_id, zone_name, r, c, img_rel, msk_rel,
                     img, win, win_tf, road_px):
        minx = win_tf.c
        maxy = win_tf.f
        maxx = minx + win.width * win_tf.a
        miny = maxy + win.height * win_tf.e
        # Tile footprint reprojected to the common CRS for the catalogue geometry.
        west, south, east, north = transform_bounds(
            img.crs, COMMON_CRS, minx, miny, maxx, maxy
        )
        total_px = win.width * win.height
        return {
            "tile_id": -1,  # filled after concat
            "zone_name": zone_name,
            "tile_row": r,
            "tile_col": c,
            "image_path": str(img_rel),
            "mask_path": str(msk_rel),
            "tile_size": self.tile_size,
            "patch_size": self.patch_size,
            "spatial_resolution": abs(win_tf.a),
            "road_pixels": road_px,
            "road_density": road_px / total_px,
            "biome": UNKNOWN_BIOME,
            "split_set": None,
            "satellite_image_dates": list(SCAFFOLD_DATES),
            "crs": img.crs.to_string(),
            "geometry": box(west, south, east, north),
        }

    # -- COG writing -------------------------------------------------------
    def _write_cog(self, path, arr, src, win_tf):
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = src.profile.copy()
        profile.update(
            height=arr.shape[1],
            width=arr.shape[2],
            transform=win_tf,
            tiled=True,
            blockxsize=self.patch_size,
            blockysize=self.patch_size,
            compress="deflate",
            predictor=3 if np.issubdtype(arr.dtype, np.floating) else 2,
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr)

    def _write_mask_cog(self, path, arr, src, win_tf):
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            height=arr.shape[0],
            width=arr.shape[1],
            transform=win_tf,
            tiled=True,
            blockxsize=self.patch_size,
            blockysize=self.patch_size,
            compress="lzw",
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    # -- augmentation ------------------------------------------------------
    def _tag_biomes(self, gdf):
        if self.biome_parquet_path is None:
            print("No --biome-parquet given; biome left as 'Unknown'.")
            return
        print(f"Tagging biomes from {self.biome_parquet_path.name}...")
        gdf["biome"] = BiomeTagger(self.biome_parquet_path).tag(gdf)
        print(gdf["biome"].value_counts(dropna=False).to_string())

    def _assign_split(self, gdf):
        """Assign whole zones to train/val/test (no tiles of one zone leak across
        splits)."""
        zones = sorted(gdf["zone_name"].unique())
        rng = np.random.default_rng(self.split_seed)
        order = rng.permutation(len(zones))
        n_test = int(round(len(zones) * self.test_frac))
        n_val = int(round(len(zones) * self.val_frac))

        split_by_zone = {}
        for rank, idx in enumerate(order):
            zone = zones[idx]
            if rank < n_test:
                split_by_zone[zone] = "test"
            elif rank < n_test + n_val:
                split_by_zone[zone] = "val"
            else:
                split_by_zone[zone] = "train"
        gdf["split_set"] = gdf["zone_name"].map(split_by_zone)

        counts = gdf["split_set"].value_counts().to_dict()
        print(f"Zone-level split (seed={self.split_seed}): {counts} tiles")

    def _write_splits(self, gdf):
        for split_name, group in gdf.groupby("split_set"):
            out = self.splits_dir / f"{split_name}.csv"
            group[SPLIT_CSV_COLUMNS].to_csv(out, index=False)
            print(f"Wrote {len(group)} tiles to {out}")
