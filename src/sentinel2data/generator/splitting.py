"""Train/val/test split strategies (fill the catalogue's ``split_set`` column).

A :class:`SplitStrategy` assigns whole units to a split so no unit leaks across
sets:
  * :class:`RandomTileSplit`  -- random holdout of whole tiles (``tile_id``).
  * :class:`ZoneHoldoutSplit` -- random holdout of whole zones (``zone_name``),
    so all tiles of one source COG share a split.
"""
from typing import Protocol, runtime_checkable
import geopandas as gpd
import numpy as np


@runtime_checkable
class SplitStrategy(Protocol):
    """Assign ``gdf['split_set']`` in place."""

    def assign(self, gdf: gpd.GeoDataFrame) -> None:
        ...


def _holdout_map(keys, val_frac, test_frac, seed):
    """Map each unique key -> 'train'/'val'/'test' by a seeded permutation."""
    keys = list(keys)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    n_test = int(round(len(keys) * test_frac))
    n_val = int(round(len(keys) * val_frac))

    assignment = {}
    for rank, idx in enumerate(order):
        key = keys[idx]
        if rank < n_test:
            assignment[key] = "test"
        elif rank < n_test + n_val:
            assignment[key] = "val"
        else:
            assignment[key] = "train"
    return assignment


class RandomTileSplit:
    """Random holdout over whole tiles (``tile_id``)."""

    def __init__(self, val_frac=0.1, test_frac=0.1, seed=42):
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def assign(self, gdf) -> None:
        if gdf.empty:
            return
        tile_ids = sorted(gdf["tile_id"].unique())
        by_tile = _holdout_map(tile_ids, self.val_frac, self.test_frac, self.seed)
        gdf["split_set"] = gdf["tile_id"].map(by_tile)

        n_tiles = {s: 0 for s in ("train", "val", "test")}
        for s in by_tile.values():
            n_tiles[s] += 1
        print(
            f"Tile-level split (seed={self.seed}): "
            f"{n_tiles['train']} train / {n_tiles['val']} val / {n_tiles['test']} test tiles"
        )


class ZoneHoldoutSplit:
    """Random holdout over whole zones (``zone_name``)."""

    def __init__(self, val_frac=0.1, test_frac=0.1, seed=42):
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def assign(self, gdf) -> None:
        if gdf.empty:
            return
        zones = sorted(gdf["zone_name"].unique())
        by_zone = _holdout_map(zones, self.val_frac, self.test_frac, self.seed)
        gdf["split_set"] = gdf["zone_name"].map(by_zone)

        counts = gdf["split_set"].value_counts().to_dict()
        print(f"Zone-level split (seed={self.seed}): {counts} tiles")
