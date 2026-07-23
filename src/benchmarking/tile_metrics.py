"""Tile-level metric plugins — the seam graph metrics (APLS) drop into.

The runner stitches each tile's binary prediction and ground truth at GT
resolution and hands them to every requested plugin. A plugin returns
tile-level values (one row per tile -> ``tiles/<run_id>.parquet``) and,
optionally, per-chip values that the runner merges onto the chip rows
(nullable columns, NaN where undefined — the stats drop NaN pairs).

Contract::

    @register("apls")
    def apls(pred_bin, gt_mask, *, transform, tile_id, grid) -> TileMetricResult

    pred_bin   (H, W) bool     stitched thresholded prediction, GT resolution
    gt_mask    (H, W) uint8    stitched ground truth, same grid
    transform  affine.Affine   GT-resolution geotransform (pixel -> CRS metres),
                               so graph extraction can work in ground units
    tile_id    str             parent image stem
    grid       list of (chip_id, ri, ci, r0, c0, h, w) IN GT PIXELS — the
               footprint cells, for per-chip values that pair with the
               pixel-metric rows

Keep plugins pure (no I/O, no store access): the runner owns orchestration and
persistence, exactly like the confusion-matrix path.

``road_frac`` is a deliberately trivial reference plugin: it proves the
plumbing (tiles shard + per-chip merge) end-to-end and is the template the
custom APLS implementation follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class TileMetricResult:
    """What a plugin hands back for ONE tile."""

    tile: dict[str, float]                              # -> tiles/<run_id>.parquet
    chips: dict[str, dict[str, float]] | None = field(default=None)  # chip_id -> cols


TILE_METRICS: dict[str, Callable[..., TileMetricResult]] = {}


def register(name: str):
    """Register a tile-metric plugin under ``name`` (CLI: ``--tile-metric name``)."""

    def deco(fn):
        if name in TILE_METRICS:
            raise ValueError(f"tile metric {name!r} already registered")
        TILE_METRICS[name] = fn
        return fn

    return deco


def resolve_tile_metrics(names: Sequence[str]):
    """Names -> plugin callables; unknown names fail loudly with the menu."""
    try:
        return [TILE_METRICS[n] for n in names]
    except KeyError as e:
        raise ValueError(
            f"unknown tile metric {e.args[0]!r}; available: {sorted(TILE_METRICS) or '(none)'}"
        ) from None


@register("apls")
def apls(pred_bin: np.ndarray, gt_mask: np.ndarray, *, transform, tile_id: str,
         grid) -> TileMetricResult:
    """APLS (Van Etten et al. 2019) between the skeleton graphs of the stitched
    prediction and GT — the protocol's connectivity metric. Emits BOTH levels
    (the ``road_frac`` convention: same column name at each level):

    * per-chip ``apls`` — merged onto the chip rows, so the paired bootstrap /
      Wilcoxon resample the SAME ``chip_id`` unit as the pixel metrics (chips
      resolve first in ``benchmarking.cli``). Each chip is skeletonized and
      scored independently; SpaceNet precedent computed APLS on 400 m cells,
      so a 2560 m chip is comfortably large enough.
    * tile-level ``apls`` (+ the two directional scores and graph sizes) ->
      ``tiles/`` shard. Chip APLS sees within-chip connectivity only — paths
      crossing a chip border are never sampled — so the tile row is the
      longer-range routing measure, plus the convenient per-tile rollup.

    NaN where a unit has no roads on either side (the stats drop NaN pairs).
    Only the pixel size is read from ``transform``, so passing the tile
    transform for chip crops is exact. Algorithm + GSD-tuned defaults live in
    ``benchmarking.graph_metrics`` (imported lazily: networkx/scipy stay off
    the runner's import path unless APLS is requested)."""
    from benchmarking.graph_metrics import apls_tile

    tile = apls_tile(pred_bin, gt_mask, transform=transform)
    chips = {
        chip_id: {"apls": apls_tile(pred_bin[r0:r0 + h, c0:c0 + w],
                                    gt_mask[r0:r0 + h, c0:c0 + w],
                                    transform=transform)["apls"]}
        for chip_id, ri, ci, r0, c0, h, w in (grid or [])
    }
    return TileMetricResult(tile=tile, chips=chips or None)


@register("cldice")
def cldice(pred_bin: np.ndarray, gt_mask: np.ndarray, *, transform, tile_id: str,
           grid) -> TileMetricResult:
    """clDice metric (hard-skeleton, official jocpae/clDice port — see
    ``benchmarking.skeleton_metrics``): the protocol composite's second
    connectivity number, cheaper than APLS and sensitive to centreline
    coverage rather than routing. Emitted at BOTH levels (`road_frac`
    convention): per-chip ``cldice`` merges onto the chip rows (paired stats
    on ``chip_id``, same unit as pixel metrics/APLS) and a tile-level rollup
    lands in ``tiles/``. NaN where both masks are road-free; 0.0 when exactly
    one side is empty."""
    from benchmarking.skeleton_metrics import cldice_score

    tile = {"cldice": cldice_score(pred_bin, gt_mask > 0)}
    chips = {
        chip_id: {"cldice": cldice_score(pred_bin[r0:r0 + h, c0:c0 + w],
                                         gt_mask[r0:r0 + h, c0:c0 + w] > 0)}
        for chip_id, ri, ci, r0, c0, h, w in (grid or [])
    }
    return TileMetricResult(tile=tile, chips=chips or None)


@register("road_frac")
def road_frac(pred_bin: np.ndarray, gt_mask: np.ndarray, *, transform, tile_id: str,
              grid) -> TileMetricResult:
    """Reference plugin: predicted / GT road-pixel fraction, tile + chip level."""
    tile = {
        "pred_road_frac": float(pred_bin.mean()),
        "gt_road_frac": float((gt_mask > 0).mean()),
    }
    chips = {
        chip_id: {
            "pred_road_frac": float(pred_bin[r0:r0 + h, c0:c0 + w].mean()),
            "gt_road_frac": float((gt_mask[r0:r0 + h, c0:c0 + w] > 0).mean()),
        }
        for chip_id, ri, ci, r0, c0, h, w in grid
    }
    return TileMetricResult(tile=tile, chips=chips)
