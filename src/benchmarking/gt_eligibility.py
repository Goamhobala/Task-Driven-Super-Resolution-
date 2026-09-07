"""Chip eligibility for the graph metrics — which units have a reference network.

APLS and clDice are defined by comparing a prediction's road graph against the
ground truth's. Where the GT graph is empty there is nothing to compare, and the
conventions in ``graph_metrics`` say so: both sides empty scores NaN, exactly one
side empty scores 0.0. Those two outcomes are the problem this module exists to
close. On a chip whose GT carries no network, an arm that predicts nothing scores
NaN and is dropped by the paired statistics, while an arm that predicts something
scores 0.0 and is kept. The chip therefore leaves any contrast involving the
silent arm and stays in every other one, so the hallucination penalty disappears
from exactly the comparisons against well-behaved arms, and n becomes a property
of which pair is being compared rather than of the split. The same asymmetry
reaches the per-model macro means, which skip NaN and so average each arm over a
different population of chips.

The fix is to decide eligibility from the GROUND TRUTH ALONE, before any model is
involved, and to score only the units that pass.

Eligibility keys on the GT *graph*, not on the GT pixel count, and the
distinction is load-bearing. A chip can carry road pixels whose skeleton is a
single sub-``min_spur_m`` spur; ``mask_to_graph`` prunes it and returns an empty
graph, reproducing the whole asymmetry on a chip that is not blank by pixel
count. Keyed on graph edges the guarantee is exact: ``apls_tile`` returns NaN
only when both directions are undefined, and the GT->prediction direction is
undefined only when the GT graph has no edges. No eligible unit can score NaN for
any arm, so n is identical across arms and across contrasts.

Because eligibility depends on nothing but the reference masks, this pass needs
no checkpoint and no inference. It walks the split's ground truth exactly as
``benchmarking.runner`` does — same footprint grid, same chip ids, same mask
reader, same stitched canvas the tile metrics see — so an existing store can be
re-aggregated under the filter without re-running a forward pass. Runs already
benched keep their numbers; only the population they are averaged over changes.

Typical use::

    benchmarking gt-eligibility --dataset-dir DATA --split test --out elig.parquet
    benchmarking report --store-dir STORE --metric apls --eligible-chips elig.parquet

New runs also emit ``gt_graph_edges`` on their own chip rows (see
``tile_metrics.apls``), which makes the lookup redundant for them; this pass
exists for stores benched before that column, and for filtering without having
run APLS at all.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from benchmarking.graph_metrics import MIN_SPUR_M, mask_to_graph

# Metric columns whose value is undefined without a GT network. The eligibility
# filter always applies to these; --eligibility-scope all extends it to the rest.
GRAPH_METRICS = ("apls", "apls_gt_to_prop", "apls_prop_to_gt", "cldice")


def is_graph_metric(metric: str) -> bool:
    """Is ``metric`` one of the graph metrics eligibility is defined for?"""
    return metric in GRAPH_METRICS


# --------------------------------------------------------------------------- #
# the GT-only pass
# --------------------------------------------------------------------------- #
def _tile_gt_canvas(dataset_dir, row, *, model, cell_m, chip_px, mask_source,
                    mask_dirname, scale):
    """One tile's stitched GT at GT resolution + its footprint grid in GT pixels.

    Mirrors ``runner._score_tile_unet`` / ``_score_tile_sr``: the canvas is
    assembled from the same windows in the same order, so a chip crop out of it
    is byte-for-byte the array the APLS plugin was handed.
    """
    from benchmarking.runner import SRMaskReader, _chip_px_from_transform, _grid

    tile_id = Path(row["image_path"]).stem
    with rasterio.open(Path(dataset_dir) / row["image_path"]) as src:
        chip_px = _chip_px_from_transform(src, cell_m, chip_px, tile_id)
        H, W = src.height, src.width
        px_m = abs(src.transform.a) / scale
        canvas = np.zeros((H * scale, W * scale), dtype="uint8")
        cells = _grid(H, W, chip_px)
        if model == "unet":
            with rasterio.open(Path(dataset_dir) / row["mask_path"]) as msrc:
                for _, _, r0, c0, h, w in cells:
                    canvas[r0:r0 + h, c0:c0 + w] = (
                        msrc.read(1, window=Window(c0, r0, w, h)) > 0)
        else:
            gt = SRMaskReader(dataset_dir, row, mask_source, mask_dirname, scale)
            for _, _, r0, c0, h, w in cells:
                m = gt.window(src, Window(c0, r0, w, h), chip_px)
                canvas[scale * r0:scale * (r0 + h),
                       scale * c0:scale * (c0 + w)] = m
    grid_gt = [(f"{tile_id}_r{ri}_c{ci}",
                scale * r0, scale * c0, scale * h, scale * w)
               for ri, ci, r0, c0, h, w in cells]
    return tile_id, canvas, grid_gt, px_m


def _tile_rows(args):
    """Eligibility rows for one tile: its chips, plus the tile itself."""
    dataset_dir, row, kw, min_spur_m = args
    tile_id, canvas, grid_gt, px_m = _tile_gt_canvas(dataset_dir, row, **kw)

    def edges(m):
        return int(mask_to_graph(m, px_m, min_spur_m).number_of_edges())

    chips = [
        {"unit": "chip", "chip_id": cid, "tile_id": tile_id,
         "gt_graph_edges": edges(canvas[r0:r0 + h, c0:c0 + w]),
         "gt_road_px": int((canvas[r0:r0 + h, c0:c0 + w] > 0).sum())}
        for cid, r0, c0, h, w in grid_gt
    ]
    # The store's tile table is renamed tile_id -> chip_id before the stats see
    # it (cli._load_metric_table), so a tile row keys on chip_id too and one
    # lookup serves both units.
    tile = {"unit": "tile", "chip_id": tile_id, "tile_id": tile_id,
            "gt_graph_edges": edges(canvas), "gt_road_px": int((canvas > 0).sum())}
    return chips, tile


def gt_eligibility(dataset_dir, split="test", *, model="sr", cell_m=None,
                   chip_px=None, mask_source=None, mask_dirname=None,
                   scale=None, min_spur_m=MIN_SPUR_M, max_tiles=None,
                   workers=0, progress=True):
    """Walk a split's ground truth -> (chip table, tile table) of GT graph sizes.

    Returns one long table, ``unit, chip_id, tile_id, gt_graph_edges,
    gt_road_px``, holding both the chip rows (``unit='chip'``) and the tile
    rows (``unit='tile'``, keyed on chip_id = tile_id so one file serves both
    granularities). A unit is eligible for the graph metrics when
    ``gt_graph_edges > 0``.

    Defaults follow ``runner.evaluate`` for the given ``model`` family, so the
    grid and mask source match a bench of the same split without restating them.
    """
    from benchmarking.runner import CELL_M_DEFAULT, _read_split_csv

    if model not in ("unet", "sr"):
        raise ValueError(f"model must be unet|sr, got {model!r}")
    cell_m = CELL_M_DEFAULT if cell_m is None else float(cell_m)
    if model == "sr":
        mask_source = mask_source or "graph"
        mask_dirname = mask_dirname or "mask_osm_2pt5"
        scale = 4 if scale is None else int(scale)
    else:
        mask_source = mask_source or "csv"
        scale = 1 if scale is None else int(scale)

    df = _read_split_csv(dataset_dir, split)
    if max_tiles is not None:
        df = df.head(int(max_tiles))
    kw = {"model": model, "cell_m": cell_m, "chip_px": chip_px,
          "mask_source": mask_source, "mask_dirname": mask_dirname, "scale": scale}
    jobs = [(str(dataset_dir), row, kw, min_spur_m) for _, row in df.iterrows()]

    chip_rows, tile_rows = [], []

    def collect(i, out):
        chips, tile = out
        chip_rows.extend(chips)
        tile_rows.append(tile)
        if progress:
            n_empty = sum(c["gt_graph_edges"] == 0 for c in chips)
            print(f"[{i + 1}/{len(jobs)}] {tile['tile_id']}: {len(chips)} chips, "
                  f"{n_empty} with no GT graph", flush=True)

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for i, out in enumerate(ex.map(_tile_rows, jobs)):
                collect(i, out)
    else:
        for i, job in enumerate(jobs):
            collect(i, _tile_rows(job))

    out = pd.DataFrame(chip_rows + tile_rows)
    if progress and chip_rows:
        n0 = sum(c["gt_graph_edges"] == 0 for c in chip_rows)
        t0 = sum(t["gt_graph_edges"] == 0 for t in tile_rows)
        print(f"\n{len(chip_rows)} chips over {len(tile_rows)} tiles | "
              f"chips ineligible {n0} ({100 * n0 / len(chip_rows):.1f}%), "
              f"eligible {len(chip_rows) - n0} | tiles ineligible {t0}")
    return out


# --------------------------------------------------------------------------- #
# applying it
# --------------------------------------------------------------------------- #
def load_eligibility(path):
    """Read a lookup written by the ``gt-eligibility`` command (parquet or csv)."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "gt_graph_edges" not in df.columns:
        raise ValueError(f"{path}: no gt_graph_edges column (not an eligibility table?)")
    return df


def eligible_ids(elig, unit="chip") -> set:
    """Ids with a non-empty GT graph, at ``chip`` or ``tile`` granularity.

    Both live in one table under a ``unit`` column, and both key on ``chip_id``
    (a tile row's chip_id is its tile_id) because the store's tile table is
    renamed the same way before the stats consume it.
    """
    if "unit" not in elig.columns:
        raise ValueError(
            "eligibility table has no 'unit' column; rebuild it with "
            "`benchmarking gt-eligibility`")
    rows = elig[(elig["unit"] == unit) & (elig["gt_graph_edges"] > 0)]
    if rows.empty and not (elig["unit"] == unit).any():
        raise ValueError(f"eligibility table holds no {unit!r} rows")
    return set(rows["chip_id"])


def apply_eligibility(df, elig, metric, *, unit="chip", scope="graph"):
    """Drop ineligible units from ``df`` -> (filtered df, note or None).

    ``scope='graph'`` filters only the graph metrics, which is the conservative
    choice: pixel metrics keep the population they have always been reported
    over. ``scope='all'`` filters every metric, which is the consistent choice —
    pixel F1 carries the identical asymmetry (a GT-empty chip is NaN for a silent
    arm and 0.0 for a hallucinating one), so leaving it unfiltered means the
    metrics in one table are averaged over different chip sets.

    Units absent from the lookup are dropped and counted in the note, since an
    id the GT pass never produced cannot be shown to be eligible.
    """
    if elig is None or (scope == "graph" and not is_graph_metric(metric)):
        return df, None
    keep = eligible_ids(elig, unit)
    before = df["chip_id"].nunique()
    out = df[df["chip_id"].isin(keep)]
    after = out["chip_id"].nunique()
    if after == before:
        return out, None
    return out, (f"eligibility: {before - after} of {before} {unit}s dropped "
                 f"(no GT graph) -> n_{unit}s={after}")
