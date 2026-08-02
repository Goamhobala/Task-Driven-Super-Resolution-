"""Once-off HR mask pre-rasterisation: masks_graph parquet -> <split>/<dirname>/{tile}.tif.

Why this exists
---------------
The on-the-fly ``mask_source="graph"`` path buffers + rasterises a tile's road
centrelines PER CROP, PER EPOCH (``upscale_dataset._graph_mask``). Under joint-SR
tuning that is ~10 crops/tile x epochs x trials of redundant CPU work, serial on
the dataloader — profiled as the GPU-starving bottleneck of the _new series
searches. The labels are frozen, so rasterisation is a once-off: run this script
one time per dataset, then train with ``mask_source="raster"`` +
``mask_dirname=<out-dirname>`` (a plain windowed COG read per crop).

Equivalence
-----------
Output is meant to be BIT-IDENTICAL to ``_graph_mask`` under any crop window:
the crop's HR grid is a sub-grid of the tile's HR grid (integer crop offsets x
integer upscale) and rasterisation is a per-pixel test, so full-tile rasterise ->
windowed read == per-crop rasterise. The ONLY permitted difference is a strict
superset of positives at crop edges: ``_graph_mask`` pre-filters candidates with
``roads.cx[window bounds]`` on the UNBUFFERED centrelines, dropping a road whose
centreline lies outside the crop but whose buffer reaches into it. Full-tile
rasterisation has no window, hence no such filter — those edge pixels are kept
(which is the more correct label). Keep the buffer/rasterize kwargs in lockstep
with ``upscale_dataset._graph_mask``; both are pinned by
``tests/test_rasterize_hr_masks equivalence`` runs done at introduction time.

No torch import — runs on a login node with only rasterio/geopandas/pyarrow.

Usage (HPC, ROSA_New)
---------------------
Standalone — no PYTHONPATH or venv activation needed; call the venv's python
on this file directly (login node is fine, no GPU):

    /scratch/$USER/InstaRoad/.venv/bin/python \
        ~/InstaRoad/InstaRoadPrototype/src/sentinel2data/dataset/rasterize_hr_masks.py \
        --dataset-dir /scratch/$USER/InstaRoad/ROSA_New \
        --out-dirname mask_new_2pt5

Then submit arms as usual: _stages_tv.sh defaults LABELS=new to
MASK_SOURCE=raster + MASK_DIRNAME=mask_new_2pt5 and fails fast if the folder is
missing. ``MASK_SOURCE=graph`` at submit time reverts to on-the-fly.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine, features

# Must match upscale_dataset._BUFFER_COL / _graph_mask semantics exactly.
_BUFFER_COL = "buffer"


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


def _out_path(dataset_dir, rel_image_path, out_dirname):
    """`<split>/imagery/x.tif` -> `<split>/<out_dirname>/x.tif` (mirrors
    joint_sr_dataset._hr_mask_path so _read_raster_hr_mask finds the file)."""
    rel = Path(rel_image_path)
    return Path(dataset_dir) / rel.parent.parent / out_dirname / rel.name


def rasterize_tile(dataset_dir, rel_image, rel_graph, out_dirname, upscale,
                   overwrite=False):
    """Buffer + rasterise ONE tile's centrelines at the upscaled transform.

    Returns (status, rel_image) where status is 'written' | 'skipped'.
    """
    dataset_dir = Path(dataset_dir)
    out = _out_path(dataset_dir, rel_image, out_dirname)
    if out.exists() and not overwrite:
        return "skipped", str(rel_image)

    with rasterio.open(dataset_dir / rel_image) as src:
        H, W = src.height, src.width
        crs = src.crs
        # Same grid as _graph_mask: crop extent, 1/upscale-sized pixels —
        # here the "crop" is the whole tile.
        up_tf = src.transform * Affine.scale(1.0 / upscale)

    out_shape = (H * upscale, W * upscale)
    roads = gpd.read_parquet(dataset_dir / rel_graph)
    if roads.empty:
        mask = np.zeros(out_shape, dtype="uint8")
    else:
        if _BUFFER_COL not in roads.columns:
            raise ValueError(
                f"masks_graph parquet {rel_graph} lacks a '{_BUFFER_COL}' column; "
                "regenerate the dataset with the current road-graph labeler."
            )
        # Keep in lockstep with upscale_dataset._graph_mask (bit-exactness).
        buffered = roads.geometry.buffer(roads[_BUFFER_COL].to_numpy(dtype="float64"))
        mask = features.rasterize(
            ((g, 1) for g in buffered), out_shape=out_shape, transform=up_tf,
            fill=0, all_touched=True, dtype="uint8",
        )
        mask = (mask > 0).astype("uint8")

    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", height=out_shape[0], width=out_shape[1], count=1,
        dtype="uint8", crs=crs, transform=up_tf,
        compress="deflate", predictor=2, tiled=True,
        blockxsize=512, blockysize=512,
    )
    # Atomic: write to .tmp then rename, so a killed run never leaves a
    # truncated .tif that a resume (skip-if-exists) would silently trust.
    tmp = out.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mask, 1)
    tmp.replace(out)
    return "written", str(rel_image)


def _worker(job):
    return rasterize_tile(*job)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Once-off pre-rasterisation of masks_graph HR labels to COGs.")
    ap.add_argument("--dataset-dir", required=True,
                    help="ROSA dataset root (contains splits/, <split>/imagery, "
                         "<split>/masks_graph).")
    ap.add_argument("--out-dirname", default="mask_new_2pt5",
                    help="Folder name under each <split>/ (default: mask_new_2pt5).")
    ap.add_argument("--upscale", type=int, default=4,
                    help="HR mask factor over native 10 m (4 -> 2.5 m).")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Split CSVs to cover. ALL of train/val/test are needed: "
                         "tune/refit read train+val, bench/test read test.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-rasterise tiles whose mask COG already exists.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel processes (rasterio/GDAL opened per process).")
    args = ap.parse_args(argv)

    jobs, seen = [], set()
    for split in args.splits:
        df = _read_split_csv(args.dataset_dir, split)
        for col in ("image_path", "mask_graph_path"):
            if col not in df.columns:
                raise SystemExit(f"splits/{split}.csv has no '{col}' column.")
        for _, row in df.iterrows():
            if row["image_path"] in seen:   # defensive: train+val overlap etc.
                continue
            seen.add(row["image_path"])
            jobs.append((args.dataset_dir, row["image_path"],
                         row["mask_graph_path"], args.out_dirname,
                         args.upscale, args.overwrite))

    print(f"[rasterize_hr_masks] {len(jobs)} unique tiles across splits "
          f"{args.splits} -> <split>/{args.out_dirname}/ "
          f"(upscale={args.upscale}, workers={args.workers})")

    n_written = n_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (status, rel) in enumerate(ex.map(_worker, jobs), 1):
            if status == "written":
                n_written += 1
            else:
                n_skipped += 1
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  (written={n_written}, "
                      f"skipped={n_skipped})", flush=True)

    print(f"[rasterize_hr_masks] done: {n_written} written, {n_skipped} "
          f"skipped (already existed).")
    if n_skipped and not args.overwrite:
        print("  NB skipped tiles were NOT validated; rerun with --overwrite "
              "if the labeler or buffer widths ever change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
