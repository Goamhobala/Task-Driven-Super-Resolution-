"""Sharded parquet store for benchmarking results.

Layout — one shard per run, so concurrent SLURM jobs sharing a ``STORE_DIR``
never rewrite each other's data (the old read-concat-rewrite append lost rows
under parallel writers):

    store_dir/
      runs/<run_id>.parquet     one row: run metadata
      chips/<run_id>.parquet    one row per (run_id, chip_id): pixel metrics
      tiles/<run_id>.parquet    one row per (run_id, tile_id): tile-level
                                metrics from plugins (e.g. APLS)

Shards are written to a temp file then ``os.replace``d into place — atomic on
a single filesystem (Lustre included). A run_id that already has a shard is an
error: the store is append-only and reruns are new runs with new IDs.

Legacy flat files (``runs.parquet`` / ``chip_metrics.parquet`` /
``tile_metrics.parquet``) are still READ by the loaders for back-compat, but
never written.
"""
from pathlib import Path
import os
import uuid

import pandas as pd

RUNS_SUBDIR = "runs"
CHIPS_SUBDIR = "chips"
TILES_SUBDIR = "tiles"

_LEGACY_FILE = {
    RUNS_SUBDIR: "runs.parquet",
    CHIPS_SUBDIR: "chip_metrics.parquet",
    TILES_SUBDIR: "tile_metrics.parquet",
}


def _write_shard(df: pd.DataFrame, store_dir, subdir: str, run_id: str) -> Path:
    """Atomically write one run's shard. Refuses to overwrite (append-only)."""
    out_dir = Path(store_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{run_id}.parquet"
    if final.exists():
        raise FileExistsError(
            f"shard already exists (store is append-only; rerun = new run_id): {final}"
        )
    tmp = out_dir / f".{run_id}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, final)  # atomic on one filesystem
    finally:
        if tmp.exists():
            tmp.unlink()
    return final


def _require_single_run_id(df: pd.DataFrame) -> str:
    ids = df["run_id"].unique()
    if len(ids) != 1:
        raise ValueError(f"one shard per run: expected a single run_id, got {list(ids)}")
    return str(ids[0])


def append_run(run_row: dict, store_dir) -> Path:
    """Write the run-metadata shard (one row)."""
    df = pd.DataFrame([run_row])
    return _write_shard(df, store_dir, RUNS_SUBDIR, str(run_row["run_id"]))


def append_chips(chips: pd.DataFrame, store_dir) -> Path:
    """Write one run's per-chip metric shard."""
    return _write_shard(chips, store_dir, CHIPS_SUBDIR, _require_single_run_id(chips))


def append_tiles(tiles: pd.DataFrame, store_dir) -> Path:
    """Write one run's per-tile metric shard (plugin metrics, e.g. APLS)."""
    return _write_shard(tiles, store_dir, TILES_SUBDIR, _require_single_run_id(tiles))


def _load(store_dir, subdir: str) -> pd.DataFrame:
    """Concat all shards under ``subdir`` plus the legacy flat file, if any."""
    store_dir = Path(store_dir)
    paths = sorted((store_dir / subdir).glob("*.parquet"))
    legacy = store_dir / _LEGACY_FILE[subdir]
    if legacy.exists():
        paths.insert(0, legacy)
    if not paths:
        raise FileNotFoundError(
            f"no benchmark data: neither {store_dir / subdir}/*.parquet nor {legacy}"
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_runs(store_dir) -> pd.DataFrame:
    return _load(store_dir, RUNS_SUBDIR)


def load_chips(store_dir) -> pd.DataFrame:
    """The long-form per-chip table the stats functions consume directly."""
    return _load(store_dir, CHIPS_SUBDIR)


def load_tiles(store_dir) -> pd.DataFrame:
    """The long-form per-tile table (graph / tile-level metrics)."""
    return _load(store_dir, TILES_SUBDIR)


def load_joined(store_dir) -> pd.DataFrame:
    """``chip_metrics`` joined with its run context (for run-level columns like
    ``config_hash`` / ``dataset_split``). Shared columns kept from the chip side."""
    chips = load_chips(store_dir)
    runs = load_runs(store_dir)
    shared = [c for c in ("model_name", "seed") if c in chips.columns and c in runs.columns]
    runs = runs.drop(columns=shared) if shared else runs
    return runs.merge(chips, on="run_id")
