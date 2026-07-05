"""Parquet store for benchmarking results
  * ``runs.parquet``         -- per patch evaluated checkpoint (run metadata).
  * ``chip_metrics.parquet`` -- one row per ``(run_id, chip_id)`` (per-chip metrics),
    with ``model_name`` + ``seed`` denormalised so it is self-contained for the
    stats functions without joining ``runs``.

Parquet has no in-place append: each writer reads the existing table (if any),
concatenates, and rewrites. Fine at benchmarking scale.
"""
from pathlib import Path
import pandas as pd

RUNS_FILE = "runs.parquet"
CHIPS_FILE = "chip_metrics.parquet"


def _append(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def append_run(run_row: dict, store_dir) -> Path:
    """Append one run-metadata row to ``runs.parquet``."""
    return _append(pd.DataFrame([run_row]), Path(store_dir) / RUNS_FILE)


def append_chips(chips: pd.DataFrame, store_dir) -> Path:
    """Append per-chip metric rows to ``chip_metrics.parquet``."""
    return _append(chips, Path(store_dir) / CHIPS_FILE)


def load_chips(store_dir) -> pd.DataFrame:
    """The long-form per-chip table the stats functions consume directly."""
    return pd.read_parquet(Path(store_dir) / CHIPS_FILE)


def load_runs(store_dir) -> pd.DataFrame:
    return pd.read_parquet(Path(store_dir) / RUNS_FILE)


def load_joined(store_dir) -> pd.DataFrame:
    """``chip_metrics`` joined with its run context (for run-level columns like
    ``config_hash`` / ``dataset_split``). Shared columns kept from the chip side."""
    chips = load_chips(store_dir)
    runs = load_runs(store_dir)
    shared = [c for c in ("model_name", "seed") if c in chips.columns and c in runs.columns]
    runs = runs.drop(columns=shared) if shared else runs
    return runs.merge(chips, on="run_id")
