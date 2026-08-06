"""Split the evaluation set into strata (urban / peri-urban / rural, biome, ...).

The stratum of a chip is a property of its TILE, and the split CSV already
carries it (``urbanisation_classification``, ``biome``, ...). So there are two
ways to get a per-stratum number, and they answer different questions:

  * **eval time** (``evaluate(stratum=...)``, ``benchmarking.cli eval
    --stratum``) -- score only that stratum's tiles. Use when you want a
    self-contained run row for the stratum, or to save inference on a subset.

  * **report time** (``benchmarking.cli report --stratum``) -- slice a store
    that was already scored over the whole split. Use this for a store you
    already have: the chips are per-tile, so restricting them to a stratum is
    exactly the same arithmetic as having evaluated only that stratum, with no
    re-inference. Chip-level metrics are per-chip, so subsetting is sound;
    micro aggregation re-pools counts over the subset, macro re-means over it.

Both routes go through ``tile_strata`` so they cannot disagree.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_COL = "urbanisation_classification"


def _split_csv(dataset_dir, split) -> pd.DataFrame:
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


def available(dataset_dir, split: str, col: str = DEFAULT_COL) -> list[str]:
    """Sorted distinct values of ``col`` in the split (the choosable strata)."""
    df = _split_csv(dataset_dir, split)
    if col not in df.columns:
        raise KeyError(
            f"{col!r} is not a column of splits/{split}.csv "
            f"(have: {', '.join(df.columns)})"
        )
    return sorted(df[col].dropna().astype(str).unique())


def resolve(value: str, choices: list[str]) -> str:
    """Match ``value`` to a stratum case- and separator-insensitively.

    'peri-urban', 'PeriUrban' and 'peri_urban' all reach ``PeriUrban``. An
    unmatched value raises rather than silently selecting nothing -- a typo
    that scored 0 tiles would otherwise look like a legitimate empty result.
    """
    def norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    hits = [c for c in choices if norm(c) == norm(value)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"unknown stratum {value!r}; choose one of: {', '.join(choices)}"
    )


def tile_strata(dataset_dir, split: str, col: str = DEFAULT_COL) -> dict[str, str]:
    """``{tile_id: stratum}`` for the split.

    ``tile_id`` is the image path's stem -- the same key ``runner`` derives
    chip_ids from, so this joins onto the chips table directly.
    """
    df = _split_csv(dataset_dir, split)
    if col not in df.columns:
        raise KeyError(
            f"{col!r} is not a column of splits/{split}.csv "
            f"(have: {', '.join(df.columns)})"
        )
    return {Path(p).stem: str(v) for p, v in zip(df["image_path"], df[col])}


def filter_split_df(df: pd.DataFrame, split: str, col: str, value: str) -> tuple[pd.DataFrame, str]:
    """Restrict a split DataFrame to one stratum. Returns (df, resolved value)."""
    if col not in df.columns:
        raise KeyError(
            f"{col!r} is not a column of splits/{split}.csv "
            f"(have: {', '.join(df.columns)})"
        )
    choices = sorted(df[col].dropna().astype(str).unique())
    resolved = resolve(value, choices)
    out = df[df[col].astype(str) == resolved]
    if out.empty:                      # defensive: resolve() should prevent this
        raise ValueError(f"stratum {resolved!r} selected 0 tiles of split {split!r}")
    return out, resolved


def annotate_chips(chips: pd.DataFrame, dataset_dir, split: str,
                   col: str = DEFAULT_COL) -> pd.DataFrame:
    """Add a ``stratum`` column to a chips/tiles table by tile_id lookup.

    Rows whose tile is absent from the split CSV get NaN rather than being
    dropped, so a mismatch shows up as missing data instead of vanishing.
    """
    mapping = tile_strata(dataset_dir, split, col)
    # Chips carry tile_id. The tiles table has no tile_id once the CLI's metric
    # loader renames it to chip_id -- there the chip_id IS the tile id.
    key = "tile_id" if "tile_id" in chips.columns else "chip_id"
    out = chips.copy()
    out["stratum"] = chips[key].astype(str).map(mapping)
    return out
