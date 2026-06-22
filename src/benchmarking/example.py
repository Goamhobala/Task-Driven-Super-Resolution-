"""Worked example: comparing two models from the benchmarking parquet.

The pilot question is "does dummy_a differ from dummy_b, and how stable is each
across seeds?" The analysis has three steps:

  1. Collapse the multi-seed runs to ONE value per (model, chip) by averaging the
     per-chip F1 across seeds. This is also the pre-aggregation the paired tests
     require: they expect exactly one row per (model_name, chip_id), so the
     multiple seeds have to be reduced first.

  2. Paired bootstrap CI + Wilcoxon signed-rank test on those seed-averaged F1s.
     Both pair on chip_id, so they ask whether the two models differ on the SAME
     chips rather than on aggregate.

  3. Per-chip mean and std of F1 across seeds -> training stability. A large
     per-chip std means a chip's score swings a lot when only the seed changes,
     i.e. training is unstable there.

Generate the input first by running dummy_pipeline.py a few times per model at
DIFFERENT seeds (the seeds must actually differ, or there is nothing to average
over and stability is trivially zero):

    for m in dummy_a dummy_b; do
      for s in 0 1 2; do
        BENCH_MODEL=$m BENCH_SEED=$s python dummy_pipeline.py
      done
    done
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from stats import bootstrap_paired_diff, wilcoxon_paired
PARQUET = Path(__file__).parent / "dummy_data"/"tile_metrics_dummy.parquet"

METRIC = "f1"
MODEL_A, MODEL_B = "dummy_a", "dummy_b"


def per_chip_seed_mean(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average the metric across seeds -> one row per (model_name, chip_id).

    This is the long-form table the paired tests consume: each chip appears once
    per model, carrying that model's seed-averaged score on the chip.
    """
    return df.groupby(["model_name", "chip_id"], as_index=False)[metric].mean()


def training_stability(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-chip mean/std of the metric across seeds, plus a per-model summary.

    `seed_std` is the spread of one chip's score across seeds. Averaging it over
    chips gives a single stability number per model: lower means the model lands
    in the same place regardless of the seed.
    """
    per_chip = (
        df.groupby(["model_name", "chip_id"])[metric]
        .agg(seed_mean="mean", seed_std="std")
        .reset_index()
    )
    summary = per_chip.groupby("model_name").agg(
        mean_f1=("seed_mean", "mean"),
        mean_chip_std=("seed_std", "mean"),
        max_chip_std=("seed_std", "max"),
    )
    return per_chip, summary


def main() -> None:
    df = pd.read_parquet(PARQUET)
    n_seeds = df.groupby("model_name")["seed"].nunique()
    print(f"loaded {len(df)} rows from {PARQUET.name}")
    print("seeds per model:")
    print(n_seeds.to_string(), "\n")

    # 1. average per-chip F1 across seeds -> one row per (model, chip)
    avg = per_chip_seed_mean(df, METRIC)

    # Make the bootstrap UNIT explicit: we resample chips, not tiles. With one
    # image these chips all share a tile_id; a real dataset spreads them across
    # many tiles and you would resample over tile_id instead.
    n_chips = avg["chip_id"].nunique()
    n_tiles = df["tile_id"].nunique()
    print(f"pairing on {n_chips} chips across {n_tiles} tile(s) "
          f"(chip is the bootstrap unit, tile_id is the parent image)\n")

    # 2. paired comparison on the seed-averaged per-chip F1
    boot = bootstrap_paired_diff(
        avg, MODEL_A, MODEL_B, metric=METRIC, n_boot=2000, rng=np.random.default_rng(0)
    )
    wil = wilcoxon_paired(avg, MODEL_A, MODEL_B, metric=METRIC)

    print(f"{MODEL_A} vs {MODEL_B} on seed-averaged per-chip {METRIC}:")
    print(
        f"  bootstrap  diff_mean = {boot['diff_mean']:+.4f}  "
        f"95% CI [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]  n_pairs = {boot['n_pairs']}"
    )
    print(
        f"  wilcoxon   p = {wil['p_value']:.4g}  "
        f"(W = {wil['statistic']:.0f}, n = {wil['n_pairs']})"
    )
    verdict = "significant" if wil["p_value"] < 0.05 else "not significant"
    print(f"  -> difference is {verdict} at alpha = 0.05\n")

    # 3. training stability: per-chip mean/std across seeds
    _, summary = training_stability(df, METRIC)
    print(f"training stability (per-chip {METRIC} across seeds):")
    print(summary.round(4).to_string())
    print("\n  mean_chip_std is the average per-chip spread across seeds; lower = more stable.")


if __name__ == "__main__":
    main()
