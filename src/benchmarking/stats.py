"""Stats — the post-hoc analysis surface for benchmarking results.

This module is the single import point for everything that turns the per-tile
benchmarking store into numbers you can put in a report:

    * the pixel confusion matrix (re-exported from ``confusion_matrix``) —
      ``confusion_counts`` and ``pixel_metrics_from_counts`` — so the same
      module that scores a model also compares models.
    * ``bootstrap_paired_diff`` — paired-bootstrap CI on the per-tile metric
      difference between two models.
    * ``wilcoxon_paired`` — Wilcoxon signed-rank test on that same difference.
    * ``cross_seed_ci`` — t-interval of a dataset-level metric across seeds for
      one held-fixed config (training-instability CI).

All three statistical functions consume the long-form joined table described in
``docs/benchmarking.md``: one row per ``(model_name, chip_id)`` (plus ``seed``
and the raw counts for the cross-seed micro path). They are pure — no I/O, no
global state — so they test in isolation against synthetic DataFrames.

Pairing convention (bootstrap + Wilcoxon): the two models are joined on
``chip_id``; only chips present for *both* models with a non-NaN metric on each
side survive. Resampling and the signed-rank test then operate on those paired
differences, never on the two models independently. ``chip_id`` is the unit, so
these resample chips; a parent ``tile_id`` rides along for tile-level rollups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Re-export the confusion matrix so `stats` is the one combined surface. Works
# both as a package (tests: `benchmarking.stats`) and as a sibling script
# (dummy_pipeline run from inside the benchmarking folder).
try:  # package context
    from benchmarking.confusion_matrix import (
        ConfusionCounts,
        confusion_counts,
        pixel_metrics_from_counts,
    )
except ImportError:  # flat-script context (same directory on sys.path)
    from confusion_matrix import (  # type: ignore[no-redef]
        ConfusionCounts,
        confusion_counts,
        pixel_metrics_from_counts,
    )

__all__ = [
    "ConfusionCounts",
    "confusion_counts",
    "pixel_metrics_from_counts",
    "bootstrap_paired_diff",
    "wilcoxon_paired",
    "cross_seed_ci",
]

# Pixel metrics that can be re-derived from summed (tp, fp, fn, tn) counts. Only
# these admit a "micro" (count-pooled) cross-seed aggregation; a graph metric
# like APLS has no count decomposition and must use macro.
# Metrics that admit a count-pooled ("micro") aggregation. The buffered ones
# qualify because their denominators (tp+fp, tp+fn) sit in the same row, so the
# pooled value is an exact weighted mean of the stored ratios — see
# `_micro_buffered`. APLS/clDice do NOT: they are computed on a stitched tile
# and carry no per-chip denominator to pool over.
_MICRO_DERIVABLE = ("iou", "f1", "precision", "recall", "accuracy",
                    "buffered_f1", "buffered_precision", "buffered_recall")


def _paired_values(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Join two models on ``chip_id`` and return their aligned metric arrays.

    Chips present for only one model, or NaN on either side, are dropped. Raises
    ValueError if a model is absent or has duplicate ``(model_name, chip_id)``
    rows (which usually means un-aggregated multi-seed data).

    ``chip_id`` is the unit of comparison: resampling/ranking these rows resamples
    chips. To compare at tile granularity instead, pre-aggregate the caller's
    table to one row per ``(model_name, tile_id)`` and rename it to ``chip_id``.
    """
    present = set(df["model_name"].unique())
    for m in (model_a, model_b):
        if m not in present:
            raise ValueError(
                f"model {m!r} not found in df['model_name'] (have {sorted(present)})"
            )

    subset = df[df["model_name"].isin([model_a, model_b])]
    dup = subset.duplicated(subset=["model_name", "chip_id"])
    if dup.any():
        offending = subset.loc[dup, ["model_name", "chip_id"]].to_dict("records")
        raise ValueError(
            "duplicate (model_name, chip_id) rows — expected exactly one row per "
            f"pair; aggregate multi-seed data first. Examples: {offending[:3]}"
        )

    a = df[df["model_name"] == model_a].set_index("chip_id")[metric]
    b = df[df["model_name"] == model_b].set_index("chip_id")[metric]
    paired = pd.DataFrame({"a": a, "b": b}).dropna()  # inner-aligns on chip_id
    return paired["a"].to_numpy(dtype=float), paired["b"].to_numpy(dtype=float)


def bootstrap_paired_diff(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "iou",
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
    confidence: float = 0.95,
) -> dict:
    """Paired-bootstrap CI on the per-chip metric difference ``model_a - model_b``.

    Each bootstrap iteration resamples *pairs* (chip-aligned differences) with
    replacement and takes the mean — so a constant per-chip offset collapses the
    CI to a point regardless of how the underlying values vary across chips.
    That paired invariant is what separates this from resampling the two models
    independently.

    Returns ``{"diff_mean", "ci_lo", "ci_hi", "n_pairs"}`` where ``diff_mean`` is
    the *observed* mean difference (not the bootstrap mean), and the interval is
    the percentile bootstrap CI at ``confidence``.
    """
    if rng is None:
        rng = np.random.default_rng()

    a, b = _paired_values(df, model_a, model_b, metric)
    diffs = a - b
    n = diffs.size

    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    alpha = 1.0 - confidence
    ci_lo, ci_hi = np.quantile(boot_means, [alpha / 2, 1.0 - alpha / 2])
    return {
        "diff_mean": float(diffs.mean()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_pairs": int(n),
    }


def wilcoxon_paired(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "iou",
) -> dict:
    """Wilcoxon signed-rank test on the paired per-chip difference.

    Tests whether ``model_a - model_b`` is symmetric about zero (the two-sided
    null of no consistent advantage). Returns ``{"statistic", "p_value",
    "n_pairs"}``. Pairing/NaN handling matches :func:`bootstrap_paired_diff`.
    """
    a, b = _paired_values(df, model_a, model_b, metric)
    result = scipy_stats.wilcoxon(a, b)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": int(a.size),
    }


def is_micro_derivable(metric: str) -> bool:
    """Can `metric` be pooled from counts?

    Accepts the radius-suffixed buffered columns (`buffered_f1_r3`) as well as
    the bare names. Kept as ONE function because the same question is asked in
    three places — cross_seed_ci's guard, _micro_metric_from_counts' dispatch,
    and cli._run_report's per-metric fallback — and they drifted apart once
    already, producing an empty per-model table with no error shown.
    """
    import re as _re
    if metric in _MICRO_DERIVABLE:
        return True
    return bool(_re.fullmatch(r"buffered_(?:precision|recall|f1)_r[0-9p]+", metric))


def _micro_buffered(g: pd.DataFrame, metric: str) -> float:
    """Pool the BUFFERED scores over the group's chips.

    The stored buffered_* columns are per-chip RATIOS, but their denominators
    are already in the same row, so the pooled quantity is recoverable exactly
    without re-scoring anything:

        buffered_precision_i = (pred px within rho of GT)_i / (tp + fp)_i
        buffered_recall_i    = (GT px within rho of pred)_i / (tp + fn)_i

    so micro precision is the (tp+fp)-weighted mean of the per-chip precisions,
    and micro recall the (tp+fn)-weighted mean of the recalls — i.e. counts
    pooled over all pixels, exactly as `_micro_metric_from_counts` does for the
    strict metrics. Micro F1 is then the harmonic mean of those two, NOT the
    weighted mean of the per-chip F1s (which is not a pooled quantity at all).

    Chips with a zero denominator drop out on both sides: an empty prediction
    contributes nothing to precision, and a road-free chip nothing to recall —
    the same rule the NaN convention in `buffered_metrics` encodes.
    """
    # Accept both the unsuffixed names and the radius-suffixed sweep columns
    # (buffered_f1_r3), so a tolerance sweep aggregates the same way. Anchored
    # regex, not partition("_r") — that splits buffered_RECALL at its own 'r'.
    import re as _re
    m = _re.fullmatch(r"(buffered_(?:precision|recall|f1))(_r[0-9p]+)?", metric)
    if m is None:
        raise ValueError(f"{metric!r} is not a buffered metric")
    base, sfx = m.group(1), m.group(2) or ""

    def pooled(ratio_col: str, den: pd.Series) -> float:
        r = g[ratio_col]
        keep = r.notna() & (den > 0)
        if not keep.any():
            return float("nan")
        return float((r[keep] * den[keep]).sum() / den[keep].sum())

    n_pred = g["tp"] + g["fp"]
    n_gt = g["tp"] + g["fn"]
    if base == "buffered_precision":
        return pooled(f"buffered_precision{sfx}", n_pred)
    if base == "buffered_recall":
        return pooled(f"buffered_recall{sfx}", n_gt)
    if base == "buffered_f1":
        prec = pooled(f"buffered_precision{sfx}", n_pred)
        rec = pooled(f"buffered_recall{sfx}", n_gt)
        if not (prec == prec) or not (rec == rec) or (prec + rec) == 0:
            return float("nan") if (prec != prec or rec != rec) else 0.0
        return 2.0 * prec * rec / (prec + rec)
    raise ValueError(f"{metric!r} is not a buffered metric")


def _micro_metric_from_counts(g: pd.DataFrame, metric: str) -> float:
    """Pool tp/fp/fn/tn over the group's tiles, then derive one scalar metric."""
    if metric.startswith("buffered_"):
        return _micro_buffered(g, metric)
    tp = float(g["tp"].sum())
    fp = float(g["fp"].sum())
    fn = float(g["fn"].sum())
    tn = float(g["tn"].sum())
    if metric == "iou":
        den = tp + fp + fn
    elif metric == "f1":
        return float("nan") if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    elif metric == "precision":
        den = tp + fp
    elif metric == "recall":
        den = tp + fn
    elif metric == "accuracy":
        return float("nan") if (tp + fp + fn + tn) == 0 else (tp + tn) / (tp + fp + fn + tn)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"metric {metric!r} is not count-derivable")
    return float("nan") if den == 0 else tp / den


def cross_seed_ci(
    df: pd.DataFrame,
    config_filters: dict,
    metric: str = "iou",
    aggregation: str = "macro",
    confidence: float = 0.95,
) -> dict:
    """t-interval of a dataset-level metric across seeds, for one fixed config.

    ``config_filters`` selects the rows of one configuration (every
    ``(column, value)`` must match, e.g. ``{"model_name": "A", "loss_fn":
    "focal"}``). Each seed's per-tile metrics are reduced to a single scalar:

        * ``macro`` — mean of the per-tile ``metric`` values.
        * ``micro`` — pool tp/fp/fn/tn across the seed's tiles, then derive the
          metric. Only valid for the count-derivable pixel metrics.

    The spread of those per-seed scalars quantifies training instability. Returns
    ``{"mean", "std", "ci_lo", "ci_hi", "n_seeds", "per_seed_values"}``. With a
    single seed, ``std`` and the CI are NaN (undefined, not zero).
    """
    if aggregation == "micro" and not is_micro_derivable(metric):
        raise ValueError(
            f"micro aggregation requires a count-derivable metric "
            f"{_MICRO_DERIVABLE}; got {metric!r} — use aggregation='macro'"
        )

    mask = pd.Series(True, index=df.index)
    for col, val in config_filters.items():
        mask &= df[col] == val
    sub = df[mask]
    if sub.empty:
        raise ValueError(f"no rows match config_filters {config_filters}")

    per_seed_values: list[float] = []
    for _seed, g in sub.groupby("seed"):
        if aggregation == "macro":
            per_seed_values.append(float(g[metric].mean()))
        elif aggregation == "micro":
            per_seed_values.append(_micro_metric_from_counts(g, metric))
        else:
            raise ValueError(f"unknown aggregation {aggregation!r} (macro|micro)")

    values = np.asarray(per_seed_values, dtype=float)
    n_seeds = values.size
    mean = float(values.mean())

    if n_seeds < 2:
        return {
            "mean": mean,
            "std": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_seeds": int(n_seeds),
            "per_seed_values": per_seed_values,
        }

    std = float(values.std(ddof=1))
    se = std / np.sqrt(n_seeds)
    t_crit = float(scipy_stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n_seeds - 1))
    half = t_crit * se
    return {
        "mean": mean,
        "std": std,
        "ci_lo": mean - half,
        "ci_hi": mean + half,
        "n_seeds": int(n_seeds),
        "per_seed_values": per_seed_values,
    }
