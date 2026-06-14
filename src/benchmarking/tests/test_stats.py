"""Tests for benchmarking.stats — paired bootstrap, Wilcoxon, cross-seed CI.

Contracts under test:

    bootstrap_paired_diff(df, model_a, model_b, metric="iou", n_boot=1000, rng)
        Returns {"diff_mean", "ci_lo", "ci_hi", "n_pairs"}.
        Pairs are constructed by joining on chip_id. Each bootstrap iteration
        resamples PAIRS (not the two models independently), which is the
        invariant the killer test below pins down.
        
        The point of this function is to get us the confidence interval for the
        difference between the two models on the given metric.

    wilcoxon_paired(df, model_a, model_b, metric="iou")
        Returns {"statistic", "p_value", "n_pairs"}.
        Wraps scipy.stats.wilcoxon on the paired difference.
        
        The point of this function is to test if the difference between the two
        models is statistically significant (to get us a p-value).

    cross_seed_ci(df, config_filters, metric="iou", aggregation="macro",
                  confidence=0.95)
        Returns {"mean", "std", "ci_lo", "ci_hi", "n_seeds", "per_seed_values"}.
        Filters df to rows matching every (column, value) in config_filters,
        reduces each seed's tile_metrics to a scalar (macro = mean of per-tile
        metric; micro = aggregate tp/fp/fn/tn then derive), and returns the
        t-interval across seeds. NaN std/CI when n_seeds < 2.

        The point of this function is to quantify per-config training
        instability — how much does the metric vary across seeds when the
        configuration is held fixed.

The first two functions expect long-form input with columns at least:
    ("model_name", "chip_id", <metric>)
and exactly one row per (model_name, chip_id). Multi-seed data must be
pre-aggregated by the caller for those two.
cross_seed_ci additionally requires a "seed" column (and tp/fp/fn/tn for micro).
"""

import math

import numpy as np
import pandas as pd
import pytest

from benchmarking.stats import (
    bootstrap_paired_diff,
    cross_seed_ci,
    wilcoxon_paired,
)


def make_long_df(per_model: dict[str, list[float]], chip_ids=None, metric="iou") -> pd.DataFrame:
    """Build a long-form (model_name, chip_id, <metric>) DataFrame."""
    if chip_ids is None:
        n = len(next(iter(per_model.values())))
        chip_ids = [f"chip_{i:03d}" for i in range(n)]
    rows = []
    for model, values in per_model.items():
        for cid, v in zip(chip_ids, values):
            rows.append({"model_name": model, "chip_id": cid, metric: v})
    return pd.DataFrame(rows)


def make_seed_df(
    per_seed: dict[int, list[float]],
    model_name: str = "A",
    tile_ids=None,
    metric: str = "iou",
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Build long-form (model_name, seed, tile_id, <metric>) rows."""
    if tile_ids is None:
        n = len(next(iter(per_seed.values())))
        tile_ids = [f"tile_{i:03d}" for i in range(n)]
    rows = []
    for seed, values in per_seed.items():
        for tid, v in zip(tile_ids, values):
            row = {"model_name": model_name, "seed": seed, "tile_id": tid, metric: v}
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    return pd.DataFrame(rows)


class TestBootstrapPairedDiff:
    def test_constant_offset_collapses_ci_to_a_point(self):
        """KILLER TEST: differences are constant, but underlying values vary.

        Paired resampling: every resampled pair has difference 0.05, so every
        bootstrap mean is 0.05 -> CI collapses.

        Independent resampling: mean(A_sample) and mean(B_sample) each vary
        across resamples (because the values vary across tiles), so their
        difference varies -> CI width on the order of std(base)/sqrt(n) ~ 0.02.
        That is ~10 orders of magnitude wider than the tolerance below, so the
        broken implementation fails loudly here.
        """
        base = np.linspace(0.30, 0.80, 50)  # varies across tiles — this is the point
        a = base + 0.05
        b = base
        df = make_long_df({"A": list(a), "B": list(b)})
        rng = np.random.default_rng(0)

        out = bootstrap_paired_diff(df, "A", "B", n_boot=500, rng=rng)

        expected = float((a - b).mean())
        assert out["diff_mean"] == pytest.approx(expected, abs=1e-12)
        assert out["ci_lo"] == pytest.approx(expected, abs=1e-9)
        assert out["ci_hi"] == pytest.approx(expected, abs=1e-9)
        assert out["n_pairs"] == 50

    def test_returns_observed_paired_mean(self):
        """diff_mean is the sample mean of (A - B), not the bootstrap mean."""
        a = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        b = np.array([0.40, 0.55, 0.65, 0.75, 0.80])
        df = make_long_df({"A": list(a), "B": list(b)})

        out = bootstrap_paired_diff(df, "A", "B", n_boot=200, rng=np.random.default_rng(0))

        assert out["diff_mean"] == pytest.approx((a - b).mean())

    def test_observed_diff_inside_ci(self):
        """Sanity: the percentile CI contains the observed sample mean."""
        rng = np.random.default_rng(0)
        n = 200
        a = rng.beta(3, 2, n)
        b = a - 0.10 + rng.normal(0, 0.05, n)
        df = make_long_df({"A": list(a), "B": list(b)})

        out = bootstrap_paired_diff(df, "A", "B", n_boot=1000, rng=rng)

        observed = float((a - b).mean())
        assert out["ci_lo"] <= observed <= out["ci_hi"]
        # CI should be reasonably tight: SE ~ 0.05/sqrt(200) ~ 0.0035, 95% width ~ 0.014.
        assert (out["ci_hi"] - out["ci_lo"]) < 0.05

    def test_returns_expected_keys(self):
        df = make_long_df({"A": [0.5, 0.6, 0.7], "B": [0.4, 0.5, 0.6]})
        out = bootstrap_paired_diff(df, "A", "B", n_boot=10, rng=np.random.default_rng(0))
        assert set(out.keys()) == {"diff_mean", "ci_lo", "ci_hi", "n_pairs"}

    def test_seed_determinism(self):
        df = make_long_df({
            "A": list(np.linspace(0.4, 0.9, 20)),
            "B": list(np.linspace(0.3, 0.8, 20)),
        })
        out_1 = bootstrap_paired_diff(df, "A", "B", n_boot=200, rng=np.random.default_rng(42))
        out_2 = bootstrap_paired_diff(df, "A", "B", n_boot=200, rng=np.random.default_rng(42))
        assert out_1 == out_2

    def test_drops_unpaired_tiles(self):
        """A tile present for one model only is excluded from the analysis."""
        df = pd.DataFrame([
            {"model_name": "A", "chip_id": "t1", "iou": 0.5},
            {"model_name": "A", "chip_id": "t2", "iou": 0.6},
            {"model_name": "A", "chip_id": "t3", "iou": 0.7},  # no B counterpart
            {"model_name": "B", "chip_id": "t1", "iou": 0.4},
            {"model_name": "B", "chip_id": "t2", "iou": 0.5},
        ])
        out = bootstrap_paired_diff(df, "A", "B", n_boot=10, rng=np.random.default_rng(0))
        assert out["n_pairs"] == 2

    def test_drops_nan_pairs(self):
        """A pair with NaN on either side is dropped before resampling."""
        df = pd.DataFrame([
            {"model_name": "A", "chip_id": "t1", "iou": 0.5},
            {"model_name": "A", "chip_id": "t2", "iou": float("nan")},
            {"model_name": "A", "chip_id": "t3", "iou": 0.7},
            {"model_name": "B", "chip_id": "t1", "iou": 0.4},
            {"model_name": "B", "chip_id": "t2", "iou": 0.5},
            {"model_name": "B", "chip_id": "t3", "iou": 0.6},
        ])
        out = bootstrap_paired_diff(df, "A", "B", n_boot=10, rng=np.random.default_rng(0))
        assert out["n_pairs"] == 2

    def test_metric_parameter_selects_column(self):
        df = pd.DataFrame([
            {"model_name": "A", "chip_id": "t1", "iou": 0.5, "f1": 0.7},
            {"model_name": "A", "chip_id": "t2", "iou": 0.6, "f1": 0.8},
            {"model_name": "B", "chip_id": "t1", "iou": 0.4, "f1": 0.6},
            {"model_name": "B", "chip_id": "t2", "iou": 0.5, "f1": 0.7},
        ])
        out_iou = bootstrap_paired_diff(df, "A", "B", metric="iou", n_boot=10, rng=np.random.default_rng(0))
        out_f1 = bootstrap_paired_diff(df, "A", "B", metric="f1", n_boot=10, rng=np.random.default_rng(0))
        assert out_iou["diff_mean"] == pytest.approx(0.1)
        assert out_f1["diff_mean"] == pytest.approx(0.1)

    def test_raises_on_unknown_model(self):
        df = make_long_df({"A": [0.5, 0.6], "B": [0.4, 0.5]})
        with pytest.raises(ValueError, match="C"):
            bootstrap_paired_diff(df, "A", "C", n_boot=10, rng=np.random.default_rng(0))

    def test_raises_on_duplicate_model_tile_rows(self):
        """Contract: one row per (model_name, chip_id). Duplicates likely indicate
        multi-seed data that should be aggregated by the caller first."""
        df = pd.DataFrame([
            {"model_name": "A", "chip_id": "t1", "iou": 0.5},
            {"model_name": "A", "chip_id": "t1", "iou": 0.6},
            {"model_name": "B", "chip_id": "t1", "iou": 0.4},
        ])
        with pytest.raises(ValueError, match="duplicate"):
            bootstrap_paired_diff(df, "A", "B", n_boot=10, rng=np.random.default_rng(0))


class TestWilcoxonPaired:
    def test_strong_dominance_low_pvalue(self):
        """A consistently beats B by 0.1 IoU — the test should detect it."""
        base = np.linspace(0.30, 0.90, 30)
        df = make_long_df({"A": list(base + 0.10), "B": list(base)})

        out = wilcoxon_paired(df, "A", "B")

        assert out["p_value"] < 0.01
        assert out["n_pairs"] == 30

    def test_no_dominance_high_pvalue(self):
        """When differences are symmetric around zero, no signal."""
        rng = np.random.default_rng(0)
        n = 50
        base = rng.uniform(0.30, 0.90, n)
        diffs = rng.normal(0, 0.10, n)  # symmetric around 0
        df = make_long_df({"A": list(base + diffs / 2), "B": list(base - diffs / 2)})

        out = wilcoxon_paired(df, "A", "B")

        assert out["p_value"] > 0.05
        assert out["n_pairs"] == 50

    def test_returns_expected_keys(self):
        df = make_long_df({
            "A": [0.6, 0.7, 0.8, 0.9, 1.0, 0.5, 0.55, 0.65, 0.75, 0.85],
            "B": [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.45, 0.55, 0.65, 0.75],
        })
        out = wilcoxon_paired(df, "A", "B")
        assert set(out.keys()) == {"statistic", "p_value", "n_pairs"}

    def test_drops_nan_pairs(self):
        """NaN pairs are dropped before scipy is invoked."""
        a_vals = [0.6, 0.7, 0.8, 0.9, 1.0, float("nan"), 0.55, 0.65, 0.75, 0.85]
        b_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4,           0.45, 0.55, 0.65, 0.75]
        df = make_long_df({"A": a_vals, "B": b_vals})

        out = wilcoxon_paired(df, "A", "B")

        assert out["n_pairs"] == 9  # one pair dropped

    def test_drops_unpaired_tiles(self):
        df = pd.DataFrame([
            {"model_name": "A", "chip_id": f"t{i}", "iou": 0.5 + i * 0.02} for i in range(10)
        ] + [
            {"model_name": "A", "chip_id": "t_extra", "iou": 0.99},  # no B counterpart
        ] + [
            {"model_name": "B", "chip_id": f"t{i}", "iou": 0.4 + i * 0.02} for i in range(10)
        ])
        out = wilcoxon_paired(df, "A", "B")
        assert out["n_pairs"] == 10

    def test_raises_on_unknown_model(self):
        df = make_long_df({
            "A": [0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65],
            "B": [0.4, 0.5, 0.6, 0.7, 0.8, 0.45, 0.55],
        })
        with pytest.raises(ValueError, match="C"):
            wilcoxon_paired(df, "A", "C")


class TestCrossSeedCI:
    def test_returns_expected_keys(self):
        df = make_seed_df({0: [0.5], 1: [0.6], 2: [0.7]})
        out = cross_seed_ci(df, config_filters={"model_name": "A"})
        assert set(out.keys()) == {
            "mean", "std", "ci_lo", "ci_hi", "n_seeds", "per_seed_values",
        }

    def test_zero_variance_three_seeds(self):
        """Identical per-tile metrics across seeds -> std=0, CI collapses."""
        df = make_seed_df({
            0: [0.6, 0.7, 0.8],
            1: [0.6, 0.7, 0.8],
            2: [0.6, 0.7, 0.8],
        })
        out = cross_seed_ci(df, config_filters={"model_name": "A"}, metric="iou")
        assert out["mean"] == pytest.approx(0.7)
        assert out["std"] == pytest.approx(0.0)
        assert out["ci_lo"] == pytest.approx(0.7)
        assert out["ci_hi"] == pytest.approx(0.7)
        assert out["n_seeds"] == 3
        assert sorted(out["per_seed_values"]) == pytest.approx([0.7, 0.7, 0.7])

    def test_known_mean_macro_t_interval(self):
        """Pins the exact t-interval at a known mean/std/n combination.

        Per-seed macro means are [0.60, 0.65, 0.70, 0.75, 0.80]:
            mean = 0.70
            std (ddof=1) = sqrt(0.025 / 4) = 0.07905694...
            t-crit at df=4, 95% two-sided = 2.7764451...
            SE = std / sqrt(5) = 0.03535534...
            half-width = t-crit * SE = 0.09817...
        """
        df = make_seed_df({
            0: [0.55, 0.65],   # per-seed macro mean 0.60
            1: [0.60, 0.70],   # 0.65
            2: [0.65, 0.75],   # 0.70
            3: [0.70, 0.80],   # 0.75
            4: [0.75, 0.85],   # 0.80
        })
        out = cross_seed_ci(df, config_filters={"model_name": "A"}, confidence=0.95)

        assert out["n_seeds"] == 5
        assert out["mean"] == pytest.approx(0.70)
        assert out["std"] == pytest.approx(0.07905694, abs=1e-6)
        assert out["ci_lo"] == pytest.approx(0.70 - 0.09817, abs=1e-4)
        assert out["ci_hi"] == pytest.approx(0.70 + 0.09817, abs=1e-4)

    def test_macro_aggregation_uses_tile_mean(self):
        df = make_seed_df({
            0: [0.5, 0.7, 0.9],   # per-seed macro mean = 0.70
            1: [0.5, 0.7, 0.9],
            2: [0.5, 0.7, 0.9],
        })
        out = cross_seed_ci(df, config_filters={"model_name": "A"}, aggregation="macro")
        assert out["mean"] == pytest.approx(0.7)

    def test_micro_aggregation_uses_count_sums(self):
        """Imbalanced tiles make macro != micro by construction.

        tile_a: tp=100, fp=10, fn=10 -> iou = 100/120 = 0.8333...
        tile_b: tp=1,   fp=0,  fn=0  -> iou = 1.0
        macro: (0.8333 + 1.0) / 2 = 0.9167
        micro: (100+1) / (100+1+10+0+10+0) = 101/121 = 0.8347
        Three seeds with identical data -> std=0 either way.
        """
        rows = []
        for seed in (0, 1, 2):
            rows.append({"model_name": "A", "seed": seed, "tile_id": "t_a",
                         "iou": 100 / 120, "tp": 100, "fp": 10, "fn": 10, "tn": 0})
            rows.append({"model_name": "A", "seed": seed, "tile_id": "t_b",
                         "iou": 1.0,        "tp": 1,   "fp": 0,  "fn": 0,  "tn": 0})
        df = pd.DataFrame(rows)

        out_macro = cross_seed_ci(df, config_filters={"model_name": "A"}, aggregation="macro")
        out_micro = cross_seed_ci(df, config_filters={"model_name": "A"}, aggregation="micro")

        assert out_macro["mean"] == pytest.approx((100 / 120 + 1.0) / 2)
        assert out_micro["mean"] == pytest.approx(101 / 121)
        assert out_macro["std"] == pytest.approx(0.0)
        assert out_micro["std"] == pytest.approx(0.0)

    def test_filter_isolates_one_config(self):
        df = pd.concat([
            make_seed_df({0: [0.5], 1: [0.6], 2: [0.7]},
                         extra_cols={"loss_fn": "focal"}),
            make_seed_df({0: [0.3], 1: [0.4], 2: [0.5]},
                         extra_cols={"loss_fn": "dice"}),
        ], ignore_index=True)

        out = cross_seed_ci(
            df,
            config_filters={"model_name": "A", "loss_fn": "focal"},
        )

        assert out["n_seeds"] == 3
        assert out["mean"] == pytest.approx(0.6)
        assert sorted(out["per_seed_values"]) == pytest.approx([0.5, 0.6, 0.7])

    def test_single_seed_returns_nan_ci(self):
        """Mean defined; std/CI undefined and reported as NaN, not zero."""
        df = make_seed_df({0: [0.5, 0.6, 0.7]})
        out = cross_seed_ci(df, config_filters={"model_name": "A"})
        assert out["n_seeds"] == 1
        assert out["mean"] == pytest.approx(0.6)
        assert math.isnan(out["std"])
        assert math.isnan(out["ci_lo"])
        assert math.isnan(out["ci_hi"])

    def test_raises_on_empty_filter_match(self):
        df = make_seed_df({0: [0.5], 1: [0.6]})
        with pytest.raises(ValueError, match="no rows"):
            cross_seed_ci(df, config_filters={"model_name": "DoesNotExist"})

    def test_raises_on_micro_for_non_derivable_metric(self):
        """Counts-based reduction is only valid for the pixel metrics."""
        df = make_seed_df({0: [0.6, 0.7], 1: [0.5, 0.8]}, metric="apls")
        with pytest.raises(ValueError, match="micro"):
            cross_seed_ci(
                df,
                config_filters={"model_name": "A"},
                metric="apls",
                aggregation="micro",
            )
