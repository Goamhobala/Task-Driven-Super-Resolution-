"""Buffered precision/recall/F1 — the tolerance semantics and the edge cases.

The point of these is that the buffer is ASYMMETRIC (precision relaxes against
the GT's buffer, recall against the prediction's), because that is the part a
"just dilate both masks" implementation gets wrong in a way that silently
inflates the score.
"""
import numpy as np
import pytest

from benchmarking.buffered_metrics import buffered_scores, gt_distance


def _blank(n=32):
    return np.zeros((n, n), dtype=bool)


def test_exact_match_is_one():
    gt = _blank()
    gt[10, 5:25] = True
    out = buffered_scores(gt.copy(), gt, rho=3)
    assert out["buffered_precision"] == pytest.approx(1.0)
    assert out["buffered_recall"] == pytest.approx(1.0)
    assert out["buffered_f1"] == pytest.approx(1.0)


@pytest.mark.parametrize("shift", [1, 2, 3])
def test_within_buffer_scores_one(shift):
    """A line offset by <= rho is perfect under the buffer — the whole point:
    strict F1 would score this 0.0 (no overlapping pixel at all)."""
    gt = _blank()
    gt[10, 5:25] = True
    pred = _blank()
    pred[10 + shift, 5:25] = True
    out = buffered_scores(pred, gt, rho=3)
    assert out["buffered_f1"] == pytest.approx(1.0)

    # ... and strict overlap really is empty, so this is not a trivial pass.
    assert not (pred & gt).any()


def test_beyond_buffer_scores_zero():
    gt = _blank()
    gt[10, 5:25] = True
    pred = _blank()
    pred[10 + 4, 5:25] = True          # 4 px away, rho=3
    out = buffered_scores(pred, gt, rho=3)
    assert out["buffered_f1"] == pytest.approx(0.0)


def test_euclidean_not_chebyshev():
    """rho is a true Euclidean radius: a (3, 3) diagonal offset is 4.24 px away
    and must MISS at rho=3, where a square 7x7 dilation would have caught it."""
    gt = _blank()
    gt[16, 16] = True
    pred = _blank()
    pred[19, 19] = True
    assert buffered_scores(pred, gt, rho=3)["buffered_f1"] == pytest.approx(0.0)
    assert buffered_scores(pred, gt, rho=4.25)["buffered_f1"] == pytest.approx(1.0)


def test_asymmetry_hallucinated_branch_costs_precision_not_recall():
    """A prediction that covers the GT AND invents a far-away branch keeps full
    recall but loses precision. If precision and recall move together here, the
    implementation has dilated both masks instead of buffering each against the
    other."""
    gt = _blank()
    gt[10, 5:25] = True
    pred = gt.copy()
    pred[28, 5:25] = True              # a second road far outside the buffer
    out = buffered_scores(pred, gt, rho=3)
    assert out["buffered_recall"] == pytest.approx(1.0)
    assert out["buffered_precision"] == pytest.approx(0.5)
    assert out["buffered_f1"] == pytest.approx(2 / 3)


def test_missed_branch_costs_recall_not_precision():
    """The mirror image: predicting only half the GT keeps precision at 1.0."""
    gt = _blank()
    gt[10, 5:25] = True
    gt[28, 5:25] = True
    pred = _blank()
    pred[10, 5:25] = True
    out = buffered_scores(pred, gt, rho=3)
    assert out["buffered_precision"] == pytest.approx(1.0)
    assert out["buffered_recall"] == pytest.approx(0.5)


def test_both_empty_is_nan():
    """Undefined, not zero — matches cldice_score and pixel_metrics_from_counts
    so the paired stats drop the pair rather than averaging in a fake 0."""
    out = buffered_scores(_blank(), _blank(), rho=3)
    assert np.isnan(out["buffered_f1"])
    assert np.isnan(out["buffered_precision"])
    assert np.isnan(out["buffered_recall"])


def test_one_side_empty_is_zero():
    gt = _blank()
    gt[10, 5:25] = True
    assert buffered_scores(_blank(), gt, rho=3)["buffered_f1"] == 0.0
    assert buffered_scores(gt, _blank(), rho=3)["buffered_f1"] == 0.0


def test_rho_zero_reduces_to_strict_pixel_scores():
    """rho=0 must degenerate to ordinary precision/recall, which is the cheapest
    check that the buffer is a relaxation of the right quantity."""
    rng = np.random.default_rng(0)
    gt = rng.random((32, 32)) > 0.7
    pred = rng.random((32, 32)) > 0.7
    out = buffered_scores(pred, gt, rho=0)
    tp = float((pred & gt).sum())
    assert out["buffered_precision"] == pytest.approx(tp / pred.sum())
    assert out["buffered_recall"] == pytest.approx(tp / gt.sum())


def test_cached_gt_distance_matches_uncached():
    """The sweep hoists the GT EDT out of the theta loop; that must not change
    a single number."""
    rng = np.random.default_rng(1)
    gt = rng.random((48, 48)) > 0.85
    pred = rng.random((48, 48)) > 0.85
    cached = buffered_scores(pred, gt, rho=3, gt_dist=gt_distance(gt))
    plain = buffered_scores(pred, gt, rho=3)
    assert cached == pytest.approx(plain)


def test_gt_distance_none_for_empty():
    assert gt_distance(_blank()) is None


# --------------------------------------------------------------------------- #
# micro (count-pooled) aggregation of the buffered scores
# --------------------------------------------------------------------------- #
def _chips(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def test_micro_buffered_pools_counts_not_ratios():
    """Micro must weight each chip by its DENOMINATOR. A tiny chip scoring 1.0
    and a huge chip scoring 0.5 is not 0.75 — that is the macro answer."""
    from benchmarking.stats import _micro_metric_from_counts

    df = _chips([
        # n_pred = tp+fp = 10,   perfect
        {"tp": 5, "fp": 5, "fn": 0, "buffered_precision": 1.0, "buffered_recall": 1.0},
        # n_pred = 990,          half
        {"tp": 490, "fp": 500, "fn": 0, "buffered_precision": 0.5, "buffered_recall": 0.5},
    ])
    micro = _micro_metric_from_counts(df, "buffered_precision")
    expected = (1.0 * 10 + 0.5 * 990) / (10 + 990)
    assert micro == pytest.approx(expected)
    assert micro == pytest.approx(0.505, abs=1e-3)
    assert micro != pytest.approx(0.75)          # that would be macro


def test_micro_buffered_f1_is_harmonic_mean_of_pooled_pr():
    """Micro F1 is the harmonic mean of pooled P and pooled R — NOT the pooled
    mean of per-chip F1s, which is not a count-derivable quantity."""
    from benchmarking.stats import _micro_metric_from_counts

    df = _chips([
        {"tp": 80, "fp": 20, "fn": 40, "buffered_precision": 0.9, "buffered_recall": 0.6},
        {"tp": 10, "fp": 90, "fn": 10, "buffered_precision": 0.3, "buffered_recall": 0.8},
    ])
    p = _micro_metric_from_counts(df, "buffered_precision")
    r = _micro_metric_from_counts(df, "buffered_recall")
    f = _micro_metric_from_counts(df, "buffered_f1")
    assert f == pytest.approx(2 * p * r / (p + r))


def test_micro_buffered_skips_zero_denominator_chips():
    """An empty prediction has no precision denominator and a road-free chip no
    recall denominator; neither may drag the pooled value toward zero."""
    from benchmarking.stats import _micro_metric_from_counts

    df = _chips([
        {"tp": 50, "fp": 50, "fn": 0, "buffered_precision": 0.8, "buffered_recall": 0.8},
        # empty prediction: tp+fp == 0, ratio 0.0 by convention
        {"tp": 0, "fp": 0, "fn": 30, "buffered_precision": 0.0, "buffered_recall": 0.0},
        # both empty: NaN on both sides
        {"tp": 0, "fp": 0, "fn": 0, "buffered_precision": float("nan"),
         "buffered_recall": float("nan")},
    ])
    # precision pools over n_pred = tp+fp: 100 for row 1, 0 for the others.
    assert _micro_metric_from_counts(df, "buffered_precision") == pytest.approx(0.8)
    # recall pools over n_gt = tp+fn: 50 for row 1 and 30 for the missed chip,
    # so the missed chip legitimately drags recall down — it had roads to find.
    assert _micro_metric_from_counts(df, "buffered_recall") == pytest.approx(
        (0.8 * 50 + 0.0 * 30) / 80)


def test_micro_is_now_accepted_for_buffered_metrics():
    """cross_seed_ci used to reject these outright with 'requires a
    count-derivable metric'."""
    from benchmarking.stats import cross_seed_ci

    df = _chips([
        {"model_name": "m", "seed": 0, "chip_id": f"c{i}", "tp": 10, "fp": 10,
         "fn": 10, "tn": 100, "buffered_precision": 0.7, "buffered_recall": 0.6,
         "buffered_f1": 0.65}
        for i in range(4)
    ])
    out = cross_seed_ci(df, {"model_name": "m"}, metric="buffered_f1",
                        aggregation="micro")
    assert out["mean"] == pytest.approx(2 * 0.7 * 0.6 / (0.7 + 0.6))
