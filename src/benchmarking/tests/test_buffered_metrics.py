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
