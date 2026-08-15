"""BUFFERED (relaxed) precision / recall / F1 — the road-extraction convention.

Strict pixel F1 punishes a centreline that is right in every way except its
exact position: a predicted road one pixel beside the label scores zero TP and
counts twice, once as FP and once as FN. At 2.5 m GSD, with labels digitised
from a different source than the imagery, that penalty is mostly measuring
label registration rather than model quality.

The standard fix (Wiedemann et al.'s relaxed completeness/correctness, and the
"buffer" scores used throughout the SpaceNet road work) scores a pixel against
a DILATED version of the other mask:

    buffered precision = |predicted road within rho px of GT road| / |predicted road|
    buffered recall    = |GT road within rho px of predicted road| / |GT road|
    buffered F1        = harmonic mean of the two

Note the asymmetry is deliberate and is why this is not just "dilate both and
take F1": precision is relaxed against the GT's buffer, recall against the
prediction's buffer. Dilating both at once would double-count the tolerance
and inflate the score.

`rho` is in PIXELS of the grid the masks are on. These masks are 2.5 m, so the
default rho=3 is a 7.5 m tolerance — about one lane width either side, and the
scale at which the labels themselves are trustworthy.

Distances are exact Euclidean (`scipy.ndimage.distance_transform_edt`), not
iterated binary dilation, so rho=3 means "within 3 px as the crow flies"
rather than a 7x7 square. The EDT is O(N) per mask and runs on the chip, which
is what makes this cheap enough to evaluate at all 37 thresholds of a sweep.

Empty-mask convention follows `cldice_score` and `pixel_metrics_from_counts`
exactly, so these columns pair with the others under the same NaN rules:
both masks empty -> NaN (undefined; the paired stats drop NaN pairs); exactly
one side empty -> 0.0 (total miss or pure hallucination).
"""
from __future__ import annotations

import numpy as np

DEFAULT_BUFFER_PX = 3


def _within(mask: np.ndarray, other: np.ndarray, rho: float) -> float:
    """Fraction of `mask`'s True pixels lying within `rho` px of `other`.

    `distance_transform_edt` measures distance to the nearest ZERO, so it is
    fed ``~other``: the result is each pixel's distance to the nearest True of
    `other`, which is what "within rho of the other mask" needs.
    """
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(~other)
    return float((dist[mask] <= rho).mean())


def buffered_scores(pred: np.ndarray, gt: np.ndarray,
                    rho: float = DEFAULT_BUFFER_PX,
                    gt_dist: np.ndarray | None = None) -> dict[str, float]:
    """Buffered precision / recall / F1 for one pair of 2D binary masks.

    `gt_dist` optionally supplies a precomputed distance transform of the GT
    (``distance_transform_edt(~gt)``). The GT does not change as a threshold
    sweeps, so hoisting it out of the theta loop halves the EDT work — see
    `runner._batch_rows`.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    p_any, g_any = pred.any(), gt.any()
    if not p_any and not g_any:
        return {"buffered_precision": float("nan"),
                "buffered_recall": float("nan"),
                "buffered_f1": float("nan")}
    if not p_any or not g_any:
        return {"buffered_precision": 0.0,
                "buffered_recall": 0.0,
                "buffered_f1": 0.0}

    from scipy.ndimage import distance_transform_edt

    if gt_dist is None:
        gt_dist = distance_transform_edt(~gt)
    precision = float((gt_dist[pred] <= rho).mean())
    recall = _within(gt, pred, rho)

    denom = precision + recall
    f1 = 0.0 if denom == 0 else 2.0 * precision * recall / denom
    return {"buffered_precision": precision,
            "buffered_recall": recall,
            "buffered_f1": f1}


def buffered_scores_multi(pred: np.ndarray, gt: np.ndarray,
                          radii=(1, 2, 3, 4, 5),
                          gt_dist: np.ndarray | None = None) -> dict[str, float]:
    """Buffered P/R/F1 at SEVERAL tolerances, from one pair of distance maps.

    Scoring rho = 1..5 costs barely more than scoring rho = 3 alone: the two
    EDTs are the whole expense, and each extra radius is one comparison against
    an array that already exists. That is what makes a tolerance SWEEP —
    "how fast does the score rise as we forgive more registration error?" —
    cheap enough to record on every bench rather than as a special study.

    Keys are suffixed with the radius (`buffered_f1_r3`), unlike the
    single-radius `buffered_scores`, whose unsuffixed names existing stores
    already use. Both may be emitted side by side without collision.
    """
    from scipy.ndimage import distance_transform_edt

    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    radii = [float(r) for r in radii]

    p_any, g_any = pred.any(), gt.any()
    out: dict[str, float] = {}
    if not p_any and not g_any:                     # undefined, as in cldice
        for r in radii:
            k = _rkey(r)
            out |= {f"buffered_precision_{k}": float("nan"),
                    f"buffered_recall_{k}": float("nan"),
                    f"buffered_f1_{k}": float("nan")}
        return out
    if not p_any or not g_any:                      # total miss / pure hallucination
        for r in radii:
            k = _rkey(r)
            out |= {f"buffered_precision_{k}": 0.0,
                    f"buffered_recall_{k}": 0.0, f"buffered_f1_{k}": 0.0}
        return out

    if gt_dist is None:
        gt_dist = distance_transform_edt(~gt)
    pred_dist = distance_transform_edt(~pred)
    d_pred_to_gt = gt_dist[pred]        # distance from each predicted px to GT
    d_gt_to_pred = pred_dist[gt]        # ... and from each GT px to prediction

    for r in radii:
        k = _rkey(r)
        precision = float((d_pred_to_gt <= r).mean())
        recall = float((d_gt_to_pred <= r).mean())
        den = precision + recall
        out[f"buffered_precision_{k}"] = precision
        out[f"buffered_recall_{k}"] = recall
        out[f"buffered_f1_{k}"] = 0.0 if den == 0 else 2 * precision * recall / den
    return out


def _rkey(r: float) -> str:
    """`3` -> 'r3', `2.5` -> 'r2p5' — a column-name-safe radius suffix."""
    return "r" + (f"{int(r)}" if float(r).is_integer() else f"{r}".replace(".", "p"))


def gt_distance(gt: np.ndarray) -> np.ndarray | None:
    """EDT of a GT chip, or None when the chip has no road at all.

    None rather than an array of infinities because the all-empty case is
    handled by the NaN/0.0 convention above, and allocating a full-size
    distance map for a road-free chip is pure waste in a sweep.
    """
    from scipy.ndimage import distance_transform_edt

    gt = np.asarray(gt).astype(bool)
    if not gt.any():
        return None
    return distance_transform_edt(~gt)
