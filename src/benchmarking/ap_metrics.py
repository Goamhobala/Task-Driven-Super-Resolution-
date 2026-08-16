"""Average Precision (= AUPRC) from probability maps, binned over the FULL range.

WHY THIS EXISTS RATHER THAN AN AP DERIVED FROM THE theta SWEEP
--------------------------------------------------------------
`theta_sweep_bench` records precision/recall at theta in [0.05, 0.95]. Integrating
that gives an area over whatever slice of RECALL those thresholds happen to
reach, which differs wildly by arm — 0.24->0.79 for one, 0.53->0.55 for
another. An arm whose probabilities are saturated therefore scores near zero
however precise it is. That is truncation of the domain, and no choice of
estimator repairs it: a trapezoid over the same sliver changes the number by a
rounding error and widens coverage by nothing.

Scoring the PROBABILITIES directly fixes it by construction. The binned
estimator sweeps thresholds across the whole [0, 1] interval, so the PR curve
is anchored at both ends — recall 0 at threshold 1, recall 1 at threshold 0 —
and every arm is integrated over the same, complete domain.

STEP, NOT TRAPEZOID
-------------------
`torchmetrics.BinaryAveragePrecision` uses the step-wise sum
``sum_n (R_n - R_{n-1}) * P_n``, matching sklearn's `average_precision_score`.
This is deliberate and correct for PR curves: precision does not vary linearly
with recall between achievable operating points, so linear interpolation
between them is systematically optimistic (Davis & Goadrich 2006, *The
Relationship Between Precision-Recall and ROC Curves*). sklearn's own docs warn
against running a trapezoidal `auc` over a PR curve for the same reason. So the
step rule is not an approximation we tolerate — it is the one that does not
inflate the score.

BINNED, AND WHY THAT IS FINE HERE
---------------------------------
Passing an int `thresholds` makes torchmetrics accumulate a multi-threshold
confusion matrix instead of retaining every probability. A 1024x1024 chip holds
~1M floats; exact AP would sort them all, per chip, for 760 chips. The binned
form is O(bins) memory and closes on the exact value as bins rise. 101 bins
resolves probabilities to 0.01, comfortably finer than the 0.025 theta grid the
sweep already uses.
"""
from __future__ import annotations

import numpy as np

DEFAULT_AP_BINS = 101


def chip_average_precision(probs, target, bins: int = DEFAULT_AP_BINS) -> float:
    """AP for ONE chip: probabilities (H, W) or (1, H, W) vs a binary mask.

    Returns NaN when the chip has no positive ground truth — AP is undefined
    with an empty positive class, and a fabricated 0.0 would drag every mean
    down in proportion to how many road-free chips a split happens to contain.
    That matches the NaN convention in `buffered_metrics` and `cldice_score`,
    so the paired stats drop the pair rather than averaging in a fiction.
    """
    import torch
    from torchmetrics.functional.classification import binary_average_precision

    p = torch.as_tensor(np.asarray(probs)).flatten().float()
    t = torch.as_tensor(np.asarray(target)).flatten().long()
    if t.sum() == 0:
        return float("nan")
    return float(binary_average_precision(p, t, thresholds=int(bins)))


def batch_average_precision(probs, targets, bins: int = DEFAULT_AP_BINS) -> list[float]:
    """Per-chip AP for a batch: probs (B, 1, H, W) or (B, H, W), targets (B, H, W)."""
    arr = np.asarray(probs)
    if arr.ndim == 4:
        arr = arr[:, 0]
    return [chip_average_precision(arr[i], np.asarray(targets[i]), bins)
            for i in range(arr.shape[0])]
