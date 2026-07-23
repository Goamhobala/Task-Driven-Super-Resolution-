"""clDice METRIC (evaluation-time, HARD skeletons) — verbatim port of the
official implementation (jocpae/clDice, cldice_metric.py), 2D branch.

Not to be confused with the differentiable soft-clDice LOSS in unet.losses:
this one binarizes and uses skimage's exact skeletonization, and is the
protocol composite's second connectivity number alongside APLS.

Official-quirk note: the reference code labels
    tprec = cl_score(v_p, skeletonize(v_l))
    tsens = cl_score(v_l, skeletonize(v_p))
which are SWAPPED relative to the paper's Tprec/Tsens definitions — harmless,
because the final harmonic mean is symmetric in the two terms; the ported
body is kept verbatim regardless.

Declared deviation (empty-case guard only — the official code zero-divides):
both masks empty -> NaN (undefined; the paired stats drop NaN, consistent
with pixel_metrics_from_counts); exactly one side empty, or a skeleton that
vanishes, -> 0.0 (total connectivity failure / pure hallucination — the same
convention as APLS).
"""
from __future__ import annotations

import numpy as np


def cl_score(v, s):
    """[this function computes the skeleton volume overlap] — verbatim."""
    return np.sum(v * s) / np.sum(s)


def cldice_score(v_p: np.ndarray, v_l: np.ndarray) -> float:
    """clDice between a predicted and a ground-truth 2D binary mask."""
    from skimage.morphology import skeletonize

    v_p = np.asarray(v_p).astype(bool)
    v_l = np.asarray(v_l).astype(bool)
    if not v_p.any() and not v_l.any():
        return float("nan")
    if not v_p.any() or not v_l.any():
        return 0.0
    skel_l = skeletonize(v_l)
    skel_p = skeletonize(v_p)
    if not skel_l.any() or not skel_p.any():
        return 0.0
    tprec = cl_score(v_p, skel_l)   # verbatim (labels swapped vs paper; see module docstring)
    tsens = cl_score(v_l, skel_p)
    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))
