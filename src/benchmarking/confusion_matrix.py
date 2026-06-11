"""Per-chip confusion matrix and derived pixel metrics for binary segmentation.

The counts (tp, fp, fn, tn) are computed PER chip — reduced over H and W but not
over the batch — because the benchmarking store keeps one row per chip. IoU, F1,
precision, recall, and everything downstream (bootstrap / Wilcoxon) are derived
from these four numbers.

Positive class = road (target == 1). Pixels equal to `ignore_index` are excluded
from all four counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ConfusionCounts:
    """Per-chip confusion-matrix counts. Each field is a 1-D LongTensor of length B."""

    tp: Tensor
    fp: Tensor
    fn: Tensor
    tn: Tensor


def _to_bhw(x: Tensor) -> Tensor:
    """Normalise [B,1,H,W] / [B,H,W] / [H,W] to [B,H,W]."""
    if x.dim() == 4:  # [B, C, H, W]
        if x.shape[1] != 1:
            raise ValueError(
                f"expected a single channel for binary segmentation, got {x.shape[1]}; "
                "for multiclass, argmax over the class dim before calling this"
            )
        x = x.squeeze(1)  # [B, H, W]
    elif x.dim() == 2:  # [H, W] -> single tile
        x = x.unsqueeze(0)
    elif x.dim() != 3:
        raise ValueError(f"unsupported shape {tuple(x.shape)}")
    return x


def confusion_counts(
    output: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    ignore_index: int | None = None,
) -> ConfusionCounts:
    """Per-chip (tp, fp, fn, tn) for binary road segmentation.

    Args:
        output: the `.output` field of a terratorch ModelOutput. Logits when
            `from_logits` (default), else probabilities in [0, 1]. Accepts
            [B,1,H,W], [B,H,W] or [H,W].
        target: ground-truth mask, same spatial layout. road == 1, background == 0.
        threshold: probability threshold for the positive (road) class.
        from_logits: if True, apply sigmoid to `output` first.
        ignore_index: target pixels equal to this are dropped from every count.

    Returns:
        ConfusionCounts, each field a 1-D LongTensor of length B (one per tile).
    """
    output = _to_bhw(output)
    target = _to_bhw(target)
    if output.shape != target.shape:
        raise ValueError(
            f"shape mismatch: output {tuple(output.shape)} vs target {tuple(target.shape)}"
        )

    probs = torch.sigmoid(output) if from_logits else output
    pred = probs >= threshold  # [B, H, W] bool

    pos = target == 1
    valid = torch.ones_like(target, dtype=torch.bool) if ignore_index is None else (target != ignore_index)

    spatial = (1, 2)  # reduce H, W; keep the batch axis
    tp = (pred & pos & valid).sum(dim=spatial)
    fp = (pred & ~pos & valid).sum(dim=spatial)
    fn = (~pred & pos & valid).sum(dim=spatial)
    tn = (~pred & ~pos & valid).sum(dim=spatial)
    return ConfusionCounts(tp=tp.long(), fp=fp.long(), fn=fn.long(), tn=tn.long())


def pixel_metrics_from_counts(c: ConfusionCounts) -> dict[str, Tensor]:
    """Per-chip IoU, F1, precision, recall from confusion counts.

    A metric with a zero denominator (e.g. a chip with no road in prediction or
    ground truth) is returned as NaN — not 0 or 1. NaN is the honest "undefined
    here" value, and the downstream bootstrap / Wilcoxon already drop NaN pairs,
    so undefined tiles are excluded from comparisons rather than biasing them.
    """
    tp, fp, fn = c.tp.float(), c.fp.float(), c.fn.float()

    def safe(num: Tensor, den: Tensor) -> Tensor:
        out = num / den
        out[den == 0] = float("nan")
        return out

    return {
        "iou": safe(tp, tp + fp + fn),
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),
    }