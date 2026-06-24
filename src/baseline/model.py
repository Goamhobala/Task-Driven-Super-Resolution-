"""Model + loss for the baseline.

The architecture is the notebook's baseline (UNet++ with a ResNet encoder) but
generalised to an arbitrary number of input channels so the same code trains the
RGB-only (M0) and the full 14-band (M3) configs — only `in_channels` changes.
smp adapts the pretrained first conv to `in_channels` automatically.

The loss is the notebook's RoadSegLoss: BCE (class imbalance) + Dice (region) +
clDice (topology). clDice / SoftSkeletonize are lifted verbatim from the notebook.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(in_channels: int = 3, encoder: str = "resnet50",
                encoder_weights: str | None = "imagenet", classes: int = 1) -> nn.Module:
    """UNet++ with a ResNet encoder.

    Outputs raw logits with `classes` channels (1 = binary sigmoid head; use 2+
    for TerraTorch's softmax convention). With `in_channels != 3` smp
    reinitialises / repeats the pretrained first conv.
    """
    return smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


class SoftSkeletonize(nn.Module):
    """Differentiable morphological skeletonisation via iterative erosion."""

    def __init__(self, num_iter: int = 10):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -nn.functional.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -nn.functional.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return nn.functional.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        skel = torch.zeros_like(img)
        for _ in range(self.num_iter):
            opened = self.soft_open(img)
            delta = nn.functional.relu(img - opened)
            skel = torch.max(skel, delta)
            img = self.soft_erode(img)
        return skel


class clDiceLoss(nn.Module):
    """Soft-Dice on the mask + soft-Dice on the skeleton (centre-line) to keep
    road topology. alpha=0 -> pure Dice, alpha=1 -> pure clDice."""

    def __init__(self, alpha: float = 0.5, skel_iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.skeletonize = SoftSkeletonize(num_iter=skel_iters)

    def soft_dice(self, pred, target):
        inter = (pred * target).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1 - (2 * inter + self.smooth) / (denom + self.smooth)

    def forward(self, logits, targets):
        pred = torch.sigmoid(logits)
        dice_loss = self.soft_dice(pred, targets).mean()

        skel_pred = self.skeletonize(pred)
        skel_target = self.skeletonize(targets)

        prec = ((skel_pred * targets).sum(dim=(1, 2, 3)) + self.smooth) / \
               (skel_pred.sum(dim=(1, 2, 3)) + self.smooth)
        rec = ((pred * skel_target).sum(dim=(1, 2, 3)) + self.smooth) / \
              (skel_target.sum(dim=(1, 2, 3)) + self.smooth)
        cl_dice_loss = 1 - (2 * prec * rec / (prec + rec + 1e-8)).mean()

        return (1 - self.alpha) * dice_loss + self.alpha * cl_dice_loss


class RoadSegLoss(nn.Module):
    """BCE (imbalance) + Dice (region) + clDice (topology)."""

    def __init__(self, pos_weight, alpha: float = 0.3, skel_iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cldice = clDiceLoss(alpha=alpha, skel_iters=skel_iters, smooth=smooth)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.cldice(logits, targets)
