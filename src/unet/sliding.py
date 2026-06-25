"""Native-CRS sliding-window inference + cosine-blended stitch.

Separate from ``unet.inference`` so ``unet.model`` can reuse it for stitched
validation without a circular import (``inference`` imports the model class).

Protocol: slide ``size`` windows (stride ``size-overlap``) over a zone in its
**native pixels** (no warp), **zero-pad** short edge windows, predict each, and
**cosine/Hann-weight blend** the overlaps into one seamless probability raster.
Score the metric once per ground pixel over that raster -- never average
per-tile IoU/F1 over overlapping tiles.
"""
from __future__ import annotations

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from unet.patch_dataset import read_window, standardize


def plan_windows(height, width, size=256, overlap=128):
    """Top-left ``(row, col)`` offsets covering the raster.

    ``stride = size - overlap``; the last row/col is anchored so its window ends
    exactly at the edge (so the whole raster is covered)."""
    stride = max(1, size - overlap)

    def offsets(n):
        if n <= size:
            return [0]
        xs = list(range(0, n - size + 1, stride))
        if xs[-1] != n - size:
            xs.append(n - size)
        return xs

    return [(r, c) for r in offsets(height) for c in offsets(width)]


def blend_weight(size):
    """2-D Hann (cosine) weight: centre-high, tapering toward the edges.

    Uses ``hanning(size+2)[1:-1]`` so the weight is strictly positive everywhere
    (no zero border) -- a pixel covered by a single window still keeps its value."""
    w1 = np.hanning(size + 2)[1:-1].astype("float32")
    return np.outer(w1, w1).astype("float32")


@torch.no_grad()
def predict_zone(model, image_path, bands, size=256, overlap=128, normalize=True):
    """Sliding-window probability raster for one zone in its NATIVE CRS.

    Returns ``(prob (H, W) float32, profile)``. Zero-pads short edge windows
    (their padded region is excluded from the blend), scrubs NaN before the
    forward, cosine-blends overlaps. ``overlap=0`` -> non-overlapping tiles.
    """
    model.eval()
    device = next(model.parameters()).device
    w2d = blend_weight(size)

    with rasterio.open(image_path) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()
        prob_sum = np.zeros((H, W), dtype="float32")
        wsum = np.zeros((H, W), dtype="float32")

        for r, c in plan_windows(H, W, size, overlap):
            win = Window(c, r, min(size, W - c), min(size, H - r))
            img = read_window(src, list(bands), win)  # (C, h, w), NaN-scrubbed
            h, w = img.shape[1], img.shape[2]
            if normalize:
                img = standardize(img)               # stats on the valid region only
            if (h, w) != (size, size):               # zero-pad edge window
                padded = np.zeros((img.shape[0], size, size), dtype="float32")
                padded[:, :h, :w] = img
                img = padded

            x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(device)
            p = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()  # (size, size)

            wv = w2d[:h, :w]
            prob_sum[r : r + h, c : c + w] += p[:h, :w] * wv
            wsum[r : r + h, c : c + w] += wv

    prob = prob_sum / np.clip(wsum, 1e-6, None)
    return prob.astype("float32"), profile
