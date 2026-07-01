"""Low-level native-CRS window reads + per-image standardisation (NaN-safe).

Shared by the train and eval crop datasets. Pure numpy/rasterio -- no
torch/lightning, so band-only consumers stay light.
"""
import numpy as np


def read_window(src, bands, window):
    """Read ``(C, h, w)`` float32 from an open rasterio dataset, NaN/inf -> 0.

    The COG contains N/A values; scrub them before they reach standardisation or
    a model (NaN would poison per-image mean/std and the forward pass)."""
    arr = src.read(bands, window=window).astype("float32")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def standardize(img):
    """Per-image, per-channel standardisation of a ``(C, H, W)`` float array."""
    mean = img.mean(axis=(1, 2), keepdims=True)
    std = img.std(axis=(1, 2), keepdims=True) + 1e-6
    return (img - mean) / std


def apply_norm(img, bands, mean=None, std=None):
    """Standardise a ``(C, h, w)`` array (``C == len(bands)``).

    Frozen per-band z-score when ``mean``/``std`` are given -- full-stack 1-D stats
    (computed train-only by :mod:`sentinel2data.dataset.compute_norm_stats`), sliced to
    the 1-based ``bands`` so val/test reuse the exact training stats (no leakage). With
    no stats it falls back to per-image :func:`standardize` (the old behaviour).
    """
    if mean is None or std is None:
        return standardize(img)
    idx = [b - 1 for b in bands]
    m = np.asarray(mean, dtype="float32")[idx].reshape(-1, 1, 1)
    s = np.asarray(std, dtype="float32")[idx].reshape(-1, 1, 1)
    s = np.where(s > 1e-6, s, 1.0)  # guard degenerate/constant bands
    return ((img - m) / s).astype("float32")
