#!/usr/bin/env python3
"""Stage 0 sanity checks for the loss ablation (protocol §4.4 + weight maps).

Pure NumPy/SciPy mirror of the weight-map builders in unet/losses.py —
runs anywhere (no torch needed). Produces:

  1. <out>/stage0_weight_maps.png — GapLoss & TL weight maps on synthetic
     scenes (line gaps, free ends, closed ring control) across the protocol
     grids r ∈ {3,5,9}, ℓ ∈ {3,5,7}.
  2. Scale-parity table (§4.4): normalized weighted-CE vs plain BCE ratio.
  3. Timing: per-batch weight-map cost at train shape (8×256×256).
  4. --cross-check: if torch + unet.losses import, assert the NumPy
     mirror matches the torch implementation exactly.

Run:  python scripts/stage0_sanity.py --out runs/stage0 [--cross-check]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import skeletonize

RNG = np.random.default_rng(0)


# ------------------------------------------------ numpy mirror of losses.py

def np_skel(binary: np.ndarray) -> np.ndarray:
    return skeletonize(binary > 0.5).astype(np.float32)


def np_gap_weights(prob: np.ndarray, r: int = 4, K: float = 60.0,
                   thresh: float = 0.5) -> np.ndarray:
    """GapLoss (Yuan & Xu 2022 Alg. 1): W = K·N in (2r+1)² window, else 1."""
    skel = np_skel(prob > thresh)
    nbrs = convolve2d(skel, np.ones((3, 3)), mode="same") - skel
    endpoints = ((skel > 0.5) & (np.round(nbrs) == 1)).astype(np.float32)
    cnt = np.round(convolve2d(endpoints, np.ones((2 * r + 1, 2 * r + 1)), mode="same"))
    return np.where(cnt > 0, K * cnt, 1.0)


def np_tl_weights(prob: np.ndarray, ell: int = 5, thresh: float = 0.5,
                  base_reset: bool = True) -> np.ndarray:
    """Topological Loss (Nanni et al. 2024 Alg. 1) + Giannini base reset."""
    kernels = [np.ones((ell, 1)), np.ones((1, ell)),
               np.eye(ell), np.fliplr(np.eye(ell))]
    skel = np_skel(prob > thresh)
    W = np.zeros_like(skel)
    for k in kernels:
        C = (np.round(convolve2d(skel, k, mode="same")) == 2).astype(np.float32)
        D = convolve2d(C, 10.0 * k, mode="same")
        D = np.minimum(D, 10.0)
        D[D == 0] = 1.0
        W = W + D
    W[W >= 10.0] = 10.0
    if base_reset:
        W[W == float(len(kernels))] = 1.0
    return W


# ----------------------------------------------------------- synthetic scene

def make_scene(n: int = 192) -> np.ndarray:
    """Roads-like prediction: lines with gaps, a free end, a ring control."""
    m = np.zeros((n, n), np.float32)
    m[40, 10:170] = 1;  m[40, 80:104] = 0          # horizontal, 24 px gap
    m[10:120, 140] = 1; m[58:74, 140] = 0          # vertical, 16 px gap
    for i in range(90):                              # diagonal with gap
        y, x = 100 + i // 2, 20 + i
        if not (45 < i < 62):
            m[y, x] = 1
    m[150:178, 30] = 1                               # free end (dead-end stub)
    yy, xx = np.mgrid[:n, :n]
    ring = np.abs(np.sqrt((yy - 150) ** 2 + (xx - 150) ** 2) - 24) < 1.0
    return np.maximum(m, ring.astype(np.float32))    # ring: no endpoints


def blobby_mask(n: int, frac: float = 0.03) -> np.ndarray:
    m = (RNG.random((n, n)) > (1 - frac)).astype(np.float32)
    return np.minimum(convolve2d(m, np.ones((5, 5)), mode="same"), 1.0)


# ------------------------------------------------------------------- checks

def scale_parity(n: int = 256) -> list[str]:
    """§4.4: sum(W·ce)/sum(W) must sit at BCE's scale."""
    logits = RNG.normal(0, 3, (n, n))
    p = 1 / (1 + np.exp(-logits))
    y = blobby_mask(n)
    ce = -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
    bce = ce.mean()
    rows = [f"plain BCE                      : {bce:.4f}  (reference)"]
    for r in (3, 5, 9):
        W = np_gap_weights(p, r=r)
        rows.append(f"gap_ce r={r} K=60  norm         : {(W*ce).sum()/W.sum():.4f}"
                    f"  ratio {(W*ce).sum()/W.sum()/bce:.2f}"
                    f"   [unnorm would be {(W*ce).mean():.4f}, "
                    f"{(W*ce).mean()/bce:.1f}x]")
    for ell in (3, 5, 7):
        W = np_tl_weights(p, ell=ell)
        rows.append(f"tl_ce  ell={ell}    norm         : {(W*ce).sum()/W.sum():.4f}"
                    f"  ratio {(W*ce).sum()/W.sum()/bce:.2f}"
                    f"   [unnorm would be {(W*ce).mean():.4f}, "
                    f"{(W*ce).mean()/bce:.1f}x]")
    return rows


def timing(batch: int = 8, n: int = 256, reps: int = 3) -> list[str]:
    masks = [blobby_mask(n, 0.02) for _ in range(batch)]
    out = []
    for name, fn in [("gap r=5", lambda m: np_gap_weights(m, r=5)),
                     ("tl ell=5", lambda m: np_tl_weights(m, ell=5)),
                     ("tl ell=7", lambda m: np_tl_weights(m, ell=7))]:
        t0 = time.perf_counter()
        for _ in range(reps):
            for m in masks:
                fn(m)
        ms = (time.perf_counter() - t0) / reps * 1000
        out.append(f"{name:9s}: {ms:6.1f} ms / batch of {batch}x{n}x{n}")
    t0 = time.perf_counter()
    for _ in range(reps):
        for m in masks:
            np_skel(m)
    out.append(f"(skeletonize alone: {(time.perf_counter()-t0)/reps*1000:6.1f} ms)")
    return out


def cross_check() -> str:
    try:
        import torch
        from unet.losses import gap_weight_map, tl_weight_map
    except ImportError as e:
        return f"cross-check skipped ({e})"
    p_np = 1 / (1 + np.exp(-RNG.normal(0, 3, (128, 128))))
    p = torch.from_numpy(p_np.astype(np.float32))[None, None]
    for r in (3, 5, 9):
        a = gap_weight_map(p, r=r)[0, 0].numpy()
        b = np_gap_weights(p_np, r=r)
        assert np.allclose(a, b, atol=1e-4), f"gap mismatch r={r}"
    for ell in (3, 5, 7):
        a = tl_weight_map(p, ell=ell)[0, 0].numpy()
        b = np_tl_weights(p_np, ell=ell)
        assert np.allclose(a, b, atol=1e-4), f"tl mismatch ell={ell}"
    return "cross-check PASSED: torch implementation == numpy mirror"


# --------------------------------------------------------------------- plot

def render(out: Path):
    scene = make_scene()
    skel = np_skel(scene)
    nbrs = convolve2d(skel, np.ones((3, 3)), mode="same") - skel
    endpoints = (skel > 0.5) & (np.round(nbrs) == 1)

    panels = [("prediction (synthetic)", scene, "gray", None),
              ("skeleton + endpoints (red)", None, None, None)]
    panels += [(f"GapLoss W  (r={r}, K=60)", np_gap_weights(scene, r=r),
                "inferno", "log") for r in (3, 5, 9)]
    panels += [(f"TL W  (ell={e}, base reset)", np_tl_weights(scene, ell=e),
                "inferno", None) for e in (3, 5, 7)]
    panels.append(("TL W (ell=5, NO reset — orig. paper)",
                   np_tl_weights(scene, ell=5, base_reset=False), "inferno", None))

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, (title, img, cmap, scale) in zip(axes.flat, panels):
        if img is None:  # skeleton overlay panel
            rgb = np.stack([skel] * 3, -1) * 0.7
            rgb[endpoints] = [1, 0, 0]
            ys, xs = np.where(endpoints)
            ax.imshow(rgb)
            ax.scatter(xs, ys, s=60, facecolors="none", edgecolors="red")
        else:
            norm = matplotlib.colors.LogNorm() if scale == "log" else None
            im = ax.imshow(img, cmap=cmap, norm=norm)
            if cmap == "inferno":
                fig.colorbar(im, ax=ax, fraction=0.045)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Stage 0 — weight-map sanity (gaps get weight, ring control stays flat)",
                 fontsize=14)
    fig.tight_layout()
    path = out / "stage0_weight_maps.png"
    fig.savefig(path, dpi=110)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/stage0")
    ap.add_argument("--cross-check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("== §4.4 scale parity (normalized weighted CE vs BCE) ==")
    lines = scale_parity()
    print("\n".join(lines))
    print("\n== weight-map timing (CPU) ==")
    tlines = timing()
    print("\n".join(tlines))
    if args.cross_check:
        print("\n== torch/numpy cross-check ==")
        print(cross_check())
    path = render(out)
    print(f"\nwrote {path}")
    (out / "stage0_report.txt").write_text(
        "== scale parity ==\n" + "\n".join(lines) +
        "\n\n== timing ==\n" + "\n".join(tlines) + "\n")


if __name__ == "__main__":
    main()
