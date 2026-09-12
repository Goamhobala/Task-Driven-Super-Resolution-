"""Instrument A in each arm's OWN frame — the U-Net-input view of separability.

    PYTHONPATH=src python -m sr.probes.lda_own \
        --cache-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/probe_cache \
        --cache-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/probe_cache_sronly

Reads only `sr_pixels.npy` + `meta.json`, so a `extract.py --sr-only` cache is
enough. Emits, into `--out-dir` (default `figures/probes/lda_own_frame/`):

    lda_own_fisher.csv      one row per arm-seed: Fisher, d', AUC, held-out Fisher
    lda_own_axes.csv        own LD1 / PC-perp loadings, raw and on the z-scored input
    F1_lda_own.pdf/.png     own-LD1 ridgeline | own-frame Fisher, rows aligned
    A1_lda_own_2d.pdf/.png  (own LD1, own PC-perp) per arm, standardised axes

WHY THE OWN FRAME
-----------------
The question is how linearly separable road is from background in the tensor
the U-Net is handed. What the U-Net does to that tensor first — the per-band
z-score, then the stem convolution — is affine, so the relevant number is the
best any linear read-out achieves on each arm's output, not how well r0's
direction transfers to it. `lda.py`'s `r0_frame` Fisher mixes "less separable"
with "separable along a different direction"; the own frame drops the second.

The maximal two-class Fisher ratio is invariant to any invertible affine map of
the bands, so a per-band drift in scale or offset (r4b's SR4RS walking off the
reflectance scale) does not move it — correctly, since the z-score removes
exactly that before the U-Net sees the input.

SHARED AXES WITHOUT A SHARED FRAME
----------------------------------
Own-frame projections are in different units per arm, so each is standardised
before plotting: background mean at 0, unit pooled within-class SD
sqrt((var_road + var_bg) / 2). Road's mean then sits at d' = sqrt(2 J), so the
ridgeline rows are comparable even though their directions are not.

HELD-OUT CHECK
--------------
The own axis is fitted and scored on the same pixels, which flatters it by
construction. `fisher_own_heldout` fits on half the fixture's chips and scores
the other half (both ways, averaged), split by CHIP so neighbouring pixels never
straddle the fold. Four coefficients on 50k pixels should make the gap
negligible; if it is not, the in-sample number is not to be read.

WHAT IT CANNOT SAY
------------------
As in `lda.py`: per-pixel, linear, 4 bands. Road is thin and the U-Net's edge is
spatial, so this is "the front-end handed over a cleaner spectral problem",
never an IoU claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sr.probes import cache, style
from sr.probes.lda import _density, fisher, lda_axis, orthogonal_pc
from sr.probes.make_fixtures import FIXTURE_DIR, PIXELS_NPZ

OUT_DIR = str(Path(style.FIGURES_DIR) / "lda_own_frame")
FOLD_SEED = 20260911


# ------------------------------------------------------------------- the maths
def own_frame(x: np.ndarray, y: np.ndarray):
    """(LD1, PC-perp) fitted on this arm, LD1 signed so road projects higher."""
    w = lda_axis(x, y)
    if (x[y] @ w).mean() < (x[~y] @ w).mean():
        w = -w
    return w, orthogonal_pc(x, y, w)


def standardise(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Background mean -> 0, pooled within-class SD -> 1."""
    s = np.sqrt((p[y].var() + p[~y].var()) / 2) or 1.0
    return (p - p[~y].mean()) / s


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """P(a road pixel projects above a background pixel), ties counted half.

    The Fisher ratio is a Gaussian summary; this is the same axis read without
    that assumption, so a multimodal background cannot flatter or hide a change.
    """
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def heldout_fisher(x, y, chip, seed=FOLD_SEED) -> float:
    """Two-fold, chip-grouped: fit LD1 on one half of the chips, score the other."""
    chips = np.unique(chip)
    half = np.random.default_rng(seed).permutation(chips)[: len(chips) // 2]
    a = np.isin(chip, half)
    return float(np.mean([fisher(x[te], y[te], lda_axis(x[tr], y[tr]))
                          for tr, te in ((a, ~a), (~a, a))]))


# ------------------------------------------------------------------- the cache
def load_runs(cache_dirs, arms):
    """[(meta, pixels)] across several cache dirs, on ONE fixture, in figure order."""
    metas = []
    for d in cache_dirs:
        try:
            metas += cache.load_metas(d, arms, require=("sr_pixels.npy",))
        except SystemExit as e:
            if "no extraction cache" not in str(e):
                raise
            print(f"  (nothing selected under {d})")
    if not metas:
        raise SystemExit("no sr_pixels.npy cache selected under any --cache-dir")
    hashes = {m["fixture_hash"] for m in metas}
    if len(hashes) > 1:
        raise SystemExit(
            f"the cache dirs hold {len(hashes)} different fixtures "
            f"({', '.join(sorted(hashes))}); they are not on one axis.")
    names = [m["run"] for m in metas]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(f"run(s) cached in more than one --cache-dir: {dupes}")
    metas.sort(key=lambda m: (style.sort_key(m["arm"]),
                              m["seed"] if m["seed"] is not None else -1))
    return [(m, np.load(m["dir"] / "sr_pixels.npy").astype("float64"))
            for m in metas]


# ------------------------------------------------------------------- figures
def _by_arm(proj):
    by_arm = {}
    for p in proj:
        by_arm.setdefault(p[0]["arm"], []).append(p)
    return by_arm, sorted(by_arm, key=style.sort_key)


def _ridgeline(ax, proj):
    """Standardised own-LD1 densities: background outlined, road filled.

    Every seed of an arm is drawn; the first carries the fill, the rest are
    outlines, so the spread between them reads as the within-arm noise floor.
    """
    by_arm, arms = _by_arm(proj)
    lo, hi = np.percentile(np.concatenate([z for _, z, _, _ in proj]), [0.2, 99.8])
    grid = np.linspace(lo, hi, 220)
    ticks = []
    for i, arm in enumerate(arms):
        base = len(arms) - 1 - i
        ticks.append(base + 0.35)
        for j, (_, z, _, y) in enumerate(by_arm[arm]):
            for cls, colour in ((False, style.BG), (True, style.ROAD)):
                p = z[y == cls]
                if p.size < 2:
                    continue
                d = _density(p, grid)
                d = d / d.max() * 0.9
                if j == 0 and cls:
                    ax.fill_between(grid, base, base + d, color=colour,
                                    alpha=0.5, lw=0)
                ax.plot(grid, base + d, color=colour,
                        lw=0.9 if j == 0 else 0.6, alpha=1.0 if j == 0 else 0.55)
        ax.axhline(base, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0, color=style.ZERO_LINE, lw=0.5, ls=":", zorder=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels([style.label(a) for a in arms])
    for t, a in zip(ax.get_yticklabels(), arms):
        t.set_color(style.color(a))
        t.set_fontsize(6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.05, len(arms))
    ax.set_xlabel("own LD1  (background mean 0, pooled within-class SD 1)")
    ax.set_title("SR output along each arm's own discriminant", loc="left")
    ax.plot([], [], color=style.ROAD, lw=3, alpha=0.5, label="road")
    ax.plot([], [], color=style.BG, lw=0.9, label="background")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.06),
              fontsize=6.5)
    return arms, ticks


def _fisher_dots(ax, df, arms, ticks):
    """Own-frame Fisher per arm-seed, on the ridgeline's rows; bar = arm mean."""
    r0 = df.loc[df["arm"] == "r0", "fisher_own_frame"]
    if len(r0):
        ax.axvline(r0.mean(), color=style.color("r0"), lw=0.7, ls=":", zorder=0)
    for arm, yt in zip(arms, ticks):
        g = df[df["arm"] == arm]
        c = style.color(arm)
        kw = style.marker_kwargs(len(g), marker=style.marker(arm), size=4.5)
        jit = np.linspace(-0.15, 0.15, len(g)) if len(g) > 1 else [0.0]
        for v, dj in zip(g["fisher_own_frame"], jit):
            ax.plot([v], [yt + dj], color=c, zorder=3, **kw)
        if len(g) > 1:
            m = g["fisher_own_frame"].mean()
            ax.plot([m, m], [yt - 0.3, yt + 0.3], color=c, lw=1.2, zorder=2)
        ax.axhline(yt - 0.35, color="0.85", lw=0.5, zorder=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.05, len(arms))
    ax.set_xlabel("Fisher ratio, own frame")
    ax.set_title("Best linear separability", loc="left")
    # The marker channels are only legible with a key: shape carries the hard
    # constraint (style.marker), fill carries replication (style.marker_kwargs),
    # and neither is stated by the row labels opposite. Drawn in neutral grey —
    # the arms' own colours are already on their rows, and repeating them here
    # would build a second, redundant arm legend.
    grey = {"color": style.GREY, "linestyle": "none"}
    ax.plot([], [], marker="o", markersize=4.5, markerfacecolor="none",
            markeredgewidth=1.4, label="HC on", **grey)
    ax.plot([], [], marker="s", markersize=4.5, markerfacecolor="none",
            markeredgewidth=1.4, label="HC off", **grey)
    ax.plot([], [], marker="D", markersize=4.5, markerfacecolor="none",
            markeredgewidth=1.4, label="anchor", **grey)
    ax.plot([], [], marker="o", markersize=4.5, markeredgewidth=0.8,
            label=f"≥{style.MIN_SEEDS_FOR_ERRORBAR} seeds (filled)", **grey)
    if len(r0):
        ax.plot([], [], color=style.color("r0"), lw=0.7, ls=":", label="r0")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.06),
              fontsize=6.5, handletextpad=0.4, columnspacing=1.0)


def figure_2d(proj, df, out_dir, stem):
    """Appendix: (own LD1, own PC-perp), both standardised, first seed per arm."""
    import matplotlib.pyplot as plt

    by_arm, arms = _by_arm(proj)
    ncol = min(len(arms), 4)
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.FULL_WIDTH_IN, 2.0 * nrow),
                             squeeze=False)
    xlim = np.percentile(np.concatenate([z for _, z, _, _ in proj]), [0.2, 99.8])
    ylim = np.percentile(np.concatenate([q for _, _, q, _ in proj]), [0.2, 99.8])
    for ax, arm in zip(axes.ravel(), arms):
        m, z, q, y = by_arm[arm][0]
        for cls, colour in ((False, style.BG), (True, style.ROAD)):
            ax.scatter(z[y == cls], q[y == cls], s=1.5, alpha=0.25, color=colour,
                       linewidths=0, rasterized=True)
        j = df.loc[df["run"] == m["run"], "fisher_own_frame"].iloc[0]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{style.label(arm)}\nseed {m['seed']}, J = {j:.3f}",
                     fontsize=6.5, loc="left", color=style.color(arm))
    for ax in axes.ravel()[len(arms):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("own LD1 (std.)")
    for ax in axes[:, 0]:
        ax.set_ylabel("own PC⊥ (std.)")
    fig.suptitle("SR output in each arm's own LDA frame (road red, background grey)",
                 fontsize=8, y=1.0)
    fig.tight_layout()
    return style.save(fig, out_dir, stem)


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", action="append", default=None,
                    help="repeatable; default probe_cache")
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm-key globs to keep (default: everything cached)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    runs = load_runs(args.cache_dir or ["probe_cache"], args.arms)
    fx = Path(args.fixture_dir)
    chip_all = np.load(fx / PIXELS_NPZ)["chip_idx"]
    bands = runs[0][0]["band_names"]
    print(f"{len(runs)} arm-seed cache(s), fixture {runs[0][0]['fixture_hash']}")

    rows, axrows, proj = [], [], []
    for m, x in runs:
        y = cache.labels_for(m, fx)
        chip = chip_all[chip_all < m["n_chips"]]
        w, v = own_frame(x, y)
        j = fisher(x, y, w)
        z, q = standardise(x @ w, y), standardise(x @ v, y)
        # The loading on the Z-SCORED band, i.e. the weight the U-Net's input
        # actually carries: raw loading x that checkpoint's own band_std.
        wz = w * np.asarray(m["band_std"]) / float(m.get("reflectance_scale", 1.0))
        wz /= np.linalg.norm(wz)
        rows.append({
            "run": m["run"], "arm": m["arm"], "seed": m["seed"],
            "epoch": m.get("epoch"), "n_pixels": m["n_pixels"],
            "fisher_own_frame": j, "dprime_own_frame": float(np.sqrt(2 * j)),
            "auc_own_frame": auc(z, y),
            "fisher_own_heldout": heldout_fisher(x, y, chip),
            "sr_only_cache": bool(m.get("sr_only", False)),
        })
        axrows.append({"run": m["run"], "arm": m["arm"], "seed": m["seed"],
                       **{f"ld1_{b}": c for b, c in zip(bands, w)},
                       **{f"ld1_zscored_{b}": c for b, c in zip(bands, wz)},
                       **{f"pcperp_{b}": c for b, c in zip(bands, v)}})
        proj.append((m, z, q, y))
    df = pd.DataFrame(rows)
    axes_df = pd.DataFrame(axrows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lda_own_fisher.csv", index=False)
    axes_df.to_csv(out / "lda_own_axes.csv", index=False)
    print("\n" + df[["arm", "seed", "fisher_own_frame", "fisher_own_heldout",
                     "dprime_own_frame", "auc_own_frame"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out / 'lda_own_fisher.csv'}, {out / 'lda_own_axes.csv'}")
    if args.no_figures:
        return 0

    style.apply_rc()
    import matplotlib.pyplot as plt

    n_arms = df["arm"].nunique()
    fig = plt.figure(figsize=(style.FULL_WIDTH_IN, max(3.4, 0.40 * n_arms + 1.6)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.08)
    arms, ticks = _ridgeline(fig.add_subplot(gs[0]), proj)
    _fisher_dots(fig.add_subplot(gs[1]), df, arms, ticks)
    for p in style.save(fig, out, "F1_lda_own"):
        print(f"wrote {p}")
    for p in figure_2d(proj, df, out, "A1_lda_own_2d"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
