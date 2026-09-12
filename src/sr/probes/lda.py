"""Instrument A — LDA in a shared frame, on the SR output (plan §3, figure F1).

    PYTHONPATH=src python -m sr.probes.lda --cache-dir probe_cache

Reads only the extraction cache. Emits, into `--out-dir` (default `figures/probes/`):

    lda_fisher.csv     one row per arm-seed x frame: the numbers behind F1
    lda_axes.csv       LD1 loadings over the 4 bands, per fitted axis
    F1_lda.pdf/.png    main text: LD1 ridgeline | Fisher slopegraph | loadings
    A1_lda_2d.pdf/.png appendix: (LD1, PC-perp) density panels on shared axes

WHY ONE FIXED FRAME, FITTED ON r0
---------------------------------
The SR->U-Net interface is image space, so band b is band b in every arm and a
single linear projection is meaningful across all of them — the property t-SNE
lacks and the reason this instrument is an LDA at all. The frame is fitted on
r0 because r0's SR path is parameter-free bicubic: its input pixels are
identical across seeds, so one fit is THE r0 frame rather than one seed's
version of it.

TWO FISHER RATIOS, AND WHY THE GAP IS THE FINDING
-------------------------------------------------
`r0_frame`  Fisher along r0's LD1 — how much more separable the output became
            along the direction that already separated the raw data.
`own_frame` Fisher along the arm's OWN refitted LD1 — how separable the output
            is in its own best direction.

`own >= r0_frame` always (the own axis is fitted to maximise exactly this), so
only the GAP is informative: a gain in both means the existing spectral
direction was sharpened; a gain confined to the own frame means the arm moved
the classes apart along a NEW direction, which is what §6.1 predicts for r2b if
its drift is affine and clutter-suppressive.

The ratio ((mu1-mu0).w)^2 / (var1 + var0) is invariant to ||w||, and invariant
to a global rescaling of the data but NOT to a per-band affine one — which is
the point: a per-band affine drift is a real change in the geometry the U-Net
is handed, and this statistic is meant to see it.

WHAT IT CANNOT SAY
------------------
This is a per-PIXEL, linear separability of road from background in 4 bands. A
higher Fisher ratio is not a higher IoU: roads are thin structures and the
U-Net's advantage is spatial, not pointwise. Read it as "the front-end handed
the network a cleaner spectral problem", never as a performance claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sr.probes import cache, style
from sr.probes.style import FIGURES_DIR
from sr.probes.make_fixtures import FIXTURE_DIR

RIDGE = 1e-9      # keeps S_W invertible if a band is ever constant


# ------------------------------------------------------------------- the maths
def within_class_scatter(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S_W as the UNWEIGHTED sum of the two class covariances, Sigma_1 + Sigma_0.

    Not the class-size-weighted pooled covariance, for two reasons that are the
    same reason:

    1. It is the denominator of the Fisher criterion this module reports,
       J(w) = ((mu_1 - mu_0).w)^2 / (var_1 + var_0). `lda_axis` solves for the
       maximiser of J, so the two must agree exactly — with the weighted form
       they do not, `own_frame` is no longer guaranteed to be the maximum, and
       the gap that carries the §3 claim can come out NEGATIVE.
    2. The 4:1 background:road ratio is a property of the fixture's SAMPLING
       DESIGN, not of the data. Weighting by class size would let a change of
       `--bg-per-road` move every arm's Fisher ratio; the unweighted form is
       invariant to it, so the numbers mean the same thing across fixtures.
    """
    c = x.shape[1]
    s = np.zeros((c, c))
    for cls in (False, True):
        d = x[y == cls]
        if len(d) > 1:
            s += np.cov(d.T, bias=True)
    return s + RIDGE * np.eye(c)


def lda_axis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit-norm two-class discriminant S_W^-1 (mu_road - mu_bg)."""
    w = np.linalg.solve(within_class_scatter(x, y),
                        x[y].mean(0) - x[~y].mean(0))
    n = np.linalg.norm(w)
    return w / n if n else w


def orthogonal_pc(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """First PC of the pooled within-class residual, orthogonalised against `w`.

    A second axis chosen for the PLOT only: it spreads the clouds so their shape
    is visible, without smuggling in a second discriminative direction — the
    residual is what is left after each class's own mean is removed.
    """
    r = x.copy()
    for cls in (False, True):
        r[y == cls] -= x[y == cls].mean(0)
    r -= np.outer(r @ w, w)                      # project out LD1
    _, _, vt = np.linalg.svd(r - r.mean(0), full_matrices=False)
    v = vt[0] - (vt[0] @ w) * w
    n = np.linalg.norm(v)
    return v / n if n else v


def fisher(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Two-class Fisher criterion of the projection x @ w."""
    p = x @ w
    a, b = p[y], p[~y]
    d = a.var() + b.var()
    return float((a.mean() - b.mean()) ** 2 / d) if d > 0 else float("nan")


# ------------------------------------------------------------------- the cache
def load_runs(cache_dir: Path, arms: list[str] | None):
    """[(meta, pixels)] for every extracted arm-seed, in figure order."""
    metas = cache.load_metas(cache_dir, arms, require=("sr_pixels.npy",))
    return [(m, np.load(m["dir"] / "sr_pixels.npy")) for m in metas]


# ------------------------------------------------------------------- figures
def _ridgeline(ax, runs, labels, w, bands):
    """LD1 densities per arm: background outlined, road filled. One row per arm.

    Every seed of an arm is drawn — for r0 the curves must coincide (its SR path
    is parameter-free bicubic, so the seeds differ only in a U-Net this
    instrument never touches), and for a joint arm the spread between them IS
    the within-arm noise floor the between-arm reading is judged against. The
    first seed carries the fill so the row stays legible; the rest are outlines.
    """
    by_arm = {}
    for (m, x), y in zip(runs, labels):
        by_arm.setdefault(m["arm"], []).append((m, x, y))
    arms = sorted(by_arm, key=style.sort_key)

    lo, hi = np.percentile(np.concatenate([x @ w for _, x in runs]), [0.2, 99.8])
    grid = np.linspace(lo, hi, 220)
    step = 1.0
    ticks = []

    for i, arm in enumerate(arms):
        base = (len(arms) - 1 - i) * step
        ticks.append(base + 0.35 * step)
        for j, (m, x, y) in enumerate(by_arm[arm]):
            for cls, colour in ((False, style.BG), (True, style.ROAD)):
                p = x[y == cls] @ w
                if p.size < 2:
                    continue
                d = _density(p, grid)
                d = d / d.max() * 0.9 * step
                if j == 0:
                    ax.fill_between(grid, base, base + d, color=colour,
                                    alpha=0.5 if cls else 0.0, lw=0)
                ax.plot(grid, base + d, color=colour,
                        lw=0.9 if j == 0 else 0.6, alpha=1.0 if j == 0 else 0.55)
        ax.axhline(base, color="0.85", lw=0.5, zorder=0)

    ax.set_yticks(ticks)
    ax.set_yticklabels([style.label(a) for a in arms])
    for t, a in zip(ax.get_yticklabels(), arms):
        t.set_color(style.color(a))
        t.set_fontsize(6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.05 * step, (len(arms) - 1) * step + 1.0 * step)
    ax.set_xlabel("LD1 (r0 frame)")
    ax.set_title("SR output along the r0 discriminant", loc="left")
    # The class encoding is local to this panel, so it is named here rather than
    # competing with the arm colours in the shared legend.
    ax.plot([], [], color=style.ROAD, lw=3, alpha=0.5, label="road")
    ax.plot([], [], color=style.BG, lw=0.9, label="background")
    # Above the axes, not inside: the densities fill the panel and where they
    # leave a gap is data-dependent, so an in-panel legend would land on a curve
    # as soon as the arm set changes.
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.06),
              fontsize=6.5)
    return arms


def _density(p, grid):
    """Gaussian KDE by hand — scipy's is not a dependency of this repo."""
    p = np.asarray(p, dtype="float64")
    h = 1.06 * p.std() * len(p) ** (-0.2) or 1e-6      # Silverman
    # Chunked so 40k background pixels x 220 grid points stays a small matrix.
    out = np.zeros_like(grid)
    for i in range(0, len(p), 20000):
        z = (grid[None, :] - p[i:i + 20000, None]) / h
        out += np.exp(-0.5 * z * z).sum(0)
    return out / (len(p) * h * np.sqrt(2 * np.pi))


def _slopegraph(ax, df):
    """r0-frame -> own-frame Fisher, one line per arm-seed. The slope IS the claim."""
    n_seeds = df.groupby("arm")["seed"].nunique()
    for arm, g in df.groupby("arm"):
        c, ls = style.color(arm), style.linestyle(arm)
        for _, r in g.iterrows():
            ax.plot([0, 1], [r["fisher_r0_frame"], r["fisher_own_frame"]],
                    color=c, ls=ls, lw=1.1, alpha=0.9, zorder=2)
            for xi, col in ((0, "fisher_r0_frame"), (1, "fisher_own_frame")):
                ax.plot([xi], [r[col]], color=c, zorder=3,
                        **style.marker_kwargs(int(n_seeds[arm])))
    # Arms whose own-frame Fisher nearly coincides would print their labels on
    # top of each other — r0 and r1a differ by 1e-4 here. Spread them along y by
    # a minimum gap, keeping their order, and leave the markers where they are.
    ends = sorted(((g["fisher_own_frame"].mean(), arm)
                   for arm, g in df.groupby("arm")), reverse=True)
    span = max(e for e, _ in ends) - min(e for e, _ in ends) if len(ends) > 1 else 1.0
    gap = 0.06 * (span or 1.0)
    ys = []
    for y, arm in ends:
        ys.append(min(y, ys[-1] - gap) if ys else y)
    for (y_raw, arm), y in zip(ends, ys):
        if abs(y - y_raw) > 1e-12:
            # Spread far enough to detach from its marker: draw the connector,
            # or the label reads as belonging to whichever line it landed near.
            ax.plot([1.02, 1.075], [y_raw, y], color=style.color(arm), lw=0.5,
                    alpha=0.7, clip_on=False, zorder=1)
        ax.annotate(arm, (1.09, y), color=style.color(arm), fontsize=6.5,
                    va="center", ha="left", annotation_clip=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["r0 frame", "own frame"])
    ax.set_xlim(-0.15, 1.45)
    ax.set_ylabel("Fisher ratio (road vs background)")
    ax.set_title("Shared frame → own frame", loc="left")


def _loadings(ax, axes_df, bands):
    """LD1 loadings over the bands — what spectral direction the axis names."""
    arms = sorted(axes_df["arm"].unique(), key=style.sort_key)
    w = 0.8 / max(len(arms), 1)
    xs = np.arange(len(bands))
    for i, arm in enumerate(arms):
        v = axes_df[axes_df["arm"] == arm].iloc[0][bands].to_numpy(dtype=float)
        ax.bar(xs + i * w - 0.4 + w / 2, v, width=w, color=style.color(arm),
               label=arm, linewidth=0, hatch=style.hatch(arm),
               edgecolor="white")
    ax.axhline(0, color=style.ZERO_LINE, lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(bands)
    ax.set_ylabel("LD1 loading", labelpad=1)
    ax.set_title("Spectral direction (own frame)", loc="left")


def figure_2d(runs, labels, w, v, out_dir, stem):
    """Appendix: (LD1, PC-perp) density panels per arm, on SHARED axes."""
    import matplotlib.pyplot as plt

    by_arm = {}
    for (m, x), y in zip(runs, labels):
        by_arm.setdefault(m["arm"], (m, x, y))
    arms = sorted(by_arm, key=style.sort_key)
    ncol = min(len(arms), 4)
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.FULL_WIDTH_IN,
                                                 2.2 * nrow), squeeze=False)
    allp = np.concatenate([x @ np.column_stack([w, v]) for _, x in runs])
    xlim = np.percentile(allp[:, 0], [0.2, 99.8])
    ylim = np.percentile(allp[:, 1], [0.2, 99.8])

    for ax, arm in zip(axes.ravel(), arms):
        m, x, y = by_arm[arm]
        p = x @ np.column_stack([w, v])
        for cls, colour in ((False, style.BG), (True, style.ROAD)):
            ax.scatter(p[y == cls, 0], p[y == cls, 1], s=1.5, alpha=0.25,
                       color=colour, linewidths=0, rasterized=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(style.label(arm), fontsize=7, loc="left",
                     color=style.color(arm))
    for ax in axes.ravel()[len(arms):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("LD1 (r0 frame)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC⊥")
    fig.suptitle("SR output in the shared LDA frame (road red, background grey)",
                 fontsize=8, y=1.0)
    fig.tight_layout()
    return style.save(fig, out_dir, stem)


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="probe_cache")
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--out-dir", default=FIGURES_DIR)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm-key globs to keep, e.g. r0 r1a r1b 'r2a@*' 'r2b@*' "
                         "(default: everything cached)")
    ap.add_argument("--fit-arm", default="r0",
                    help="arm supplying the shared frame; r0 by design (§3)")
    ap.add_argument("--no-figures", action="store_true",
                    help="write the CSVs only")
    args = ap.parse_args(argv)

    runs = load_runs(Path(args.cache_dir), args.arms)
    fx = Path(args.fixture_dir)
    labels = [cache.labels_for(m, fx) for m, _ in runs]
    bands = runs[0][0]["band_names"]
    print(f"{len(runs)} arm-seed cache(s), fixture {runs[0][0]['fixture_hash']}, "
          f"{runs[0][0]['n_pixels']} pixels")

    fits = [(m, x, y) for (m, x), y in zip(runs, labels) if m["arm"] == args.fit_arm]
    if not fits:
        raise SystemExit(
            f"no {args.fit_arm} cache to fit the shared frame on. The frame must "
            "come from the un-adapted anchor, so extract that arm first (or pass "
            "--fit-arm to state a deliberate substitute).")
    fit_m, fit_x, fit_y = fits[0]
    w = lda_axis(fit_x, fit_y)
    v = orthogonal_pc(fit_x, fit_y, w)
    print(f"shared frame from {fit_m['run']}: LD1 = "
          + ", ".join(f"{b} {c:+.3f}" for b, c in zip(bands, w)))
    if len(fits) > 1:
        # r0's SR path is parameter-free, so its seeds must project identically;
        # if they do not, something arm-specific leaked into the extraction.
        d = max(float(np.abs(x - fit_x).max()) for _, x, _ in fits[1:])
        print(f"  {len(fits)} {args.fit_arm} seeds agree to {d:.2e} in pixel value"
              + ("" if d < 1e-5 else "  <-- WARNING: r0 seeds should be identical"))

    rows, axrows = [], []
    for (m, x), y in zip(runs, labels):
        own = lda_axis(x, y)
        # Sign is arbitrary in an eigen-problem; pin it so road is on the
        # positive side and the loadings of different arms are comparable.
        if (x[y] @ own).mean() < (x[~y] @ own).mean():
            own = -own
        rows.append({
            "run": m["run"], "arm": m["arm"], "seed": m["seed"],
            "n_pixels": m["n_pixels"], "n_chips": m["n_chips"],
            "fisher_r0_frame": fisher(x, y, w),
            "fisher_own_frame": fisher(x, y, own),
            "theta_provenance": m.get("theta_provenance"),
        })
        axrows.append({"run": m["run"], "arm": m["arm"], "seed": m["seed"],
                       **dict(zip(bands, own))})
    df = pd.DataFrame(rows)
    df["fisher_gap"] = df["fisher_own_frame"] - df["fisher_r0_frame"]
    axes_df = pd.DataFrame(axrows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lda_fisher.csv", index=False)
    axes_df.to_csv(out / "lda_axes.csv", index=False)
    print("\n" + df[["arm", "seed", "fisher_r0_frame", "fisher_own_frame",
                     "fisher_gap"]].to_string(index=False,
                                              float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out / 'lda_fisher.csv'}, {out / 'lda_axes.csv'}")
    if args.no_figures:
        return 0

    style.apply_rc()
    import matplotlib.pyplot as plt

    # One ridgeline row per arm, so the figure has to grow with the arm set —
    # eleven configs in the 3.4in box built for four would overlap into mush.
    n_arms = df["arm"].nunique()
    fig = plt.figure(figsize=(style.FULL_WIDTH_IN,
                              max(3.4, 0.40 * n_arms + 1.6)))
    # TWO panels in the main text: the ridgeline and the slopegraph. The
    # per-band loadings panel was cut — it answered a different question from
    # the other two (which bands the axis leans on, rather than whether the
    # axis separates at all), and its numbers survive verbatim in
    # `lda_axes.csv`. `_loadings` is kept below so it can be promoted to an
    # appendix figure without rewriting it.
    #
    # Dropping it also removes the figure-level arm legend, which was built
    # from that panel's handles — deliberately, not by oversight: both
    # surviving panels label their arms in place (a row label per arm in the
    # ridgeline, leader-lined names at the slopegraph's right edge), so a
    # legend would have restated identity the reader already has.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.55)
    _ridgeline(fig.add_subplot(gs[0]), runs, labels, w, bands)
    _slopegraph(fig.add_subplot(gs[1]), df)
    for p in style.save(fig, out, "F1_lda"):
        print(f"wrote {p}")
    for p in figure_2d(runs, labels, w, v, out, "A1_lda_2d"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
