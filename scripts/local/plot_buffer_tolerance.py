#!/usr/bin/env python
"""Plot the buffered-F1 TOLERANCE sweep already recorded by every R-series run.

WHERE THE DATA COMES FROM — NOTHING IS RECOMPUTED
-------------------------------------------------
Two sources, both read-only:

`--runs-dir` — `sr/_stages_tv.sh` ends its fit stage with a "TEST θ SENSITIVITY
SWEEP" at a hardcoded `--buffer-px 1,2,3,4,5`, writing `<run>/test_sweep.json`:
19 θ x {iou, f1, buffered_{precision,recall,f1}_r1..r5} on the TEST split, for a
single seed. Carries precision/recall, so it also draws panel C.

`--report` — a benchmark report markdown (the `## <metric> (macro|micro, ...)`
mean/std/n_seeds tables written by the final bench). Seed-aggregated, so it gets
error bars; it carries no buffered precision/recall, so it draws panels A and B
only. The aggregation and its unit are the report's — this script does not
re-aggregate anything, and states neither on the figure.

WHY ρ = 0 IS THE STRICT F1
--------------------------
`buffered_metrics` relaxes position by ρ px of the 2.5 m grid: precision counts
predicted road within ρ of GT, recall counts GT within ρ of prediction. At
ρ = 0 that is exactly strict pixel F1, which the same sweep entry records as
`f1` — so plotting it as the ρ = 0 point is the real curve, not an extrapolated
one.

WHAT THE CURVE ANSWERS
----------------------
"Where does the error live?" A steep first step means the model found the road
and put it a pixel or two off — registration error, and at 2.5 m GSD with
labels digitised from another source, partly the LABEL's error. A flat curve
that never reaches 1 means roads genuinely missed, which no tolerance recovers.
The precision/recall panel says which side the residual sits on.

    python scripts/local/plot_buffer_tolerance.py --runs-dir <SRruns>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Categorical slots 1-5 of the validated reference palette, in slot order.
# Lines and grouped bars are validated on the ADJACENT pairlist, which this
# order clears in both modes through all eight slots — the three-slot cap only
# binds for all-pairs forms (scatter, choropleth, small multiples).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
RADII = (1, 2, 3, 4, 5)


class Arm:
    """One plotted arm. `err`/`pr`/`rc` are None when the source lacks them."""

    def __init__(self, name, f1, err=None, pr=None, rc=None, theta=None):
        self.name, self.f1, self.err = name, f1, err
        self.pr, self.rc, self.theta = pr, rc, theta

    @property
    def legend(self) -> str:
        return self.name if self.theta is None else f"{self.name}  (θ*={self.theta:g})"


def read_run(run: Path, name: str) -> Arm:
    """F1[ρ=0..5] + precision/recall[ρ=1..5] at that run's θ*."""
    j = json.loads((run / "test_sweep.json").read_text())
    th = str(j["best_threshold"])
    e = j["sweep"][th]
    return Arm(name,
               [e["f1"]] + [e[f"buffered_f1_r{r}"] for r in RADII],
               pr=[e[f"buffered_precision_r{r}"] for r in RADII],
               rc=[e[f"buffered_recall_r{r}"] for r in RADII],
               theta=float(th))


def read_report(path: Path, agg: str):
    """Parse `## <metric> (<agg>, per-<unit>)` mean/std tables out of a report md.

    Returns {metric: {model: (mean, std)}} for the metrics this figure needs.
    """
    want = ["f1"] + [f"buffered_f1_r{r}" for r in RADII]
    out, metric = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            m = re.match(r"#+\s+(\S+)\s+\((\w+), per-\w+\)$", line)
            metric = m.group(1) if m and m.group(2) == agg and m.group(1) in want else None
            continue
        if metric is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("model",) or set(cells[0]) <= set("- "):
            continue
        out.setdefault(metric, {})[cells[0]] = (float(cells[1]), float(cells[2]))
    missing = [m for m in want if m not in out]
    if missing:
        raise SystemExit(f"{path}: no {agg} table for {', '.join(missing)}")
    return out


def shorten(names):
    """Drop the underscore-token prefix and suffix every arm name shares."""
    parts = [n.split("_") for n in names]
    if len(parts) == 1:
        return [names[0]]
    head = 0
    while all(p[head] == parts[0][head] for p in parts) and head + 1 < min(map(len, parts)):
        head += 1
    tail = 0
    while (all(p[-1 - tail] == parts[0][-1 - tail] for p in parts)
           and head + tail + 1 < min(map(len, parts))):
        tail += 1
    trimmed = [p[head:len(p) - tail] if tail else p[head:] for p in parts]
    # `new` survives the affix trim (it does not sit in a shared run) but
    # carries nothing — every arm here is a `new`-dataset run.
    return ["_".join([t for t in toks if t != "new"]) or orig
            for toks, orig in zip(trimmed, names)]


def finish(fig, args, arms, rho) -> int:
    """Save the figure and its table view. No suptitle — the caption is written
    where the figure is used, so a heading here would only duplicate it."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {out}")

    # A table view is not optional: near-identical curves are exactly the case
    # where a reader needs the numbers to see the ordering.
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "theta_star", "rho_px", "tolerance_m",
                     "buffered_f1", "buffered_f1_sd", "delta_f1",
                     "buffered_precision", "buffered_recall"])
        for arm in arms:
            for r in rho:
                wr.writerow([arm.name, "" if arm.theta is None else arm.theta,
                             r, r * 2.5, f"{arm.f1[r]:.4f}",
                             "" if arm.err is None else f"{arm.err[r]:.4f}",
                             "" if r == 0 else f"{arm.f1[r] - arm.f1[r - 1]:.4f}",
                             "" if r == 0 or arm.pr is None else f"{arm.pr[r - 1]:.4f}",
                             "" if r == 0 or arm.rc is None else f"{arm.rc[r - 1]:.4f}"])
    print(f"  -> {csv_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs-dir", help="dir of run dirs holding test_sweep.json")
    src.add_argument("--report", help="benchmark report .md of mean/std tables")
    ap.add_argument("--agg", default="macro", choices=("macro", "micro"),
                    help="--report only: which aggregation's tables to read "
                         "(default macro)")
    ap.add_argument("--runs", action="append", default=None,
                    help="run/model names, in plot order (default: all of them, "
                         "sorted)")
    ap.add_argument("--out", default="figures/buffer_tolerance.png")
    args = ap.parse_args(argv)

    if args.report:
        tables = read_report(Path(args.report), args.agg)
        names = args.runs or sorted(tables["f1"])
        missing = [n for n in names if any(n not in t for t in tables.values())]
        if missing:
            raise SystemExit(f"not in every table: {', '.join(missing)}")
        keys = ["f1"] + [f"buffered_f1_r{r}" for r in RADII]
        arms = [Arm(s, [tables[k][n][0] for k in keys],
                    err=[tables[k][n][1] for k in keys])
                for n, s in zip(names, shorten(names))]
    else:
        root = Path(args.runs_dir)
        names = args.runs or sorted(
            d.name for d in root.iterdir() if (d / "test_sweep.json").is_file())
        arms = [read_run(root / n, s) for n, s in zip(names, shorten(names))]

    if len(arms) > len(SERIES):
        raise SystemExit(
            f"{len(arms)} arms but only {len(SERIES)} validated categorical "
            "slots — split into two figures rather than generating a hue.")
    # Panel C needs buffered precision/recall, which only the sweep JSONs carry.
    npanel = 3 if arms[0].pr is not None else 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # No figure heading, and no aggregation/unit wording on the axes — the
    # caption states what the numbers are.
    ylab = "F1"
    sub_a = sub_b = ""

    rho = list(range(6))
    fig, axes = plt.subplots(1, npanel, figsize=(5.0 * npanel, 4.5),
                             facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=0)

    # --- A: the curve ------------------------------------------------------
    a = axes[0]
    for i, arm in enumerate(arms):
        a.plot(rho, arm.f1, color=SERIES[i], linewidth=2.0, marker="o",
               markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.4,
               label=arm.legend, zorder=3)
        if arm.err is not None:
            # ±1 SD over seeds. The arms sit within ~0.01 of each other, which
            # is the same order as the spread — drawing it is what keeps the
            # eye from reading the ordering as settled.
            a.errorbar(rho, arm.f1, yerr=arm.err, fmt="none", ecolor=SERIES[i],
                       elinewidth=1.2, capsize=2.5, capthick=1.2, alpha=0.8,
                       zorder=2)
    a.set_title("Buffered F1 vs position tolerance" + sub_a, fontsize=11,
                color=INK, pad=10, loc="left")
    a.set_xlabel("tolerance ρ  (px of the 2.5 m grid)", fontsize=9.5, color=INK2)
    a.set_ylabel(ylab, fontsize=9.5, color=INK2)
    a.set_xticks(rho)
    a.set_xticklabels(["0\n(strict)", "1\n2.5 m", "2\n5 m", "3\n7.5 m",
                       "4\n10 m", "5\n12.5 m"], fontsize=8.5)
    # Upper-left: the curve rises left-to-right, so that corner is the only
    # region of panel A that never carries a mark.
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

    # The headline: how much of everything tolerance can recover arrives in the
    # FIRST pixel. Annotated from the first series, and stated as a share so it
    # does not read as an absolute claim about the other two.
    f1_0 = arms[0].f1
    frac = (f1_0[1] - f1_0[0]) / (f1_0[-1] - f1_0[0])
    a.annotate("", xy=(0, f1_0[0]), xytext=(0, f1_0[1]),
               arrowprops=dict(arrowstyle="<->", color=INK3, lw=1.2))
    # Parked in the empty lower-middle, clear of both the rising curve and the
    # legend, with a leader back to the arrow. Placed in axes fractions so it
    # lands in the same empty corner whatever the y-range of the source.
    lo, hi = a.get_ylim()
    a.annotate(f"first pixel alone: {frac:.0%} of all\nthe F1 tolerance can recover",
               xy=(0.06, (f1_0[0] + f1_0[1]) / 2), xytext=(1.7, lo + 0.16 * (hi - lo)),
               fontsize=8.4, color=INK2, va="center", linespacing=1.5,
               arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8,
                               shrinkA=2, shrinkB=2))

    # --- B: marginal gain --------------------------------------------------
    b = axes[1]
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        d = [arm.f1[r] - arm.f1[r - 1] for r in range(1, 6)]
        # 2 px surface gap between adjacent bars, per the mark spec.
        b.bar([r + (i - (len(arms) - 1) / 2) * w for r in RADII], d,
              width=w * 0.92, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5,
              label=arm.name, zorder=3)
    b.set_title("Marginal F1 gained per extra pixel of tolerance" + sub_b,
                fontsize=11, color=INK, pad=10, loc="left")
    b.set_xlabel("tolerance step  ρ−1 → ρ", fontsize=9.5, color=INK2)
    b.set_ylabel(f"Δ {ylab}", fontsize=9.5, color=INK2)
    b.set_xticks(list(RADII))
    b.legend(frameon=False, fontsize=8.5, labelcolor=INK2)

    # --- C: which side the residual sits on --------------------------------
    if npanel < 3:
        return finish(fig, args, arms, rho)
    c = axes[2]
    for i, arm in enumerate(arms):
        c.plot(RADII, arm.pr, color=SERIES[i], linewidth=2.0, linestyle="--",
               marker="s", markersize=5, markeredgecolor=SURFACE,
               markeredgewidth=1.2, zorder=3)
        c.plot(RADII, arm.rc, color=SERIES[i], linewidth=2.0, marker="o",
               markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.4,
               zorder=3)
    # Identity is carried by the dash pattern here, not by colour, so the key
    # is drawn in ink rather than in any series hue.
    from matplotlib.lines import Line2D
    c.legend(handles=[Line2D([], [], color=INK2, lw=2, ls="--", marker="s",
                             markersize=5, label="buffered precision"),
                      Line2D([], [], color=INK2, lw=2, marker="o",
                             markersize=5.5, label="buffered recall")],
             frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    c.set_title("Precision rises faster than recall\n"
                "→ the residual error is MISSED road, not hallucinated road",
                fontsize=11, color=INK, pad=10, loc="left")
    c.set_xlabel("tolerance ρ  (px)", fontsize=9.5, color=INK2)
    c.set_ylabel("buffered precision / recall", fontsize=9.5, color=INK2)
    c.set_xticks(list(RADII))

    return finish(fig, args, arms, rho)


if __name__ == "__main__":
    raise SystemExit(main())
