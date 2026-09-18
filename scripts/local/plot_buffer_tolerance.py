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

ENCODING IS BORROWED, NOT INVENTED
-----------------------------------
Colour, linestyle, label and ordering all come from `sr.probes.style`, so this
figure says the same thing with the same ink as the probe figures: hue is
generator provenance (grey bicubic / blue SEN2SR / green SR4RS), lightness is
frozen-vs-jointly-tuned, and solid/dashed is the FFT hard constraint. That
scheme has a channel per factor of the R-series design, which is why it scales
past the fixed 5-colour categorical palette this script used to cap itself at.

    python scripts/local/plot_buffer_tolerance.py --runs-dir <SRruns>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RADII = (1, 2, 3, 4, 5)


class Arm:
    """One plotted arm. `err`/`pr`/`rc` are None when the source lacks them.

    `name` is the raw run/model name — the string `sr.probes.style` keys its
    colour, linestyle and label off (`style.arm_of` parses the `rN[ab]` token
    out of it), not a display string.
    """

    def __init__(self, name, f1, err=None, pr=None, rc=None, theta=None):
        self.name, self.f1, self.err = name, f1, err
        self.pr, self.rc, self.theta = pr, rc, theta

    def legend(self, style) -> str:
        lab = style.label(self.name)
        return lab if self.theta is None else f"{lab}  (θ*={self.theta:g})"


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


def read_csv(path: Path):
    """Read back a table this script itself wrote (`<out>.csv`).

    The table view carries exactly what the curve panel plots — arm, F1 and its
    SD at every ρ — so a figure can be re-laid-out (fewer panels, legend moved)
    without going back to the report or the run dirs, and without any risk of
    re-plotting different numbers than the original figure showed.
    """
    rows = list(csv.DictReader(path.open()))
    arms, order = {}, []
    for r in rows:
        name = r["arm"]
        if name not in arms:
            arms[name] = ({}, {}, r["theta_star"])
            order.append(name)
        f1, err, _ = arms[name]
        f1[int(r["rho_px"])] = float(r["buffered_f1"])
        if r["buffered_f1_sd"]:
            err[int(r["rho_px"])] = float(r["buffered_f1_sd"])
    out = []
    for name in order:
        f1, err, th = arms[name]
        rho = sorted(f1)
        out.append(Arm(name, [f1[r] for r in rho],
                       err=[err[r] for r in rho] if len(err) == len(f1) else None,
                       theta=float(th) if th else None))
    return out


def finish(fig, args, arms, rho) -> int:
    """Save the figure (PDF + PNG, via `style.save`) and its table view. No
    suptitle — the caption is written where the figure is used, so a heading
    here would only duplicate it."""
    import matplotlib.pyplot as plt
    from sr.probes import style

    fig.tight_layout()
    out = Path(args.out)
    paths = style.save(fig, out.parent, out.stem)
    plt.close(fig)
    for p in paths:
        print(f"  -> {p}")

    # A table view is not optional: near-identical curves are exactly the case
    # where a reader needs the numbers to see the ordering.
    csv_path = out.parent / f"{out.stem}.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "theta_star", "rho_px", "tolerance_m",
                     "buffered_f1", "buffered_f1_sd", "delta_f1",
                     "buffered_precision", "buffered_recall"])
        for arm in arms:
            for r in rho:
                wr.writerow([style.arm_of(arm.name),
                             "" if arm.theta is None else arm.theta,
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
    src.add_argument("--csv", help="re-plot a table this script wrote earlier")
    ap.add_argument("--agg", default="macro", choices=("macro", "micro"),
                    help="--report only: which aggregation's tables to read "
                         "(default macro)")
    ap.add_argument("--runs", action="append", default=None,
                    help="run/model names, in plot order (default: all of them, "
                         "sorted)")
    ap.add_argument("--panels", default="all", choices=("all", "curve"),
                    help="'curve' keeps only the tolerance curve (panel A) and "
                         "moves the legend under it")
    ap.add_argument("--legend-gap", type=float, default=0.18,
                    help="--panels curve only: axes-fraction gap between the "
                         "panel and the legend below it")
    ap.add_argument("--out", default="figures/buffer_tolerance.png")
    args = ap.parse_args(argv)

    from sr.probes import style

    if args.csv:
        arms = read_csv(Path(args.csv))
        if args.runs:
            keep = {style.arm_of(n) for n in args.runs}
            arms = [a for a in arms if style.arm_of(a.name) in keep]
    elif args.report:
        tables = read_report(Path(args.report), args.agg)
        names = args.runs or sorted(tables["f1"], key=style.sort_key)
        missing = [n for n in names if any(n not in t for t in tables.values())]
        if missing:
            raise SystemExit(f"not in every table: {', '.join(missing)}")
        keys = ["f1"] + [f"buffered_f1_r{r}" for r in RADII]
        arms = [Arm(n, [tables[k][n][0] for k in keys],
                    err=[tables[k][n][1] for k in keys])
                for n in names]
    else:
        root = Path(args.runs_dir)
        names = args.runs or sorted(
            (d.name for d in root.iterdir() if (d / "test_sweep.json").is_file()),
            key=style.sort_key)
        arms = [read_run(root / n, n) for n in names]

    # Panel C needs buffered precision/recall, which only the sweep JSONs carry.
    npanel = 1 if args.panels == "curve" else (3 if arms[0].pr is not None else 2)

    style.apply_rc()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # No figure heading, and no aggregation/unit wording on the axes — the
    # caption states what the numbers are.
    ylab = "F1"
    label_fs = mpl.rcParams["axes.labelsize"] * 1.3
    tick_fs = mpl.rcParams["xtick.labelsize"] * 1.2

    # Shape carries only the baseline: r0 has no a/b pairing to read off a
    # dash, so it keeps a distinct marker. Every other arm is a round marker —
    # solid vs. dashed line already says HC on/off, and repeating that in the
    # marker shape (as `style.marker` does) would just be the same bit twice.
    def marker_for(name):
        return "D" if style.arm_of(name) == "r0" else "o"

    # Line proxies, not bar patches, so the dash-vs-solid convention of the
    # curve panel is what the reader learns to read rather than the bar panel's
    # hatch, which is its own separate encoding of the same on/off split.
    from matplotlib.lines import Line2D

    def legend_handle(arm):
        ls = style.linestyle(arm.name)
        h = Line2D([], [], color=style.color(arm.name), linestyle=ls,
                   marker=marker_for(arm.name), markersize=5.5,
                   linewidth=1.6, label=arm.legend(style))
        if ls == "--":
            # The default dash period is too long for a ~2.5-point legend
            # swatch — it can render as a single dash, i.e. indistinguishable
            # from solid. Tighter on/off (in points) keeps 2-3 dashes visible
            # at that length.
            h.set_dashes((2, 1.2))
        return h

    rho = list(range(6))
    # Alone, the curve panel gets the full text width rather than one panel's
    # 5 in: its six "ρ (m)" tick labels are what set the minimum width, and at
    # 5 in they overlap once there is no neighbouring panel to be read against.
    width = style.FULL_WIDTH_IN if npanel == 1 else 5.0 * npanel
    fig, axes = plt.subplots(1, npanel, figsize=(width, 4.5), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.tick_params(labelsize=tick_fs)

    # --- A: the curve ------------------------------------------------------
    a = axes[0]
    for arm in arms:
        c, ls = style.color(arm.name), style.linestyle(arm.name)
        a.plot(rho, arm.f1, color=c, linestyle=ls, linewidth=1.6,
               marker=marker_for(arm.name), markersize=5.5, zorder=3)
        if arm.err is not None:
            # ±1 SD over seeds. The arms sit within ~0.01 of each other, which
            # is the same order as the spread — drawing it is what keeps the
            # eye from reading the ordering as settled.
            a.errorbar(rho, arm.f1, yerr=arm.err, fmt="none", ecolor=c,
                       elinewidth=1.0, capsize=2.5, capthick=1.0, alpha=0.8,
                       zorder=2)
    a.set_title("Buffered F1 vs position tolerance", loc="left")
    a.set_xlabel("tolerance ρ", fontsize=label_fs)
    a.set_ylabel(ylab, fontsize=label_fs)
    a.set_xticks(rho)
    a.set_xticklabels(["0 (0 m)", "1 (2.5 m)", "2 (5 m)", "3 (7.5 m)",
                       "4 (10 m)", "5 (12.5 m)"])
    # One legend for the whole figure, not one per panel — it is built after
    # panel B below and parked beside it, since B is where the reader's eye
    # already goes for a comparison across all arms.

    # The headline: how much of everything tolerance can recover arrives in the
    # FIRST pixel. Annotated from the first series, and stated as a share so it
    # does not read as an absolute claim about the other two.
    f1_0 = arms[0].f1
    frac = (f1_0[1] - f1_0[0]) / (f1_0[-1] - f1_0[0])
    a.annotate("", xy=(0, f1_0[0]), xytext=(0, f1_0[1]),
               arrowprops=dict(arrowstyle="<->", color=style.BLACK, lw=1.2))
    # Parked in the empty lower-middle, clear of both the rising curve and the
    # legend, with a leader back to the arrow. Placed in axes fractions so it
    # lands in the same empty corner whatever the y-range of the source.
    lo, hi = a.get_ylim()
    a.annotate(f"1 px recovers {frac:.0%} of the total\ngain available by 5 px",
               xy=(0.06, (f1_0[0] + f1_0[1]) / 2), xytext=(1.7, lo + 0.16 * (hi - lo)),
               fontsize=9, color=style.BLACK, va="center", linespacing=1.5,
               arrowprops=dict(arrowstyle="-", color=style.BLACK, lw=0.8,
                               shrinkA=2, shrinkB=2))

    # Alone, panel A has no neighbouring dead space to park a key in, and its
    # own is taken by the annotation — so the legend goes under the axes, one
    # `--legend-gap` of axes height below them, in three columns so the nine
    # arms stay on three short rows rather than one wide one.
    if npanel == 1:
        a.legend(handles=[legend_handle(arm) for arm in arms],
                 loc="upper center", bbox_to_anchor=(0.5, -args.legend_gap),
                 ncol=3, handlelength=2.6, columnspacing=1.0,
                 fontsize=mpl.rcParams["legend.fontsize"] * 1.2,
                 labelspacing=0.5 * 1.1)
        return finish(fig, args, arms, rho)

    # --- B: marginal gain --------------------------------------------------
    b = axes[1]
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        d = [arm.f1[r] - arm.f1[r - 1] for r in range(1, 6)]
        # HC on/off reads as solid colour vs. white stripes over the same
        # colour — `style.hatch` already returns None for "a" (HC on) and
        # "///" for "b" (HC off); drawing that hatch in white (edgecolor)
        # with no border keeps the fill colour as the identity channel.
        b.bar([r + (i - (len(arms) - 1) / 2) * w for r in RADII], d,
              width=w * 0.92, color=style.color(arm.name),
              hatch=style.hatch(arm.name), edgecolor="white", linewidth=0,
              zorder=3)
    b.set_title("Marginal F1 gained per extra pixel of tolerance", loc="left")
    b.set_xlabel("tolerance ρ", fontsize=label_fs)
    b.set_ylabel(f"Δ {ylab}", fontsize=label_fs)
    b.set_xticks(list(RADII))
    b.set_xticklabels(["1 (2.5 m)", "2 (5 m)", "3 (7.5 m)",
                       "4 (10 m)", "5 (12.5 m)"])

    # The one legend for the figure, parked in panel B's own dead space
    # (upper right — the marginal gain is smallest at ρ=4,5, so no bar there
    # reaches high enough to collide with it). Line proxies, not bar patches,
    # so the dash-vs-solid convention from panel A is what the reader learns
    # to read rather than the hatch, which is B's own separate encoding of
    # the same on/off split.
    b.legend(handles=handles, loc="upper right", ncol=1, handlelength=2.6,
             columnspacing=1.0, fontsize=mpl.rcParams["legend.fontsize"] * 1.2,
             labelspacing=0.5 * 1.1)

    # --- C: which side the residual sits on --------------------------------
    if npanel < 3:
        return finish(fig, args, arms, rho)
    c = axes[2]
    for arm in arms:
        clr = style.color(arm.name)
        c.plot(RADII, arm.pr, color=clr, linewidth=1.6, linestyle="--",
               marker="s", markersize=5, zorder=3)
        c.plot(RADII, arm.rc, color=clr, linewidth=1.6, marker="o",
               markersize=5.5, zorder=3)
    # Identity is carried by the dash pattern here, not by colour, so the key
    # is drawn in ink rather than in any series hue.
    from matplotlib.lines import Line2D
    c.legend(handles=[Line2D([], [], color=style.GREY, lw=2, ls="--", marker="s",
                             markersize=5, label="buffered precision"),
                      Line2D([], [], color=style.GREY, lw=2, marker="o",
                             markersize=5.5, label="buffered recall")],
             loc="lower right")
    c.set_title("Precision rises faster than recall\n"
                "→ the residual error is MISSED road, not hallucinated road",
                loc="left")
    c.set_xlabel("tolerance ρ", fontsize=label_fs)
    c.set_ylabel("buffered precision / recall", fontsize=label_fs)
    c.set_xticks(list(RADII))

    return finish(fig, args, arms, rho)


if __name__ == "__main__":
    raise SystemExit(main())
