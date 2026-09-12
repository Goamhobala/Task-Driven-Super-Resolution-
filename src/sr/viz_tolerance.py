"""Where the error lives: buffered-F1 tolerance anatomy over the R-arms.

    PYTHONPATH=src python -m sr.viz_tolerance --runs-dir <SRruns> --runs-dir <SRruns/refits>

Two panels off the `test_sweep.json` files that already exist -- no compute.

**A -- error anatomy.** One stacked bar per arm: strict F1 at that arm's theta*,
then the increments buffered F1 recovers at rho = 1, 2 and 3-5 px, and the
remainder to 1.0 as the structural band -- misses and spurious segments no
tolerance forgives. Tolerance is an ORDERED factor, so it gets one sequential
ramp plus grey for the structural remainder, never categorical hues.

**B -- advantage over r0 across rho.** The slope is the finding: a gap that
shrinks as tolerance grows is positional precision (it evaporates once
placement is forgiven); a gap that stays flat is structural -- road found that
bicubic never finds at any tolerance.

THE THETA* RULE THIS SCRIPT EXISTS TO NOT BREAK
-----------------------------------------------
theta* is re-argmaxed from each run's **val** `sweep.json` and then LOOKED UP in
its `test_sweep.json`. Taking the argmax on the test sweep instead would select
the operating point on the same split it is scored on, which is the one thing
the theta protocol forbids. `resolve_theta` is the same helper the probes and
the bench use, so all three agree on what an arm's operating point is.

Arms are pooled over seeds; r0's own cross-seed spread is drawn as the grey
band at zero in panel B, because that is the noise floor any Delta must clear.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sr.probes import style

# Buffered F1 is monotone in rho, so these group into disjoint increments that
# sum with the strict F1 to bf1_r5, and the remainder to 1.0 is what tolerance
# never recovers.
BANDS = [("strict", None), ("±1 px", 1), ("±2 px", 2), ("±3–5 px", 5)]
RHOS = (0, 1, 2, 3, 4, 5)
# Sequential ramp for the ordered tolerance factor (ColorBrewer Blues), then
# grey for the structural remainder -- which is not a tolerance at all.
RAMP = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]
STRUCT = "#b0b0b0"
ERA = "*gap_ce_anorm_recalpost*"


def f1_at(entry, rho: int) -> float:
    return float(entry["f1"] if rho == 0 else entry[f"buffered_f1_r{rho}"])


def load_runs(runs_dirs, include, select_on, default_theta):
    """[(arm, run, theta, provenance, {rho: F1})] for every run with both sweeps."""
    import fnmatch

    from sr.viz_models import resolve_theta

    out = []
    for d in runs_dirs:
        for run in sorted(Path(d).iterdir()):
            if not run.is_dir() or not fnmatch.fnmatch(run.name, include):
                continue
            ts = run / "test_sweep.json"
            if not ts.is_file() or not (run / "sweep.json").is_file():
                continue
            theta, note = resolve_theta(run, select_on, default_theta)
            prov = ("sweep" if note.startswith(select_on)
                    else "recorded" if note == "recorded θ*" else "fallback")
            sweep = json.loads(ts.read_text())["sweep"]
            # The val argmax must exist on the test grid; both sweeps are
            # written on the same theta ladder, so a miss is a real mismatch
            # rather than something to round away silently.
            key = min(sweep, key=lambda k: abs(float(k) - theta))
            if abs(float(key) - theta) > 1e-6:
                print(f"  WARN {run.name}: val θ*={theta:g} not on the test grid; "
                      f"nearest is {key} — skipping")
                continue
            curve = {r: f1_at(sweep[key], r) for r in RHOS}
            bad = [r for r in RHOS[1:] if curve[r] < curve[r - 1] - 1e-9]
            if bad:
                print(f"  WARN {run.name}: buffered F1 not monotone at ρ={bad}")
            out.append((style.arm_of(run.name), run.name, theta, prov, curve))
    return out


def aggregate(rows, arms_keep=None):
    """arm -> (n_seeds, mean curve over rho, per-rho std, mean theta)."""
    by = {}
    for arm, _run, theta, _prov, curve in rows:
        if arms_keep and arm not in arms_keep:
            continue
        by.setdefault(arm, []).append((theta, curve))
    agg = {}
    for arm, items in by.items():
        m = {r: float(np.mean([c[r] for _t, c in items])) for r in RHOS}
        s = {r: float(np.std([c[r] for _t, c in items], ddof=1)) if len(items) > 1
             else 0.0 for r in RHOS}
        agg[arm] = (len(items), m, s, float(np.mean([t for t, _c in items])))
    return agg


def panel_a(ax, agg, order):
    """Stacked error anatomy. Bar base is annotated with the strict F1."""
    xs = np.arange(len(order))
    # A small gap between generator families, so the bars read as the grouped
    # constructions they are without needing a second encoding channel.
    fams, off, prev = [], [], None
    shift = 0.0
    for a in order:
        fam = style.ARMS[style.arm_of(a)][0]
        if prev is not None and fam != prev:
            shift += 0.45
        off.append(shift)
        prev = fam
    xs = xs + np.array(off)

    bottoms = np.zeros(len(order))
    segs = []
    for i, (label, rho) in enumerate(BANDS):
        if rho is None:
            vals = np.array([agg[a][1][0] for a in order])
        else:
            lo = {1: 0, 2: 1, 5: 2}[rho]
            vals = np.array([agg[a][1][rho] - agg[a][1][lo] for a in order])
        ax.bar(xs, vals, bottom=bottoms, color=RAMP[i], width=0.8,
               edgecolor="white", linewidth=0.6, label=label)
        bottoms = bottoms + vals
        segs.append(vals)
    ax.bar(xs, 1 - bottoms, bottom=bottoms, color=STRUCT, width=0.8,
           edgecolor="white", linewidth=0.6, label="structural")

    for x, a in zip(xs, order):
        ax.text(x, 0.012, f"{agg[a][1][0]:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="white", fontweight="bold")
        if agg[a][0] < style.MIN_SEEDS_FOR_ERRORBAR:
            ax.text(x, 1.01, "1 seed", ha="center", va="bottom", fontsize=5.5,
                    color=style.GREY)
    ax.set_xticks(xs)
    # Short keys on the axis, full names in the caption: the descriptive labels
    # are ~30 characters and five of them cannot share one axis legibly.
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("F1 at θ*, decomposed by buffer tolerance")
    ax.set_title("A — where the error lives")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              handlelength=1.1, columnspacing=0.9, fontsize=6.5)
    return segs


def panel_b(ax, agg, order, ref="r0"):
    """Δ vs r0 across ρ, direct-labelled, with r0's own seed spread at zero."""
    n_ref, m_ref, s_ref, _ = agg[ref]
    x = np.array(RHOS, dtype=float)
    # The noise floor: r0's cross-seed SD at each rho. A Delta inside this band
    # is not distinguishable from re-running the anchor.
    ax.fill_between(x, [-100 * s_ref[r] for r in RHOS],
                    [100 * s_ref[r] for r in RHOS],
                    color=style.GREY, alpha=0.18, lw=0,
                    label=f"r0 seed spread (n={n_ref})")
    ax.axhline(0, color=style.ZERO_LINE, lw=0.8)

    ends = []
    for a in order:
        if a == ref:
            continue
        n, m, s, _ = agg[a]
        y = 100 * np.array([m[r] - m_ref[r] for r in RHOS])
        ax.plot(x, y, color=style.color(a), ls=style.linestyle(a), lw=1.6,
                marker="o", ms=3, **({} if n >= style.MIN_SEEDS_FOR_ERRORBAR
                                     else {"mfc": "white"}))
        # Whiskers at rho=0 only: the user-facing risk is over-reading a
        # near-zero Delta at the strict end, which is where the arms crowd.
        ax.errorbar(0, y[0], yerr=100 * s[0], color=style.color(a), lw=1.0,
                    capsize=2.5, zorder=5)
        ends.append((y[-1], a))
    # Direct labels, nudged apart: r2a and r2b converge to within 0.05 pp by
    # rho=5 and their labels would otherwise print on top of each other.
    ax.set_xticks(RHOS)
    ax.set_xlabel("buffer tolerance ρ (px at 2.5 m)")
    ax.set_ylabel("ΔF1 vs r0  (percentage points)")
    ax.set_title("B — does the advantage survive forgiveness?")
    ax.margins(y=0.14)
    ax.set_xlim(-0.35, RHOS[-1] + 0.1)
    ax.legend(loc="lower left", fontsize=6.5)
    _label_ends(ax, ends)


def _label_ends(ax, ends):
    """Direct labels at the right edge, pushed apart by rendered text height."""
    ends.sort()
    lo, hi = ax.get_ylim()
    minsep = 0.062 * (hi - lo)
    placed = []
    for yy, a in ends:
        if placed and yy - placed[-1] < minsep:
            yy = placed[-1] + minsep
        placed.append(yy)
        ax.annotate(f"  {style.label(a)}", (ax.get_xlim()[1] - 0.1, yy),
                    fontsize=6.5, color=style.color(a), va="center", ha="left",
                    annotation_clip=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", action="append", required=True)
    ap.add_argument("--include", default=ERA)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm keys to keep (default: the formal R-arms)")
    ap.add_argument("--select-on", default="iou")
    ap.add_argument("--default-theta", type=float, default=0.5)
    ap.add_argument("--out-dir", default=style.FIGURES_DIR)
    args = ap.parse_args(argv)

    rows = load_runs(args.runs_dir, args.include, args.select_on, args.default_theta)
    if not rows:
        raise SystemExit("no runs with BOTH sweep.json and test_sweep.json found")

    keep = set(args.arms) if args.arms else {a for a, *_ in rows if "@" not in a}
    drop = [r for r in rows if r[3] == "fallback" and r[0] in keep]
    for arm, run, _t, _p, _c in drop:
        print(f"  DROP {run}: θ provenance is fallback (plan §9)")
    rows = [r for r in rows if r[3] != "fallback"]

    agg = aggregate(rows, keep)
    order = [a for a in style.ORDER if a in agg]
    print(f"\n{len(order)} arms: " + ", ".join(f"{a}(n={agg[a][0]})" for a in order))
    print(f"\n{'arm':<6} {'θ*':>6} {'F1@0':>7} {'ρ1':>7} {'ρ2':>7} {'ρ5':>7} "
          f"{'struct':>7} {'1st-px':>7}")
    for a in order:
        n, m, s, th = agg[a]
        first = (m[1] - m[0]) / (1 - m[0]) if m[0] < 1 else float("nan")
        print(f"{a:<6} {th:6.3f} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f} {m[5]:7.3f} "
              f"{1 - m[5]:7.3f} {first:6.1%}")

    import matplotlib.pyplot as plt

    style.apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL_WIDTH_IN + 1.4, 3.2),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    panel_a(axes[0], agg, order)
    panel_b(axes[1], agg, order)
    # Direct labels live outside the axes, so the right margin is reserved
    # rather than left to tight_layout, which clips annotation_clip=False text.
    fig.tight_layout(rect=(0, 0.02, 0.80, 1))
    paths = style.save(fig, args.out_dir, "F6_tolerance_anatomy")

    cap = ("F6. Buffered-F1 tolerance anatomy on the test split, at each arm's "
           "own θ* (re-argmaxed on val " + args.select_on + "): "
           + "; ".join(f"{a} θ*={agg[a][3]:.2f} (n={agg[a][0]})" for a in order)
           + ". Panel A decomposes F1 into what a ±ρ px buffer recovers; the "
           "grey remainder is structural error no tolerance forgives. Panel B "
           "shows each arm's advantage over r0 across ρ — a shrinking gap is "
           "positional precision, a flat gap is structure. Band = r0's "
           "cross-seed spread; whiskers at ρ=0 are the arm's own.")
    Path(args.out_dir, "F6_caption.txt").write_text(cap + "\n")
    for p in paths:
        print(f"  wrote {p}")
    print(f"  wrote {Path(args.out_dir, 'F6_caption.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
