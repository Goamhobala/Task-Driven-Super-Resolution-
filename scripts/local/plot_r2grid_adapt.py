#!/usr/bin/env python
"""The lr_sr grid's adapter trace: four band means + the normalised L2 drift.

    [ Red   ][ Green ]
    [ Blue  ][ NIR   ]
    [   normalised L2 drift   ]     <- spans both columns

Reads the W&B CSV exports in `Data/InstaRoad/r2grid` and `Data/InstaRoad/r4grid`
(one file per band plus one for `sr_drift_rel`, r2grid only — r4grid has no
drift export yet). r2grid carries eight series (r2a/r2b x lr_sr 1e-4..1e-7);
r4grid carries whatever cells were actually run (currently on:1e-4..1e-7,
off:1e-4..1e-6 — off@1e-7 is missing, not merely empty).

ENCODING IS BORROWED, NOT INVENTED
----------------------------------
Colour, linestyle, label and ordering all come from `sr.probes.style`, so this
figure and F1-F3 say the same thing with the same ink: a ColorBrewer Blues (r2,
SEN2SR) / Greens (r4, SR4RS) ramp keyed on lr_sr (1e-4 darkest, 1e-7 lightest)
and SOLID = FFT hard constraint mounted, DASHED = off. That matters here more
than usual — with both grids in one figure there are up to fifteen series,
past the point where distinct hues would survive a colourblind check, and the
grid is not arbitrary categories: lr_sr is an ORDERED magnitude, so it gets a
sequential ramp, and the on/off split is the one genuinely categorical axis,
so it gets the non-colour channel.

THE FOUR BAND PANELS SHARE ONE Y AXIS
-------------------------------------
They are the same quantity in the same units, and the question the figure is
asked is "which band moves most" — giving each panel its own limits would make
every band look equally displaced, which is the standard way to mislead with a
small-multiple. NIR genuinely runs ~2.5x the visible bands' range; that is a
result, not a layout problem. `--free-y` opts out.

THE DRIFT PANEL IS LOG
----------------------
`sr_drift_rel` spans 1.3e-4 to 0.45 — three and a half decades, one per decade
of lr_sr. On a linear axis the 1e-6 and 1e-7 arms are indistinguishable from
zero and from each other. Every value is strictly positive, so a log axis
invents nothing. `--linear-drift` opts out.

R4 RUNS ARE NOT ALL 100 EPOCHS, AND SOME LOOK LIKE CONTINUATIONS
------------------------------------------------------------------
Unlike r2grid, several r4grid W&B columns are empty for a stretch then pick up
mid-run (e.g. `on_ls1e-4` is blank through epoch 60, then starts at epoch 61
with its own step counter reset to 268 — the same step r2/r4 runs record at
epoch 0). That is the signature of a resumed job logged under a new run name,
not a run that simply started late. This script does NOT merge such pairs —
that is a modelling decision (what the merged arm would even mean) left to the
caller — it only detects and prints a warning per band so the gap is not
mistaken for a run that silently died. `--no-continuity-check` silences it.

    python scripts/local/plot_r2grid_adapt.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DATA_DIR = "/Volumes/MAC_KIOXIA/Data/InstaRoad/r2grid"
R4_DATA_DIR = "/Volumes/MAC_KIOXIA/Data/InstaRoad/r4grid"
R0_DATA_DIR = "/Volumes/MAC_KIOXIA/Data/InstaRoad/r0grid"
# panel order = reading order of the 2x2 block.
BANDS = [("red", "ema_mean_red.csv", "Red  (b0)"),
         ("green", "ema_mean_green.csv", "Green  (b1)"),
         ("blue", "ema_mean_blue.csv", "Blue  (b2)"),
         ("nir", "ema_mean_nir.csv", "NIR  (b3)")]
DRIFT_GLOB = "wandb_export_*.csv"


def read_series(path: Path) -> dict[str, "object"]:
    """{arm_key: Series indexed by epoch} from one W&B export.

    W&B writes each series as `"<name>\\n <lr> - <metric>"` plus `__MIN`/`__MAX`
    companions and a `_step` triple. The companions are dropped: they are the
    smoothing envelope, and with one run per series they equal the mean exactly
    (verified — max |MIN-MAX| is 0), so drawing a band would be drawing zero.

    Three naming schemes show up across r2grid, r4grid and r0grid exports:
    `"r2a, 1e-7 - adapt_mean_b0"` (r2grid), `"r4grid_on_ls1e-5_seed0 -
    adapt_mean_b0"` (r4grid) — the latter is exactly what `style.arm_of`
    already parses for the probe figures, so it is reused here rather than
    re-deriving the on/off -> a/b mapping a second time — and `"r0-gap -
    adapt_mean_b0"` (r0grid), which has no lr_sr/HC split so it maps straight
    to the single key "r0".

    The r0grid export also has no `epoch` column at all (W&B logged it against
    `trainer/global_step`, a finer, unrelated counter) — every export's rows
    are already in epoch order, though, epoch == row position (verified
    against r2grid/r4grid's own literal `epoch` column, which is 0..99 in
    row order), so the x-axis is read from row POSITION rather than any named
    column, which works for all three schemes without special-casing r0grid.
    """
    from sr.probes import style
    import pandas as pd

    df = pd.read_csv(path)
    xcol = df.columns[0]           # "epoch" (r2/r4) or "trainer/global_step" (r0)
    out = {}
    for c in df.columns:
        if c == xcol or c.endswith(("__MIN", "__MAX")) or "_step" in c:
            continue
        name = re.sub(r"\s*-\s*\S+$", "", c)           # strip " - <metric>"
        name = re.sub(r"\s+", " ", name.replace("\n", " ")).strip()
        m = re.match(r"(r2[ab])\s*,?\s*(1e-[0-9]+)", name)
        if m:
            key = f"{m.group(1)}@{m.group(2)}"
        elif re.match(r"^(?:sr_)?r4grid_(on|off)_ls[0-9.e+-]+_", name):
            key = style.arm_of(name)
        elif name.startswith("r0"):
            key = "r0"
        else:
            raise SystemExit(f"cannot parse series name {name!r} in {path.name}")
        out[key] = df[c].dropna()

        # `_step` is W&B's own global counter for that column. A column whose
        # step DROPS partway through (e.g. 16420 then back to 268) is one
        # column carrying two concatenated runs under the same name — not a
        # run that paused and resumed its own count. That is a stronger,
        # single-arm signal than the cross-arm epoch-adjacency in
        # `check_continuity`, and needs the raw (non-dropna'd) step column to
        # see, since the reset shows up exactly where the value column is
        # also NaN for the old run and starts fresh for the new one.
        candidates = [sc for sc in df.columns
                      if sc.startswith(name) and sc.endswith("_step")
                      and not sc.endswith(("__MIN", "__MAX"))]
        if candidates:
            steps = df[candidates[0]].dropna()
            drops = steps[steps.diff() < 0]
            if len(drops):
                print(f"  note [{path.name}:{key}]: _step resets at epoch "
                      f"{drops.index[0]} ({steps.loc[drops.index[0] - 1]:.0f} -> "
                      f"{drops.iloc[0]:.0f}) — this column concatenates two "
                      "runs under one name; its earlier and later halves may "
                      "not belong to the same grid cell.")
    return out


def check_continuity(series: dict, band_title: str) -> None:
    """Flag arm pairs that look like one resumed job under two run names.

    Heuristic: arm B's data starts the epoch right after arm A's data ends,
    and both belong to the same on/off family (only lr_sr differs) — plotting
    them as two independent grid cells would misrepresent a single trajectory
    as two, so this is surfaced rather than silently plotted.
    """
    bounds = {a: (s.index.min(), s.index.max()) for a, s in series.items() if len(s)}
    for a, (a_lo, a_hi) in bounds.items():
        for b, (b_lo, b_hi) in bounds.items():
            if a == b or b_lo != a_hi + 1:
                continue
            fam_a, fam_b = a.split("@")[0], b.split("@")[0]
            if fam_a == fam_b:
                print(f"  note [{band_title}]: {b} (epoch {b_lo}-{b_hi}) starts "
                      f"immediately after {a} (epoch {a_lo}-{a_hi}) ends — "
                      "check whether this is one resumed run logged under two "
                      "names before treating them as separate grid cells.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", nargs="+", default=[DATA_DIR, R4_DATA_DIR, R0_DATA_DIR],
                    help="one or more W&B-export dirs to merge (default: r2grid "
                         "+ r4grid + r0grid)")
    ap.add_argument("--out-dir", default=DATA_DIR,
                    help="where the figure lands (default: beside the r2grid CSVs)")
    ap.add_argument("--stem", default="r2r4grid_bands_drift")
    ap.add_argument("--free-y", action="store_true",
                    help="give each band panel its own y limits (see module docstring)")
    ap.add_argument("--linear-drift", action="store_true",
                    help="linear y on the drift panel instead of log")
    ap.add_argument("--no-continuity-check", action="store_true",
                    help="skip the resumed-run heuristic (see module docstring)")
    ap.add_argument("--no-drift", action="store_true",
                    help="drop the L2-drift row and draw only the 4 band panels")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="fixed y limits for the (shared) band panels, "
                         "overriding the auto range (implies not --free-y)")
    ap.add_argument("--no-r0", action="store_true",
                    help="drop the r0 (bicubic) trace")
    args = ap.parse_args(argv)

    dirs = [Path(p) for p in args.data_dir]

    # Merge each band's series across every data dir. A key collision would
    # mean two dirs claim the same arm, which never happens across r2grid
    # (r2a/r2b) and r4grid (r4a/r4b) — asserted rather than silently
    # overwritten so a future third grid doesn't merge wrong.
    #
    # Prefer "<stem>_completed.csv" over "<stem>.csv" in a given dir when both
    # exist: `_completed` is a hand-fixed export that stitches the resumed-run
    # halves `check_continuity`/the `_step`-reset check flag below, so it is
    # strictly better data for whichever arms it covers. Falls back to the
    # continuation-affected raw file for the rest (e.g. r4a@1e-4, which the
    # completed export also only carries partially — it never reached epoch 100).
    band_data = []
    for _, fname, title in BANDS:
        merged: dict = {}
        for d in dirs:
            stem, ext = fname.rsplit(".", 1)
            completed = d / f"{stem}_completed.{ext}"
            p = completed if completed.is_file() else d / fname
            if not p.is_file():
                continue
            for key, s in read_series(p).items():
                if key in merged:
                    raise SystemExit(f"{key!r} appears in more than one --data-dir "
                                     f"for {fname} — that shouldn't happen across "
                                     "r2grid/r4grid/r0grid; check the merge")
                merged[key] = s
        if not merged:
            raise SystemExit(f"{fname} not found under any of {dirs}")
        band_data.append((title, merged))

    # r0 (bicubic) is read from `r0grid`, same as every other arm — a real
    # `adapt_mean` export (`"r0-gap"` in W&B), not a modelled zero line. It
    # measures something with the same name as the generators' adapter stat
    # but no generator behind it, so it is real data worth plotting, not an
    # invented anchor.
    if args.no_r0:
        for _, merged in band_data:
            merged.pop("r0", None)

    # The drift export exists only for r2grid so far; r4grid arms simply don't
    # appear in that panel until one is recorded.
    drift = {}
    if not args.no_drift:
        for d in dirs:
            for f in sorted(d.glob(DRIFT_GLOB)):
                for key, s in read_series(f).items():
                    drift.setdefault(key, s)

    from sr.probes import style
    style.apply_rc()
    import matplotlib.pyplot as plt

    if not args.no_continuity_check:
        for title, series in band_data:
            check_continuity(series, title)

    # Union of every arm seen in any band panel — the legend and draw order
    # must cover r4 arms too, not just whatever the (r2grid-only) drift panel
    # happens to carry.
    arms = sorted({a for _, series in band_data for a in series}, key=style.sort_key)

    nrows = 2 if args.no_drift else 3
    # The legend below the axes needs about the same vertical room whether or
    # not the drift row is there (same arm count -> same number of legend
    # rows), so dropping a whole gridspec row without growing `bottom` would
    # let the legend collide with the band panels above it.
    fig = plt.figure(figsize=(style.FULL_WIDTH_IN, 6.6 if args.no_drift else 7.0))
    # Row 3 (when present) spans both columns; the extra height ratio buys the
    # log decades room to separate, which is the whole reason that panel is log.
    height_ratios = [1.0, 1.0] if args.no_drift else [1.0, 1.0, 1.25]
    gs = fig.add_gridspec(nrows, 2, height_ratios=height_ratios,
                          hspace=0.42, wspace=0.22,
                          bottom=(0.18),
                          top=0.93, left=0.09, right=0.985)

    ax0 = None
    for i, (title, series) in enumerate(band_data):
        ax = fig.add_subplot(gs[i // 2, i % 2],
                             **({} if args.free_y or args.ylim or ax0 is None
                                else {"sharey": ax0}))
        ax0 = ax0 or ax
        ax.axhline(0.0, color=style.ZERO_LINE, lw=0.7, zorder=1)
        for a in arms:
            if a in series:
                ax.plot(series[a].index, series[a].values, color=style.color(a),
                        linestyle=style.linestyle(a), lw=1.4, zorder=3)
        if args.ylim:
            ax.set_ylim(*args.ylim)
        ax.set_title(title, loc="left")
        if i % 2 == 0:
            ax.set_ylabel("adapt_mean")
        else:
            ax.tick_params(labelleft=args.free_y or bool(args.ylim))
        if i >= 2:
            ax.set_xlabel("epoch")

    if not args.no_drift:
        axd = fig.add_subplot(gs[2, :])
        for a in arms:
            if a in drift:
                axd.plot(drift[a].index, drift[a].values, color=style.color(a),
                         linestyle=style.linestyle(a), lw=1.6, zorder=3)
        if not args.linear_drift:
            axd.set_yscale("log")
        axd.set_title("Normalised L2 drift of the SR generator  (sr_drift_rel)"
                      + ("" if all(a in drift for a in arms) else
                         "  — r2grid only, no r4grid export yet"), loc="left")
        axd.set_xlabel("epoch")
        axd.set_ylabel("‖Δθ‖ / ‖θ₀‖" + ("  (log)" if not args.linear_drift else ""))

    # ONE legend for the whole figure, built from the full arm union (not just
    # whatever the drift panel happens to carry) via proxy handles — every
    # panel draws the same series in the same ink, so five legends would be
    # the same key printed five times, and drift-less r4 arms still need one.
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=style.color(a), linestyle=style.linestyle(a),
                      lw=1.6, label=style.label(a)) for a in arms]
    fig.legend(handles, [h.get_label() for h in handles], ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.002), frameon=False,
               fontsize=7.2, columnspacing=1.4, handlelength=2.4)

    for p in style.save(fig, Path(args.out_dir), args.stem):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
