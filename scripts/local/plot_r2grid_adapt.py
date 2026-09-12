#!/usr/bin/env python
"""The lr_sr grid's adapter trace: four band means + the normalised L2 drift.

    [ Red   ][ Green ]
    [ Blue  ][ NIR   ]
    [   normalised L2 drift   ]     <- spans both columns

Reads the W&B CSV exports in `Data/InstaRoad/r2grid` (one file per band plus
one for `sr_drift_rel`), which all carry the same eight series: r2a/r2b x
lr_sr 1e-4..1e-7 over 100 epochs.

ENCODING IS BORROWED, NOT INVENTED
----------------------------------
Colour, linestyle, label and ordering all come from `sr.probes.style`, so this
figure and F1-F3 say the same thing with the same ink: a ColorBrewer Blues ramp
keyed on lr_sr (1e-4 darkest, 1e-7 lightest) and SOLID = FFT hard constraint
mounted, DASHED = off. That matters here more than usual — eight series is past
the point where eight distinct hues would survive a colourblind check, and the
grid is not eight arbitrary categories: lr_sr is an ORDERED magnitude, so it
gets a sequential ramp, and the on/off split is the one genuinely categorical
axis, so it gets the non-colour channel.

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

    python scripts/local/plot_r2grid_adapt.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DATA_DIR = "/Volumes/MAC_KIOXIA/Data/InstaRoad/r2grid"
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
    """
    import pandas as pd

    df = pd.read_csv(path)
    out = {}
    for c in df.columns:
        if c == "epoch" or c.endswith(("__MIN", "__MAX")) or "_step" in c:
            continue
        name = re.sub(r"\s*-\s*\S+$", "", c)           # strip " - <metric>"
        name = re.sub(r"\s+", " ", name.replace("\n", " ")).strip()
        m = re.match(r"(r2[ab])\s*,?\s*(1e-[0-9]+)", name)
        if not m:
            raise SystemExit(f"cannot parse series name {name!r} in {path.name}")
        out[f"{m.group(1)}@{m.group(2)}"] = df.set_index("epoch")[c]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=DATA_DIR,
                    help="where the figure lands (default: beside the CSVs)")
    ap.add_argument("--stem", default="r2grid_bands_drift")
    ap.add_argument("--free-y", action="store_true",
                    help="give each band panel its own y limits (see module docstring)")
    ap.add_argument("--linear-drift", action="store_true",
                    help="linear y on the drift panel instead of log")
    args = ap.parse_args(argv)

    d = Path(args.data_dir)
    drift_files = sorted(d.glob(DRIFT_GLOB))
    if not drift_files:
        raise SystemExit(f"no {DRIFT_GLOB} (the sr_drift_rel export) under {d}")
    band_data = [(t, read_series(d / f)) for _, f, t in BANDS]
    drift = read_series(drift_files[-1])

    from sr.probes import style
    style.apply_rc()
    import matplotlib.pyplot as plt

    arms = sorted(drift, key=style.sort_key)

    fig = plt.figure(figsize=(style.FULL_WIDTH_IN, 7.0))
    # Row 3 spans both columns; the extra height ratio buys the log decades
    # room to separate, which is the whole reason that panel is log.
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25],
                          hspace=0.42, wspace=0.22,
                          bottom=0.13, top=0.965, left=0.09, right=0.985)

    ax0 = None
    for i, (title, series) in enumerate(band_data):
        ax = fig.add_subplot(gs[i // 2, i % 2],
                             **({} if args.free_y or ax0 is None else {"sharey": ax0}))
        ax0 = ax0 or ax
        ax.axhline(0.0, color=style.ZERO_LINE, lw=0.7, zorder=1)
        for a in arms:
            if a in series:
                ax.plot(series[a].index, series[a].values, color=style.color(a),
                        linestyle=style.linestyle(a), lw=1.4, zorder=3)
        ax.set_title(title, loc="left")
        if i % 2 == 0:
            ax.set_ylabel("adapt_mean")
        else:
            ax.tick_params(labelleft=args.free_y)
        if i >= 2:
            ax.set_xlabel("epoch")

    axd = fig.add_subplot(gs[2, :])
    for a in arms:
        axd.plot(drift[a].index, drift[a].values, color=style.color(a),
                 linestyle=style.linestyle(a), lw=1.6, zorder=3,
                 label=style.label(a))
    if not args.linear_drift:
        axd.set_yscale("log")
    axd.set_title("Normalised L2 drift of the SR generator  (sr_drift_rel)", loc="left")
    axd.set_xlabel("epoch")
    axd.set_ylabel("‖Δθ‖ / ‖θ₀‖" + ("  (log)" if not args.linear_drift else ""))

    # ONE legend for the whole figure: every panel draws the same eight series
    # in the same ink, so five legends would be the same key printed five times.
    h, l = axd.get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.002),
               frameon=False, fontsize=7.2, columnspacing=1.4, handlelength=2.4)

    for p in style.save(fig, Path(args.out_dir), args.stem):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
