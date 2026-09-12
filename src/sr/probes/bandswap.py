"""Instrument D readout — does the gain travel with the high band?

    PYTHONPATH=src python -m sr.probes.bandswap --cache-dir bandswap_cache

Reads only `bandswap_extract.py`'s cache. Emits a tidy CSV per readout plus
F4 (the frequency-ablation curve) and F5 (the swap + transfer summary).

WHAT THE NUMBERS MEAN
---------------------
Everything is PAIRED BY CHIP and restricted to chips that hold road: AP is
undefined on a road-free chip and the fixture holds 29% of them by design, so
a fabricated 0.0 would move an arm's mean in proportion to its empty-chip count
rather than to anything about the arm (`occlusion.py`'s rule, same reason).

The headline statistic is the **carried fraction**

    c = (AP[swap_hi] - AP[r0]) / (AP[own] - AP[r0])

on chip-paired means: the share of an arm's deployed advantage over r0 that
survives when the arm contributes ONLY its high band and r0 supplies the low
one. c ~ 1 means the advantage is high-band — "sharpness", demonstrated. c ~ 0
means it lived in the low band, i.e. radiometry. The denominator is a
difference of two small numbers, so c is reported with a chip-bootstrap CI and
suppressed outright when the denominator's own CI straddles zero: a ratio to a
gain that is not established is not a quantity.

**The HC-on lane is a NULL, and that is the instrument's self-test.** For an
HC-on arm the deployed forward already IS the splice, so `swap_hi` must
reproduce `own` to within the sr_pad crop residue and c must come out at 1.
An on-cell that does NOT return c ~ 1 is a bug in this script, not a finding.
The off lane is where the counterfactual is real.

Single-seed grid cells have no within-arm noise floor, so every between-cell
statement here is descriptive, per the grid plan's §4 pre-commitment, and the
instrument itself is EXPLORATORY under the probes plan's §2 — it was added
after the LDA and occlusion reads. Both facts are stamped into the CSVs and
the figure captions rather than left to the reader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sr.probes import cache, style
from sr.probes.bandswap_extract import RADII
from sr.probes.make_fixtures import CHIPS_JSON, FIXTURE_DIR

ANCHORS = ("own", "r0")
BOOT = 2000
BOOT_SEED = 20260831


def _strata(fixture_dir) -> dict[int, str]:
    rec = json.loads((Path(fixture_dir) / CHIPS_JSON).read_text())
    return {i: c["stratum"] for i, c in enumerate(rec["chips"])}


def _boot(diff: np.ndarray, rng, n=BOOT):
    """Percentile CI of a paired mean, resampling CHIPS (the unit of pairing)."""
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    b = diff[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load_long(cache_dir, fixture_dir, arms=None):
    """Tidy frame: one row per (arm, chip, decoder, condition) on road chips."""
    import pandas as pd

    metas = cache.load_metas(cache_dir, arms=arms, require=("bandswap.parquet",))
    strat = _strata(fixture_dir)
    frames = []
    for m in metas:
        df = pd.read_parquet(Path(m["dir"]) / "bandswap.parquet")
        df = df[df["road_px"] > 0].copy()
        df["arm"] = m["arm"]
        df["run"] = m["run"]
        df["seed"] = m["seed"]
        df["hc"] = "on" if style.ARMS[m["arm"]][2] else "off"
        df["stratum"] = df["chip"].map(strat)
        frames.append(df)
    return pd.concat(frames, ignore_index=True), metas


def summarise(long, rng):
    """Paired per-(arm, stratum, decoder, condition) means and ΔAP vs r0.

    The pairing is enforced by pivoting on chip and dropping any chip missing a
    condition, so every condition of one arm-stratum cell is a mean over
    exactly the same chips. Without that, an arm whose AP is NaN on a handful
    of chips in one condition would be compared against a different chip set.
    """
    import pandas as pd

    rows = []
    for (arm, stratum, decoder), g in long.groupby(["arm", "stratum", "decoder"],
                                                   dropna=False):
        piv = g.pivot_table(index="chip", columns="condition", values="ap").dropna()
        if piv.empty or not set(ANCHORS) <= set(piv.columns):
            continue
        own, ref = piv["own"].to_numpy(), piv["r0"].to_numpy()
        gain = own - ref
        g_lo, g_hi = _boot(gain, rng)
        for cond in piv.columns:
            v = piv[cond].to_numpy()
            d = v - ref
            lo, hi = _boot(d, rng)
            rows.append(dict(
                arm=arm, stratum=stratum, decoder=decoder, condition=cond,
                n_chips=len(piv), ap=float(v.mean()),
                ap_own=float(own.mean()), ap_r0=float(ref.mean()),
                d_vs_r0=float(d.mean()), d_lo=lo, d_hi=hi,
                gain=float(gain.mean()), gain_lo=g_lo, gain_hi=g_hi,
                # A ratio to an unestablished gain is not a quantity: the CI
                # straddling zero means the denominator could be either sign.
                carried=(float(d.mean() / gain.mean())
                         if g_lo * g_hi > 0 else np.nan)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_curve(summ, out_dir, stratum="Rural"):
    """F4 — carried fraction vs the ideal-disk radius, one line per grid cell.

    x is the radius of r0's low-band donation, so the LEFT edge is "the arm
    keeps everything but DC" and the RIGHT edge is "r0 supplies almost
    everything". A line that stays near 1 until the radius passes road width
    is the sharpness claim, drawn.
    """
    import matplotlib.pyplot as plt

    style.apply_rc()
    s = summ[(summ.stratum == stratum) & (summ.decoder == "arm")]
    fig, ax = plt.subplots(figsize=(style.FULL_WIDTH_IN * 0.62, 2.9))
    for arm in sorted(s.arm.unique(), key=style.sort_key):
        a = s[s.arm == arm].set_index("condition")
        xs, ys = [], []
        for r in RADII:
            k = f"swap_hi_r{r}"
            if k in a.index and np.isfinite(a.loc[k, "carried"]):
                xs.append(r)
                ys.append(a.loc[k, "carried"])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=style.color(arm),
                ls=style.linestyle(arm), label=style.label(arm))
    ax.axhline(1.0, color=style.ZERO_LINE, lw=0.7, ls=":")
    ax.axhline(0.0, color=style.ZERO_LINE, lw=0.7)
    ax.axvline(35, color=style.GREY, lw=0.7, ls="--")
    ax.annotate("HC cut (σ=35, ≈37 m)", (35, ax.get_ylim()[1]), fontsize=6,
                ha="left", va="top", rotation=90, color=style.GREY,
                xytext=(2, -2), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=8)
    ax.set_xticks([0, 8, 16, 35, 64, 128, 256])
    ax.set_xticklabels(["0", "8", "16", "35", "64", "128", "256"])
    ax.set_xlabel("radius of r0's low-band donation  (cycles / 512 px;  "
                  "wavelength = 1280/r m)")
    ax.set_ylabel("carried fraction of the gain over r0")
    ax.set_title(f"F4 — where the {stratum.lower()} advantage lives "
                 f"(exploratory; 1 seed/cell)")
    ax.legend(ncol=2)
    fig.tight_layout()
    return style.save(fig, out_dir, f"F4_bandswap_curve_{stratum.lower()}")


def fig_summary(summ, out_dir):
    """F5 — the σ=35 swap per stratum, plus the transfer row on r0's decoder."""
    import matplotlib.pyplot as plt

    style.apply_rc()
    strata = ["Rural", "PeriUrban", "Urban"]
    fig, axes = plt.subplots(1, len(strata), figsize=(style.FULL_WIDTH_IN, 2.7),
                             sharey=True)
    for ax, st in zip(axes, strata):
        s = summ[(summ.stratum == st) & (summ.decoder == "arm")]
        arms = sorted(s.arm.unique(), key=style.sort_key)
        for i, arm in enumerate(arms):
            a = s[s.arm == arm].set_index("condition")
            for cond, mk, off in (("swap_hi", "o", -0.15), ("swap_lo", "s", 0.15)):
                if cond not in a.index:
                    continue
                ax.errorbar(i + off, a.loc[cond, "d_vs_r0"],
                            yerr=[[a.loc[cond, "d_vs_r0"] - a.loc[cond, "d_lo"]],
                                  [a.loc[cond, "d_hi"] - a.loc[cond, "d_vs_r0"]]],
                            marker=mk, ms=4, lw=0.8, capsize=2,
                            color=style.color(arm),
                            mfc=style.color(arm) if cond == "swap_hi" else "white")
            if "own" in a.index:
                ax.plot([i - 0.3, i + 0.3], [a.loc["own", "d_vs_r0"]] * 2,
                        color=style.color(arm), lw=1.4, alpha=0.6)
        ax.axhline(0, color=style.ZERO_LINE, lw=0.7)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([style.label(a) for a in arms], rotation=90)
        ax.set_title(st)
    axes[0].set_ylabel("ΔAP vs r0  (paired, road chips)")
    fig.suptitle("F5 — σ=35 band swap.  filled = r0 low + arm high;  hollow = "
                 "arm low + r0 high;  bar = deployed arm", fontsize=8)
    fig.tight_layout()
    return style.save(fig, out_dir, "F5_bandswap_summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="bandswap_cache")
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--out-dir", default=style.FIGURES_DIR)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--stratum", default="Rural", help="stratum for F4")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(BOOT_SEED)
    long, metas = load_long(args.cache_dir, args.fixture_dir, args.arms)
    summ = summarise(long, rng)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "bandswap_summary.csv", index=False)

    # The self-test, printed loudly: an HC-on cell must return carried ~ 1.
    print("\nself-test — HC-on cells must carry ~1.00 (the deployed forward "
          "already IS the splice):")
    on = summ[(summ.decoder == "arm") & (summ.condition == "swap_hi")
              & (summ.arm.str.startswith("r2a@"))]
    for _, r in on.iterrows():
        flag = "" if abs(r.carried - 1) < 0.05 or not np.isfinite(r.carried) else "  <-- CHECK"
        print(f"  {r.arm:<10} {r.stratum:<10} carried={r.carried:6.3f}{flag}")

    print("\nσ=35 swap, ΔAP vs r0 (paired, road chips):")
    for _, r in summ[(summ.decoder == "arm")
                     & (summ.condition.isin(["own", "swap_hi", "swap_lo"]))
                     ].sort_values(["stratum", "arm", "condition"]).iterrows():
        print(f"  {r.stratum:<10} {r.arm:<10} {r.condition:<9} "
              f"ΔAP={r.d_vs_r0:+.4f} [{r.d_lo:+.4f},{r.d_hi:+.4f}]  n={r.n_chips}")

    print()
    for p in fig_curve(summ, out, args.stratum) + fig_summary(summ, out):
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
