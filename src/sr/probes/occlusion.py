"""Instrument C — band occlusion (plan §5, figure F2).

    PYTHONPATH=src python -m sr.probes.occlusion --cache-dir probe_cache

Reads only the extraction cache. Emits into `--out-dir` (default `figures/probes/`):

    occlusion_delta.csv   per arm-seed x condition: AP and IoU@theta*, deltas
    F2_occlusion.pdf/.png main text: dAP dot plot | first-conv weight norms
    A2_occlusion_iou.*    appendix: the dIoU@theta* version

WHAT THE NUMBER IS
------------------
Band b of the pre-adapter SR output was replaced by that checkpoint's own
post-recalibration `band_mean[b]`, so after its z-score the channel is exactly
zero. dAP is then (occluded - intact), macro over the fixture chips that HAVE
road: AP is undefined on a road-free chip and the fixture holds 29% of them by
design, so a fabricated 0.0 would drag every arm's mean down in proportion to
its empty-chip count rather than in proportion to anything about the arm.

The same chips are used for the intact and occluded means of a given arm, so
each dAP is a paired difference.

RELIANCE, NOT INFORMATION
-------------------------
Mean-substitution is mildly off-manifold and the bands are correlated, so a
large dAP says the network *relies* on that channel, not that the channel
carries unique information. Two arms with different intact AP also have
different room to fall, which is why `rel_dap` (dAP / AP_intact) is reported
alongside and annotated on the figure where the baselines diverge.

THE WEIGHT-SPACE TWIN
---------------------
The companion panel is each arm's stem-convolution per-band input norm divided
by the shared ImageNet initialisation's. That reference is NOT flat: smp builds
a 4-channel `conv1` from the 3-channel ImageNet one by duplicating a channel,
so the init's NIR norm equals its R norm exactly, and a raw norm plot would
read that artefact as a finding. The ratio removes it.

This panel is a cross-check, not a second measurement: first-layer norm is
where the network *could* look, dAP is where it *does*.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sr.probes import cache, style
from sr.probes.style import FIGURES_DIR

CONDITIONS = ["R", "G", "B", "NIR", "RGB"]
INIT_CACHE = "imagenet_conv1_norms.npy"


# ---------------------------------------------------------------- the numbers
def _macro(series: pd.Series) -> float:
    """Mean over the chips where the statistic is defined."""
    v = series.to_numpy(dtype="float64")
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def _iou_per_chip(df: pd.DataFrame) -> pd.Series:
    """tp / (tp + fp + fn), NaN where the union is empty.

    An empty union means the chip has no road AND none was predicted — a
    correct outcome with no IoU. Scoring it 1.0 would reward an arm for the
    fixture's empty-chip rate; scoring it 0.0 would punish it for the same.
    """
    u = df["tp"] + df["fp"] + df["fn"]
    return (df["tp"] / u.where(u > 0)).astype("float64")


def deltas(meta, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per condition for one arm-seed: intact vs occluded, AP and IoU.

    `table` overrides the cached parquet — the arithmetic is the part worth
    testing, and it should be reachable without a cache on disk.
    """
    t = (table.copy() if table is not None
         else pd.read_parquet(meta["dir"] / "occlusion.parquet"))
    t["iou"] = _iou_per_chip(t)

    base = t[t["condition"] == "none"].set_index("chip")
    # AP is defined only on chips with road; restrict BOTH arms of every
    # difference to that same set so each delta is paired.
    ap_chips = base.index[~base["ap"].isna()]
    rows = []
    for cond in CONDITIONS:
        c = t[t["condition"] == cond].set_index("chip")
        if c.empty:
            continue
        ap_b, ap_c = base.loc[ap_chips, "ap"], c.loc[ap_chips, "ap"]
        # AP's no-skill value is the chip's road prevalence, so dAP cannot fall
        # below (chance - intact). An occluded model that has collapsed sits
        # exactly there, which is why several conditions can coincide to the
        # last digit; the figure draws this floor rather than letting the tie
        # read as a coincidence.
        prev = (base.loc[ap_chips, "road_px"]
                / (base.loc[ap_chips, ["tp", "fp", "fn", "tn"]].sum(axis=1)))
        iou_b, iou_c = base["iou"], c.reindex(base.index)["iou"]
        # Micro IoU pools the counts instead of averaging per-chip ratios: it is
        # the area-weighted view, and the two disagree exactly when the arms
        # differ on small-road chips.
        def micro(d):
            return float(d["tp"].sum() / max(int((d["tp"] + d["fp"] + d["fn"]).sum()), 1))
        rows.append({
            "run": meta["run"], "arm": meta["arm"], "seed": meta["seed"],
            "condition": cond,
            "n_chips": meta["n_chips"], "n_ap_chips": int(len(ap_chips)),
            "theta": meta["theta"], "theta_provenance": meta.get("theta_provenance"),
            "ap_base": _macro(ap_b), "ap_cond": _macro(ap_c),
            "ap_chance": _macro(prev), "dap_floor": _macro(prev) - _macro(ap_b),
            "dap": _macro(ap_c) - _macro(ap_b),
            "rel_dap": (_macro(ap_c) - _macro(ap_b)) / _macro(ap_b),
            "iou_base_macro": _macro(iou_b), "iou_cond_macro": _macro(iou_c),
            "diou_macro": _macro(iou_c) - _macro(iou_b),
            "iou_base_micro": micro(base), "iou_cond_micro": micro(c),
            "diou_micro": micro(c) - micro(base),
        })
    return pd.DataFrame(rows)


def imagenet_conv1_norms(cache_dir: Path, encoder="resnet34", in_channels=4):
    """Per-band L2 norm of the shared ImageNet stem convolution (cached to disk).

    Built once from `unet.build_model`, i.e. the same constructor every arm was
    initialised with, so the ratio is against the actual starting point rather
    than a reconstruction of it.
    """
    p = Path(cache_dir) / INIT_CACHE
    if p.exists():
        return np.load(p)
    from unet.model import build_model

    w = build_model(encoder, "imagenet", in_channels, 1).encoder.conv1.weight
    n = np.sqrt((w.detach().numpy() ** 2).sum(axis=(0, 2, 3)))
    np.save(p, n)
    return n


def conv1_ratios(metas, cache_dir: Path, bands) -> pd.DataFrame:
    """Per-band stem-conv norm / the ImageNet init's, one row per arm-seed."""
    ref = None
    rows = []
    for m in metas:
        f = m["dir"] / "first_conv.npy"
        if not f.exists():
            continue
        w = np.load(f)
        if ref is None:
            ref = imagenet_conv1_norms(cache_dir, in_channels=w.shape[1])
        n = np.sqrt((w ** 2).sum(axis=(0, 2, 3)))
        rows.append({"run": m["run"], "arm": m["arm"], "seed": m["seed"],
                     **{b: v for b, v in zip(bands, n / ref)}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figures
def _needs_symlog(v: np.ndarray) -> bool:
    """True when the deltas span enough decades that a linear axis hides the
    small ones — the plan's condition for not clipping the r0 collapse."""
    a = np.abs(v[np.isfinite(v) & (np.abs(v) > 0)])
    return a.size > 2 and a.max() / np.percentile(a, 25) > 25


def _dotplot(ax, df, value, seeds, ylabel, yscale="auto",
             annotate_rel=False, floor_col=None):
    arms = sorted(df["arm"].unique(), key=style.sort_key)
    w = 0.72 / max(len(arms), 1)
    # Markers shrink as the arm set grows, or eleven of them per condition
    # overlap into a smear and the seed range bars vanish behind them.
    msize = 5.5 if len(arms) <= 6 else 4.0
    for i, arm in enumerate(arms):
        g = df[df["arm"] == arm]
        off = i * w - 0.36 + w / 2
        for j, cond in enumerate(CONDITIONS):
            v = g[g["condition"] == cond][value].to_numpy(dtype="float64")
            if not v.size:
                continue
            ax.plot(np.full(v.size, j + off), v, color=style.color(arm),
                    zorder=3, **style.marker_kwargs(seeds.get(arm, 1),
                                                    style.marker(arm), msize))
            # Error bars only at n >= 3 (§9): a range drawn over two points
            # would look like a measured spread.
            if v.size >= style.MIN_SEEDS_FOR_ERRORBAR:
                ax.plot([j + off, j + off], [v.min(), v.max()],
                        color=style.color(arm), lw=0.9, alpha=0.6, zorder=2)
        # The legend key must show the MARKER, since that is what the panel
        # draws; a line-only key would not say which dots belong to which arm.
        ax.plot([], [], color=style.color(arm), ls=style.linestyle(arm),
                label=style.label(arm),
                **{k: v for k, v in style.marker_kwargs(
                    seeds.get(arm, 1), style.marker(arm), msize).items()
                   if k != "linestyle"})
    if annotate_rel:
        for j, cond in enumerate(CONDITIONS):
            r = df[df["condition"] == cond]["rel_dap"]
            if r.notna().any():
                ax.annotate(f"{100 * r.mean():.0f}%", (j, 1.0), fontsize=6,
                            xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", color="0.35")
    ax.axhline(0, color=style.ZERO_LINE, lw=0.8, ls="--", zorder=1)
    if floor_col and floor_col in df:
        for arm in arms:
            f = df[df["arm"] == arm][floor_col].mean()
            if np.isfinite(f):
                ax.axhline(f, color=style.color(arm), lw=0.7, ls=":",
                           alpha=0.7, zorder=1)
        ax.annotate("chance floor", (0.005, df[floor_col].mean()), fontsize=6,
                    xycoords=("axes fraction", "data"), ha="left", va="bottom",
                    color="0.35")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlabel("band(s) flattened to the arm's own mean")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    use_symlog = (yscale == "symlog" or
                  (yscale == "auto" and _needs_symlog(df[value].to_numpy("float64"))))
    if use_symlog:
        a = np.abs(df[value].to_numpy("float64"))
        a = a[np.isfinite(a) & (a > 0)]
        ax.set_yscale("symlog", linthresh=max(a.min(), 1e-4))
        ax.set_ylabel(ylabel + "  (symlog)")
    return use_symlog


def _conv1_panel(ax, ratios, bands, seeds):
    arms = sorted(ratios["arm"].unique(), key=style.sort_key)
    w = 0.8 / max(len(arms), 1)
    xs = np.arange(len(bands))
    for i, arm in enumerate(arms):
        g = ratios[ratios["arm"] == arm][bands].to_numpy(dtype="float64")
        ax.bar(xs + i * w - 0.4 + w / 2, g.mean(0), width=w,
               color=style.color(arm), linewidth=0, hatch=style.hatch(arm),
               edgecolor="white",
               yerr=(g.max(0) - g.min(0)) / 2 if len(g) >= style.MIN_SEEDS_FOR_ERRORBAR else None,
               error_kw={"lw": 0.7, "ecolor": "0.35"})
    ax.axhline(1.0, color=style.ZERO_LINE, lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(bands)
    ax.set_ylabel("stem conv1 norm / ImageNet init")
    ax.set_title("Where the network could look", loc="left")


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="probe_cache")
    ap.add_argument("--out-dir", default=FIGURES_DIR)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm-key globs to keep, e.g. r0 r1a r1b 'r2a@*' 'r2b@*'")
    ap.add_argument("--yscale", default="auto", choices=("auto", "linear", "symlog"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    metas = cache.load_metas(args.cache_dir, args.arms,
                             require=("occlusion.parquet",))
    seeds = cache.seed_counts(metas)
    bands = metas[0]["band_names"]
    df = pd.concat([deltas(m) for m in metas], ignore_index=True)
    ratios = conv1_ratios(metas, Path(args.cache_dir), bands)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "occlusion_delta.csv", index=False)
    ratios.to_csv(out / "occlusion_conv1.csv", index=False)

    print(f"{len(metas)} arm-seed cache(s): "
          + ", ".join(f"{a}x{n}" for a, n in sorted(seeds.items())))
    print("\nΔAP (macro over road-bearing chips)")
    print(df.pivot_table(index=["arm", "seed"], columns="condition",
                         values="dap").reindex(columns=CONDITIONS)
          .to_string(float_format=lambda v: f"{v:+.4f}"))

    # theta-dependent readouts drop any run whose theta is a fallback (§9).
    keep = {m["run"] for m in metas if cache.theta_usable(m)}
    dropped = sorted({m["run"] for m in metas} - keep)
    if dropped:
        print(f"\nΔIoU@θ* excludes {len(dropped)} run(s) with a fallback θ: "
              + ", ".join(dropped))
    iou_df = df[df["run"].isin(keep)]
    print(f"\nwrote {out / 'occlusion_delta.csv'}, {out / 'occlusion_conv1.csv'}")
    if args.no_figures:
        return 0

    style.apply_rc()
    import matplotlib.pyplot as plt

    n_arms = df["arm"].nunique()
    wide = n_arms > 6
    # ONE panel in the main text: the ΔAP dot plot. The first-conv weight-norm
    # panel was cut — it is an argument about the model's parameters, not about
    # what occluding a band does to the score, and mixing the two invited the
    # reader to treat the second as evidence for the first. Its numbers survive
    # verbatim in `occlusion_conv1.csv`, and `_conv1_panel` is kept below so it
    # can be promoted to an appendix figure without rewriting it.
    #
    # Width tracks what the dot plot alone used to get (it held 2.6/3.6 of the
    # old two-panel box when wide, 1.75/2.75 when not), so the dots keep their
    # spacing instead of stretching to fill the vacated column.
    fig, ax = plt.subplots(figsize=(style.FULL_WIDTH_IN * (0.98 if wide else 0.64),
                                    3.6 if wide else 3.2))
    _dotplot(ax, df, "dap", seeds, "ΔAP (occluded − intact)", args.yscale,
             annotate_rel=True, floor_col="dap_floor")
    # `pad` clears the per-condition relative-drop annotations, which sit at the
    # top of the axes.
    ax.set_title("Band reliance: what the U-Net actually uses", loc="left", pad=14)
    # Figure-level: an in-panel legend lands on the dots as soon as an arm's
    # deltas move.
    # No tight_layout here: the gridspec already carries an explicit wspace,
    # which tight_layout warns about and would override. savefig's tight bbox
    # takes care of the outer margins, legend included.
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=min(len(l), 4), loc="upper center",
               bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    for p in style.save(fig, out, "F2_occlusion"):
        print(f"wrote {p}")

    if not iou_df.empty:
        fig2, ax2 = plt.subplots(figsize=(style.SINGLE_COL_IN * 1.6, 3.0))
        _dotplot(ax2, iou_df, "diou_macro", seeds, "ΔIoU@θ* (macro)", args.yscale)
        ax2.set_title("Band reliance at each arm's operating point", loc="left")
        fig2.tight_layout()
        h2, l2 = ax2.get_legend_handles_labels()
        fig2.legend(h2, l2, ncol=min(len(l2), 3), loc="upper center",
                    bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
        for p in style.save(fig2, out, "A2_occlusion_iou"):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
