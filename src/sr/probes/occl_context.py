"""Occlusion suite v3 — analysis and figures (docs/occlusion_suite_plan.md §3).

    PYTHONPATH=src python -m sr.probes.occl_context --cache-dir probe_cache

Reads only `<cache>/<run>/occl2/` (and `occlusion.parquet` beside it, for the
instrument-C consistency panel). Emits into `--out-dir`:

    occl2_band_region.csv   arm-seed x band x region: Δmargin
    occl2_context.csv       arm-seed x chip x pixel: weighted context
    occl2_robustness.csv    the §4 protocol: every headline at patch 16/32/64
    F6_occl_band_region.*   main: band x region reliance | instrument-C agreement
    F7_weighted_context.*   main: weighted context per arm (HC x lr_sr)
    A6_occl_appendix.*      exemplar Δmargin maps, counterfactual bars

WEIGHTED CONTEXT (O'Sullivan & Dev, IGARSS 2025, Eq. 1)
--------------------------------------------------------
For a pixel p at (y, x) with saliency map S^p normalised to [0, 1],

    W_p = Σ_ij s^p_ij · sqrt((y−i)² + (x−j)²)  /  Σ_ij sqrt((y−i)² + (x−j)²)

— the mean distance from p to every pixel, weighted by that pixel's
importance, over the same sum with every importance set to 1. So it reads as
the PROPORTION OF AVAILABLE CONTEXT, is bounded in [0, 1], and is comparable
across models, which is the whole reason it is here rather than a saliency
map per arm.

Two implementation notes the paper leaves open:

* **the map is patch-constant.** Stride equals patch in this suite, so every
  pixel belongs to exactly one window and s is constant on each. The sums
  therefore collapse to per-patch distance sums — exact, not an approximation,
  and it turns a 512×512 sum per pixel into a 16×16 one.
* **"normalised to [0,1]"** is not defined in the paper. `--norm minmax`
  (default) is the literal reading; `--norm relu` clips the negative Δ (a
  patch whose removal HELPED) to zero first. The choice is written into the
  CSV, and the robustness table reports both — a conclusion that depends on it
  is not reported, per §4.

Δ is `intact − occluded`, so a positive value means occluding that patch cost
the model logit at p: reliance, in the paper's sign convention.

CLAIM DISCIPLINE (spec §5)
--------------------------
Occlusion measures RELIANCE, not information content: the bands are
correlated and the mean fill is off-manifold (which is what the counterfactual
fills exist to blunt). Captions say "reliance on band b within NDVI-defined
vegetation" and "reliance on SR-added structural content" — never "importance
of the region". Everything here is θ-free: GT masks and margins, never a
thresholded prediction.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sr.probes import occlusion, style
from sr.probes.occl_context_extract import REGIONS
from sr.probes.style import FIGURES_DIR

BANDS = ("R", "G", "B", "NIR")


# ------------------------------------------------------------------ loading
def load_metas(cache_dir, arms=None) -> list[dict]:
    """Every `occl2/meta.json` under `cache_dir`, guarded and in figure order.

    Reuses `cache.py`'s guards in spirit and adds the one this suite needs: a
    shared NDVI τ. Two caches extracted at different τ hold different region
    classes under the same names, which is exactly the kind of silent axis
    mismatch the fixture-hash rule exists to stop.
    """
    import fnmatch

    out = []
    for d in sorted(Path(cache_dir).iterdir()):
        f = d / "occl2" / "meta.json"
        if not f.is_file():
            continue
        m = json.loads(f.read_text())
        m["arm"] = style.arm_of(m["run"])
        if arms and not any(fnmatch.fnmatch(m["arm"], p) for p in arms):
            continue
        m["dir"] = d / "occl2"
        m["run_dir"] = d
        # The extraction cache next door must be on the same fixture: the
        # consistency panel pairs the two, and nothing else checks across them.
        e = d / "meta.json"
        if e.exists():
            em = json.loads(e.read_text())
            if em.get("fixture_hash") != m["fixture_hash"]:
                raise SystemExit(
                    f"{m['run']}: occl2 was extracted against fixture "
                    f"{m['fixture_hash']}, the probe cache holds "
                    f"{em['fixture_hash']}. Re-extract the stale one.")
        out.append(m)
    if not out:
        raise SystemExit(
            f"no occl2 cache under {cache_dir} — run "
            "`python -m sr.probes.occl_context_extract` first")
    for key, what in (("fixture_hash", "fixtures"), ("ndvi_tau", "NDVI τ")):
        vals = {str(m.get(key)) for m in out}
        if len(vals) > 1:
            raise SystemExit(
                f"caches disagree on {what} ({', '.join(sorted(vals))}); they "
                "are not on one axis and must not share a figure.")
    return sorted(out, key=lambda m: (style.sort_key(m["arm"]),
                                      m["seed"] if m["seed"] is not None else -1))


def _read(metas, name) -> pd.DataFrame:
    out = []
    for m in metas:
        p = m["dir"] / name
        if not p.exists():
            continue
        out.append(pd.read_parquet(p).assign(run=m["run"], arm=m["arm"],
                                             seed=m["seed"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------- tables
def band_region(conds: pd.DataFrame) -> pd.DataFrame:
    """arm-seed x band x region -> mean Δmargin over chips (paired per chip).

    Δmargin is already `occluded − intact` on the same chip, so the mean is a
    mean of paired differences; no re-pairing is needed here.
    """
    t = conds[conds["band"].notna()]
    return (t.groupby(["run", "arm", "seed", "band", "region"], dropna=False)
            .agg(dmargin=("dmargin", "mean"), n_chips=("chip", "nunique"),
                 sd=("dmargin", "std")).reset_index())


def counterfactual(conds: pd.DataFrame) -> pd.DataFrame:
    """arm-seed x source x region -> mean Δmargin for the substitution fills."""
    t = conds[conds["condition"].str.startswith("cf_", na=False)]
    if t.empty:
        return t
    return (t.groupby(["run", "arm", "seed", "fill", "region"], dropna=False)
            .agg(dmargin=("dmargin", "mean"), n_chips=("chip", "nunique"),
                 affine_shift=("affine_shift", "mean")).reset_index())


# -------------------------------------------------------- weighted context
@lru_cache(maxsize=4096)
def patch_distance_sums(py: int, px: int, size: int, patch: int) -> tuple:
    """Σ of distances from (py, px) to the pixels of each window, row-major.

    Cached across arms: it depends only on the pixel position and the grid, so
    the same road pixel costs this once for the whole figure rather than once
    per checkpoint.
    """
    g = np.arange(size, dtype="float64")
    d = np.sqrt((g[:, None] - py) ** 2 + (g[None, :] - px) ** 2)
    n = size // patch
    return tuple(d.reshape(n, patch, n, patch).sum(axis=(1, 3)).ravel())


def weighted_context(ctx: pd.DataFrame, size: int, norm="minmax") -> pd.DataFrame:
    """One W_p per (run, chip, sampled pixel) — Eq. 1 on the patch-constant map."""
    rows = []
    for (run, arm, seed, chip, pix_k, patch), g in ctx.groupby(
            ["run", "arm", "seed", "chip", "pix_k", "patch"], dropna=False):
        patch = int(patch)
        n = size // patch
        s = np.full(n * n, np.nan)
        s[(g["row"].to_numpy() // patch) * n + g["col"].to_numpy() // patch] = \
            g["dlogit"].to_numpy()
        if np.isnan(s).any():
            raise SystemExit(f"{run} chip {chip} pixel {pix_k}: the sliding pass "
                             "does not cover the grid — cache is incomplete.")
        if norm == "relu":
            s = np.clip(s, 0, None)
            s = s / s.max() if s.max() > 0 else s
        else:
            rng = s.max() - s.min()
            s = (s - s.min()) / rng if rng > 0 else np.zeros_like(s)
        p = int(g["pix"].iloc[0])
        d = np.asarray(patch_distance_sums(p // size, p % size, size, patch))
        rows.append(dict(run=run, arm=arm, seed=seed, chip=chip, pix_k=pix_k,
                         patch=patch, norm=norm, W=float((s * d).sum() / d.sum())))
    return pd.DataFrame(rows)


def patch_sum_sanity(conds: pd.DataFrame, patches: pd.DataFrame,
                     patch: int) -> pd.DataFrame:
    """Does the sliding pass point the same way as the whole-image fill?

    Spec §6.5. A patch far from a road pixel is NOT guaranteed to have zero
    effect — the U-Net's effective receptive field is wide and the margin is a
    whole-chip statistic — so the check that is actually available is
    directional: per chip, the SUM of the patch deltas against the delta of the
    all-band mean fill over the whole image. They are not equal (the model is
    not linear in the fill), so this reports rank correlation and sign
    agreement, and only a NEGATIVE correlation would be evidence of a
    bookkeeping error.
    """
    if patches.empty or conds.empty:
        return pd.DataFrame()
    p = (patches[(patches["fill"] == "mean") & (patches["patch"] == patch)]
         .groupby(["run", "chip"], dropna=False)["dmargin"].sum()
         .rename("patch_sum").reset_index())
    w = (conds[conds["condition"] == "allbands@all"]
         .groupby(["run", "chip"], dropna=False)["dmargin"].mean()
         .rename("whole").reset_index())
    m = p.merge(w, on=["run", "chip"], how="inner")
    if m.empty:
        return pd.DataFrame()
    return (m.groupby("run")
            .apply(lambda g: pd.Series({
                "rho": spearman(g["patch_sum"], g["whole"]),
                "same_sign": float((np.sign(g["patch_sum"]) ==
                                    np.sign(g["whole"])).mean()),
                "n_chips": int(len(g))}), include_groups=False)
            .reset_index())


# ---------------------------------------------------------- instrument C
def occlusion_dap(metas) -> pd.DataFrame:
    """Instrument C's ΔAP per arm-seed x band, on THIS suite's chips only."""
    rows = []
    for m in metas:
        p = m["run_dir"] / "occlusion.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        t = t[t["chip"].isin(set(m["chips_evaluated"]))]
        if t.empty:
            continue
        rows.append(occlusion.deltas({**m, "dir": m["run_dir"]}, table=t))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def consistency(br: pd.DataFrame, dap: pd.DataFrame) -> pd.DataFrame:
    """arm x band: whole-image Δmargin against instrument C's ΔAP."""
    s = (br[br["region"] == "all"].groupby(["arm", "band"], dropna=False)
         ["dmargin"].mean().reset_index())
    if dap.empty:
        return s.assign(dap=np.nan, reliance=np.nan)
    d = (dap[dap["condition"].isin(BANDS)]
         .groupby(["arm", "condition"], dropna=False)["dap"].mean().reset_index()
         .rename(columns={"condition": "band"}))
    m = s.merge(d, on=["arm", "band"], how="left")
    m["reliance"] = -m["dap"]
    return m


def spearman(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])


# ------------------------------------------------------------------ figures
def _band_region_panel(ax, br, arm, seeds):
    """Δmargin per band, grouped by region, for one arm."""
    regions = [r for r in REGIONS]
    w = 0.8 / len(regions)
    xs = np.arange(len(BANDS))
    g = br[br["arm"] == arm]
    for i, region in enumerate(regions):
        v, err = [], []
        for band in BANDS:
            s = g[(g["band"] == band) & (g["region"] == region)]["dmargin"]
            v.append(s.mean() if len(s) else np.nan)
            err.append((s.max() - s.min()) / 2 if len(s) >= style.MIN_SEEDS_FOR_ERRORBAR
                       else 0.0)
        # Region is a WITHIN-panel factor, so it gets lightness inside the arm's
        # own hue rather than a new colour: the arm channel must keep meaning
        # what it means in every other probe figure.
        ax.bar(xs + i * w - 0.4 + w / 2, v, width=w, color=style.color(arm),
               alpha=1.0 - 0.28 * i, linewidth=0, label=region,
               yerr=err if any(err) else None,
               error_kw={"lw": 0.7, "ecolor": "0.35"})
    ax.axhline(0, color=style.ZERO_LINE, lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(BANDS)
    ax.set_title(style.label(arm), loc="left")


def _context_panel(ax, wc, seeds):
    """Weighted context per arm — the distribution over sampled road pixels.

    5-25-50-75-95, drawn by hand rather than as a boxplot so the arm's own
    colour and marker carry through from every other probe figure. The median
    is a MARKER, not a line: a distribution with no spread (one seed, one chip,
    or a genuinely tight arm) would otherwise draw a zero-length line and
    vanish.
    """
    arms = sorted(wc["arm"].unique(), key=style.sort_key)
    for i, arm in enumerate(arms):
        v = wc[wc["arm"] == arm]["W"].to_numpy(dtype="float64")
        if not v.size:
            continue
        q = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([i, i], [q[0], q[4]], color=style.color(arm), lw=0.9, alpha=0.5,
                zorder=2)
        ax.plot([i, i], [q[1], q[3]], color=style.color(arm), lw=3.5, alpha=0.85,
                solid_capstyle="projecting", zorder=3)
        kw = style.marker_kwargs(seeds.get(arm, 1), style.marker(arm), 6.0)
        # A white edge only on the FILLED (replicated) marker: the hollow
        # single-seed marker draws its outline in `color`, and overriding that
        # with white would erase it against the page — which is exactly what it
        # did the first time.
        if "markerfacecolor" not in kw:
            kw["markeredgecolor"] = "white"
        ax.plot([i], [q[2]], color=style.color(arm), zorder=4, **kw)
    ax.set_xticks(range(len(arms)))
    # The arm KEY, not the full label: the legend vocabulary is established in
    # every other figure, and three rotated multi-word labels take more room
    # than the panel itself.
    ax.set_xticklabels(arms, rotation=0)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel("weighted context")
    ax.set_title("How far the model reaches to classify a road pixel\n"
                 "(proportion of available context; O'Sullivan & Dev Eq. 1)",
                 loc="left", fontsize=8)


def _consistency_panel(ax, con, seeds):
    for _, r in con.iterrows():
        if not np.isfinite(r.get("reliance", np.nan)):
            continue
        ax.plot(-r["dmargin"], r["reliance"], color=style.color(r["arm"]),
                **style.marker_kwargs(seeds.get(r["arm"], 1),
                                      style.marker(r["arm"]), 5.0))
        ax.annotate(r["band"], (-r["dmargin"], r["reliance"]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points", color="0.35")
    rho = spearman(-con["dmargin"], con["reliance"])
    ax.axhline(0, color=style.ZERO_LINE, lw=0.8, ls="--")
    ax.set_xlabel("−Δmargin, whole-image band fill (this suite)")
    ax.set_ylabel("−ΔAP (instrument C)")
    ax.set_title(f"Agreement with instrument C: ρ = {rho:.2f}", loc="left")
    return rho


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="probe_cache")
    ap.add_argument("--out-dir", default=FIGURES_DIR)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--norm", default="minmax", choices=("minmax", "relu"),
                    help="how the saliency map is put on [0,1] before Eq. 1")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    metas = load_metas(args.cache_dir, args.arms)
    seeds = {}
    for m in metas:
        seeds[m["arm"]] = seeds.get(m["arm"], 0) + 1
    size = metas[0]["fixture_meta"]["crop"] * metas[0]["fixture_meta"]["upscale"]

    conds = _read(metas, "conditions.parquet")
    ctx = _read(metas, "context.parquet")
    patches = _read(metas, "patches.parquet")
    br = band_region(conds)
    cf = counterfactual(conds)
    dap = occlusion_dap(metas)
    con = consistency(br, dap)
    # The headline uses the registered patch size only. The cache also holds
    # the §4 sweep's sizes, and pooling them would make the main figure an
    # average over occlusion parameters rather than a measurement at one.
    head_patch = metas[0]["patch"]
    wc = (weighted_context(ctx[ctx["patch"] == head_patch], size, args.norm)
          if len(ctx) else pd.DataFrame())

    # --- §4 robustness: the same headline numbers at every patch size run.
    rob = []
    for norm in ("minmax", "relu"):
        for patch, g in (patches.groupby("patch") if len(patches) else []):
            sub = ctx[ctx["patch"] == patch]
            if sub.empty:
                continue
            w = weighted_context(sub, size, norm)
            for arm, gg in w.groupby("arm"):
                rob.append(dict(statistic="weighted_context", arm=arm,
                                patch=int(patch), norm=norm,
                                median=float(gg["W"].median()),
                                n=int(len(gg))))
    rob = pd.DataFrame(rob)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    br.to_csv(out / "occl2_band_region.csv", index=False)
    if len(cf):
        cf.to_csv(out / "occl2_counterfactual.csv", index=False)
    if len(wc):
        wc.to_csv(out / "occl2_context.csv", index=False)
    if len(rob):
        rob.to_csv(out / "occl2_robustness.csv", index=False)
    con.to_csv(out / "occl2_consistency.csv", index=False)
    sanity = patch_sum_sanity(conds, patches, metas[0]["patch"])
    if len(sanity):
        sanity.to_csv(out / "occl2_patch_sanity.csv", index=False)

    print(f"{len(metas)} arm-seed cache(s): "
          + ", ".join(f"{a}x{n}" for a, n in sorted(seeds.items())))
    print(f"τ={metas[0]['ndvi_tau']} patch={metas[0]['patch']} "
          f"K={metas[0]['k_pixels']} chips={len(metas[0]['chips_evaluated'])}")
    print("\nΔmargin, band x region (mean over chips)")
    print(br.pivot_table(index=["arm", "seed", "band"], columns="region",
                         values="dmargin")
          .to_string(float_format=lambda v: f"{v:+.4f}"))
    if len(wc):
        print("\nweighted context (median over sampled road pixels)")
        print(wc.groupby("arm")["W"].median()
              .to_string(float_format=lambda v: f"{v:.4f}"))
    if not dap.empty:
        print(f"\nagreement with instrument C: ρ = "
              f"{spearman(-con['dmargin'], con['reliance']):.2f}")
    if len(sanity):
        print("\nsliding-pass sanity: Σ patch Δmargin vs the whole-image fill")
        print(sanity.to_string(index=False,
                               float_format=lambda v: f"{v:.3f}"))
    if len(rob):
        print("\nrobustness (spec §4) — median weighted context")
        print(rob.pivot_table(index=["arm", "norm"], columns="patch",
                              values="median").to_string(float_format=lambda v: f"{v:.4f}"))
        print("A headline that flips across these settings is not reported.")
    if args.no_figures:
        return 0

    style.apply_rc()
    import matplotlib.pyplot as plt

    arms = sorted(br["arm"].unique(), key=style.sort_key)
    n = len(arms)
    fig, axes = plt.subplots(1, n + 1, figsize=(style.FULL_WIDTH_IN, 2.9))
    axes = np.atleast_1d(axes)
    for k, (ax, arm) in enumerate(zip(axes, arms)):
        # The arm panels share ONE y-axis — comparing arms is the entire point,
        # and per-panel autoscaling would let a small effect look like a large
        # one. The consistency panel is deliberately left out of the sharing:
        # its y is ΔAP, a different quantity.
        if k:
            ax.sharey(axes[0])
            ax.tick_params(labelleft=False)
        _band_region_panel(ax, br, arm, seeds)
    axes[0].set_ylabel("Δmargin (occluded − intact)")
    if con["reliance"].notna().any():
        _consistency_panel(axes[-1], con, seeds)
    else:
        axes[-1].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="upper center", bbox_to_anchor=(0.5, -0.02),
               fontsize=6.5, title="region of the fill")
    fig.tight_layout()
    for p in style.save(fig, out, "F6_occl_band_region"):
        print(f"wrote {p}")

    if len(wc):
        fig2, ax2 = plt.subplots(figsize=(style.SINGLE_COL_IN * 1.7, 3.0))
        _context_panel(ax2, wc, seeds)
        fig2.tight_layout()
        for p in style.save(fig2, out, "F7_weighted_context"):
            print(f"wrote {p}")

    if len(cf) or len(rob):
        fig3, ax3 = plt.subplots(1, 2, figsize=(style.FULL_WIDTH_IN, 2.8))
        if len(cf):
            arms_cf = sorted(cf["arm"].unique(), key=style.sort_key)
            w = 0.8 / max(len(arms_cf), 1)
            xs = np.arange(len(REGIONS))
            for i, arm in enumerate(arms_cf):
                g = cf[(cf["arm"] == arm) & (cf["fill"] == "bic")]
                v = [g[g["region"] == r]["dmargin"].mean() for r in REGIONS]
                ax3[0].bar(xs + i * w - 0.4 + w / 2, v, width=w,
                           color=style.color(arm), hatch=style.hatch(arm),
                           edgecolor="white", linewidth=0, label=style.label(arm))
            ax3[0].set_xticks(xs)
            ax3[0].set_xticklabels(REGIONS)
            ax3[0].axhline(0, color=style.ZERO_LINE, lw=0.8, ls="--")
            ax3[0].set_ylabel("Δmargin, r0 bicubic pasted in")
            ax3[0].set_title("Reliance on SR-added structural content", loc="left")
        else:
            ax3[0].axis("off")
        if len(rob):
            for arm, g in rob[rob["norm"] == args.norm].groupby("arm"):
                g = g.sort_values("patch")
                ax3[1].plot(g["patch"], g["median"], color=style.color(arm),
                            ls=style.linestyle(arm), marker=style.marker(arm),
                            markersize=4, lw=1.0, label=style.label(arm))
            ax3[1].set_xlabel("occlusion patch (px)")
            ax3[1].set_ylabel("median weighted context")
            ax3[1].set_title("Robustness to the occlusion parameters", loc="left")
        else:
            ax3[1].axis("off")
        fig3.tight_layout()
        for p in style.save(fig3, out, "A6_occl_appendix"):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
