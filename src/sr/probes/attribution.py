"""Instrument E — the attribution figures (docs/attribution_plan.md v2).

    PYTHONPATH=src python -m sr.probes.attribution \
        --attr-dir attr_cache --cache-dir probe_cache

Reads only caches — `attr_extract.py`'s (this instrument) and `extract.py`'s
(occlusion, for the consistency panel). Emits into `--out-dir` (default
`figures/probes/`):

    attribution_mass.csv    per arm-seed x band x region: mass shares
    attribution_band.csv    per arm-seed x band: the share the dot plot draws
    attribution_consistency.csv  per arm x band: IG share vs occlusion dAP
    F9_attribution.pdf/.png main text: band shares | occlusion-consistency
    A9_attribution_regions.*     appendix: where each band's mass sits

Instrument E's IG passes are SHELVED (attribution_plan v3 pivot: the backward
graph is what made them unaffordable on both platforms, and the occlusion
suite answers the same questions forward-only). The figures are numbered F9/A9
rather than F6/A6 so the numbers stay clear of `occl_context.py`, which owns
F6/F7/A6 now. Nothing else about this module changed; it still runs if an
attribution cache exists.

WHAT THE MAIN PANEL CLAIMS, AND WHAT WOULD FALSIFY IT
-----------------------------------------------------
The left panel is each arm's per-band share of the attribution mass; the right
one puts that share against the SAME band's occlusion dAP, one point per band x
arm. Two instruments, one method-free question: does the network's gradient say
it uses the bands that occluding actually costs it? Agreement is the claim, and
plan §8 is explicit that a disagreement stops the instrument rather than being
narrated around — so the correlation is printed whether or not it flatters.

dAP is NEGATED on that axis (occluding a band the network uses makes AP fall,
so reliance is a large negative dAP), which puts agreement on the rising
diagonal instead of asking the reader to invert one axis mentally.

ABSOLUTE MASS BY DEFAULT, SIGNED AVAILABLE
------------------------------------------
Completeness certifies the SIGNED sum: `attr.sum() == f(x) - f(baseline)`. But
"which band does the network use" is a question about magnitude — a band whose
evidence cancels between road and background still carries signal — so the
figures draw the absolute share and `--mass signed` switches. Both are in the
cache and both are written to CSV, so neither reading needs a re-run.

THE TWO GUARDS THIS SCRIPT ADDS
-------------------------------
* **fixture agreement across instruments.** `cache.load_metas` enforces one
  fixture within a cache directory. It cannot see ACROSS two directories, and
  an attribution pass re-run against a newer fixture would otherwise be paired
  happily with a stale occlusion cache. The hashes are compared per run here.
* **one chip set.** The attribution pass runs on a subsample; occlusion ran on
  the whole fixture. Every dAP on the consistency panel is recomputed from
  `occlusion.parquet` restricted to the attributed chips, so the two axes are
  statements about the same ground.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sr.probes import cache, occlusion, style
from sr.probes.attr_extract import REGIONS
from sr.probes.style import FIGURES_DIR

# The occlusion conditions that name ONE band; `RGB` has no attribution twin.
BANDS = ("R", "G", "B", "NIR")


# ------------------------------------------------------------------ loading
def load_attr(metas, space="y", baseline="mean") -> pd.DataFrame:
    """Every arm-seed's mass table, filtered to one space and baseline."""
    out = []
    for m in metas:
        t = pd.read_parquet(m["dir"] / "attr_mass.parquet")
        t = t[(t["space"] == space) & (t["baseline"] == baseline)]
        if t.empty:
            continue
        t = t.assign(run=m["run"], arm=m["arm"], seed=m["seed"])
        out.append(t)
    if not out:
        raise SystemExit(
            f"no cached attribution for space={space!r} baseline={baseline!r} — "
            "run `python -m sr.probes.attr_extract` with those settings first.")
    return pd.concat(out, ignore_index=True)


def check_fixtures(attr_metas, cache_dir: Path) -> dict:
    """Per run, assert the attribution and extraction caches share a fixture.

    `cache.load_metas` guards WITHIN a directory. Across two of them nothing
    stops a re-fixtured attribution pass being paired with a stale occlusion
    cache, and the pairing is exactly what the consistency panel is.
    """
    occ = {}
    for m in attr_metas:
        p = Path(cache_dir) / m["run"] / "meta.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text())
        if e["fixture_hash"] != m["fixture_hash"]:
            raise SystemExit(
                f"{m['run']}: the attribution cache was extracted against "
                f"fixture {m['fixture_hash']} but the probe cache holds "
                f"{e['fixture_hash']}. They are not on one axis — re-extract "
                "the stale one rather than pairing them.")
        occ[m["run"]] = Path(cache_dir) / m["run"]
    return occ


# ------------------------------------------------------------------- tables
def band_shares(df: pd.DataFrame, mass="abs") -> pd.DataFrame:
    """arm-seed x band -> the band's share of the chip's attribution mass.

    Averaged over chips, not pooled: a chip is the unit the fixture samples and
    a pooled sum would weight chips by their attribution magnitude, i.e. by how
    confidently the arm reads them.
    """
    col = "absmass_frac" if mass == "abs" else "mass_frac"
    per_chip = (df.groupby(["run", "arm", "seed", "chip", "band"], dropna=False)[col]
                .sum().reset_index())
    return (per_chip.groupby(["run", "arm", "seed", "band"], dropna=False)[col]
            .mean().reset_index().rename(columns={col: "share"}))


def region_shares(df: pd.DataFrame, mass="abs") -> pd.DataFrame:
    """arm-seed x band x region -> share of that BAND's mass, and enrichment.

    `enrichment` divides the region's share of a band's mass by the region's
    share of the chip's pixels: "other" is most of every chip, so a raw share
    says little and >1 here is the actual concentration claim (plan §9).
    """
    col = "absmass" if mass == "abs" else "mass"
    g = (df.groupby(["run", "arm", "seed", "band", "region"], dropna=False)
         .agg(mass=(col, "sum"), n_px=("n_px", "sum")).reset_index())
    tot = g.groupby(["run", "arm", "seed", "band"], dropna=False)["mass"].transform(
        lambda v: v.abs().sum())
    px = g.groupby(["run", "arm", "seed", "band"], dropna=False)["n_px"].transform("sum")
    g["share"] = g["mass"].abs() / tot.where(tot > 0)
    g["px_share"] = g["n_px"] / px.where(px > 0)
    g["enrichment"] = g["share"] / g["px_share"].where(g["px_share"] > 0)
    return g


def occlusion_dap(attr_metas, occ_dirs, chips_of) -> pd.DataFrame:
    """Per arm-seed x band, dAP recomputed on the ATTRIBUTED chips only.

    `occlusion.deltas` takes a table override precisely so the arithmetic can be
    re-run on a subset without a second GPU pass.
    """
    rows = []
    for m in attr_metas:
        d = occ_dirs.get(m["run"])
        if d is None or not (d / "occlusion.parquet").exists():
            continue
        t = pd.read_parquet(d / "occlusion.parquet")
        keep = set(chips_of[m["run"]])
        t = t[t["chip"].isin(keep)]
        if t.empty:
            continue
        rows.append(occlusion.deltas({**m, "dir": d}, table=t))
    if not rows:
        return pd.DataFrame(columns=["run", "arm", "seed", "condition", "dap"])
    return pd.concat(rows, ignore_index=True)


def _spearman(a, b):
    """Rank correlation with average ranks for ties; no scipy dependency."""
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    ra, rb = a[ok].rank(), b[ok].rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def consistency(shares: pd.DataFrame, dap: pd.DataFrame) -> pd.DataFrame:
    """arm x band: mean IG share against mean -dAP, the two axes of the scatter."""
    s = (shares.groupby(["arm", "band"], dropna=False)["share"].mean().reset_index())
    d = dap[dap["condition"].isin(BANDS)]
    d = (d.groupby(["arm", "condition"], dropna=False)["dap"].mean().reset_index()
         .rename(columns={"condition": "band", "dap": "dap"}))
    m = s.merge(d, on=["arm", "band"], how="left")
    m["reliance"] = -m["dap"]
    return m


# ------------------------------------------------------------------ figures
def _band_dotplot(ax, shares, seeds, ylabel):
    """The F2 idiom, re-used deliberately: same encoding, so a reader who has
    read the occlusion figure already knows how to read this one."""
    arms = sorted(shares["arm"].unique(), key=style.sort_key)
    w = 0.72 / max(len(arms), 1)
    msize = 5.5 if len(arms) <= 6 else 4.0
    for i, arm in enumerate(arms):
        g = shares[shares["arm"] == arm]
        off = i * w - 0.36 + w / 2
        for j, band in enumerate(BANDS):
            v = g[g["band"] == band]["share"].to_numpy(dtype="float64")
            if not v.size:
                continue
            ax.plot(np.full(v.size, j + off), v, color=style.color(arm), zorder=3,
                    **style.marker_kwargs(seeds.get(arm, 1), style.marker(arm), msize))
            if v.size >= style.MIN_SEEDS_FOR_ERRORBAR:
                ax.plot([j + off, j + off], [v.min(), v.max()],
                        color=style.color(arm), lw=0.9, alpha=0.6, zorder=2)
        ax.plot([], [], color=style.color(arm), ls=style.linestyle(arm),
                label=style.label(arm),
                **{k: v for k, v in style.marker_kwargs(
                    seeds.get(arm, 1), style.marker(arm), msize).items()
                   if k != "linestyle"})
    # An equal split is the null the eye needs: four bands, so 0.25 each.
    ax.axhline(1 / len(BANDS), color=style.ZERO_LINE, lw=0.8, ls="--", zorder=1)
    ax.annotate("equal split", (0.005, 1 / len(BANDS)), fontsize=6,
                xycoords=("axes fraction", "data"), ha="left", va="bottom",
                color="0.35")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS)
    ax.set_xlabel("band")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(BANDS) - 0.4)


def _consistency_panel(ax, con, seeds):
    """IG share vs occlusion reliance, one point per band x arm.

    `seeds` is passed through rather than assumed: the hollow-marker cue means
    "single seed, no within-arm noise floor" everywhere else in the suite, and
    filling every point here would quietly promote a provisional arm.
    """
    for _, r in con.iterrows():
        if not np.isfinite(r.get("reliance", np.nan)):
            continue
        ax.plot(r["share"], r["reliance"], color=style.color(r["arm"]),
                **style.marker_kwargs(seeds.get(r["arm"], 1),
                                      style.marker(r["arm"]), 5.0))
        ax.annotate(r["band"], (r["share"], r["reliance"]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points", color="0.35")
    rho = _spearman(con["share"], con["reliance"])
    ax.set_xlabel("IG share of attribution mass")
    ax.set_ylabel("occlusion reliance (−ΔAP)")
    ax.axhline(0, color=style.ZERO_LINE, lw=0.8, ls="--", zorder=1)
    ax.set_title(f"Do the two instruments agree?  Spearman ρ = {rho:.2f}",
                 loc="left")
    return rho


def _region_panel(ax, reg, band, seeds):
    """Enrichment of one band's mass in each region, per arm (A6).

    Seed range is drawn as a bar only at n >= 3, the suite's rule: a range over
    two points looks like a measured spread and is not one.
    """
    arms = sorted(reg["arm"].unique(), key=style.sort_key)
    w = 0.8 / max(len(arms), 1)
    xs = np.arange(len(REGIONS))
    for i, arm in enumerate(arms):
        g = reg[(reg["arm"] == arm) & (reg["band"] == band)]
        per = [g[g["region"] == r]["enrichment"].to_numpy(dtype="float64")
               for r in REGIONS]
        v = np.array([p.mean() if p.size else np.nan for p in per])
        err = None
        if seeds.get(arm, 1) >= style.MIN_SEEDS_FOR_ERRORBAR:
            err = np.array([(p.max() - p.min()) / 2 if p.size else 0.0
                            for p in per])
        ax.bar(xs + i * w - 0.4 + w / 2, v, width=w, color=style.color(arm),
               linewidth=0, hatch=style.hatch(arm), edgecolor="white",
               yerr=err, error_kw={"lw": 0.7, "ecolor": "0.35"})
    ax.axhline(1.0, color=style.ZERO_LINE, lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(REGIONS)
    ax.set_title(band, loc="left")


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr-dir", default="attr_cache")
    ap.add_argument("--cache-dir", default="probe_cache",
                    help="extract.py's cache, for the occlusion comparison")
    ap.add_argument("--out-dir", default=FIGURES_DIR)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--space", default="y", choices=("y", "x"),
                    help="y = E1 (U-Net only, the main text), x = E2")
    ap.add_argument("--baseline", default="mean",
                    help="which cached baseline to draw ('mean' is occlusion's own fill)")
    ap.add_argument("--mass", default="abs", choices=("abs", "signed"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    metas = cache.load_metas(args.attr_dir, args.arms,
                             require=("attr_mass.parquet",))
    seeds = cache.seed_counts(metas)
    df = load_attr(metas, args.space, args.baseline)
    kept = set(df["run"])
    metas = [m for m in metas if m["run"] in kept]

    shares = band_shares(df, args.mass)
    reg = region_shares(df, args.mass)
    occ_dirs = check_fixtures(metas, Path(args.cache_dir))
    dap = occlusion_dap(metas, occ_dirs, {m["run"]: m["chips"] for m in metas})
    con = consistency(shares, dap)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shares.to_csv(out / "attribution_band.csv", index=False)
    reg.to_csv(out / "attribution_mass.csv", index=False)
    con.to_csv(out / "attribution_consistency.csv", index=False)

    print(f"{len(metas)} arm-seed cache(s): "
          + ", ".join(f"{a}x{n}" for a, n in sorted(seeds.items())))
    print(f"space={args.space} baseline={args.baseline!r} mass={args.mass} "
          f"chips={metas[0]['n_chips']} target={metas[0]['target']}")
    resid = [m.get("resid_rel_max") for m in metas if m.get("resid_rel_max")]
    if resid:
        print(f"worst relative completeness residual across arms: {max(resid):.2e}")
    print("\nper-band share of attribution mass")
    print(shares.pivot_table(index=["arm", "seed"], columns="band", values="share")
          .reindex(columns=list(BANDS)).to_string(float_format=lambda v: f"{v:.3f}"))
    if not dap.empty:
        print(f"\nconsistency with occlusion (same chips): "
              f"Spearman ρ = {_spearman(con['share'], con['reliance']):.2f}")
    else:
        print("\nno occlusion cache alongside these runs — the consistency panel "
              "is empty. Point --cache-dir at extract.py's output.")
    print(f"\nwrote {out / 'attribution_band.csv'}, {out / 'attribution_mass.csv'}, "
          f"{out / 'attribution_consistency.csv'}")
    if args.no_figures:
        return 0

    style.apply_rc()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(style.FULL_WIDTH_IN, 3.2))
    _band_dotplot(axes[0], shares, seeds,
                  ("share of |attribution|" if args.mass == "abs"
                   else "signed share of attribution"))
    axes[0].set_title("Where the gradient says the signal is", loc="left")
    if con["reliance"].notna().any():
        _consistency_panel(axes[1], con, seeds)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=min(len(l), 4), loc="upper center",
               bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    for p in style.save(fig, out, "F9_attribution"):
        print(f"wrote {p}")

    fig2, axes2 = plt.subplots(1, len(BANDS),
                               figsize=(style.FULL_WIDTH_IN, 2.6), sharey=True)
    for ax, band in zip(axes2, BANDS):
        _region_panel(ax, reg, band, seeds)
    axes2[0].set_ylabel("enrichment (mass share / pixel share)")
    fig2.suptitle("Where each band's attribution mass sits", x=0.02, ha="left")
    fig2.tight_layout()
    h2, l2 = axes[0].get_legend_handles_labels()
    fig2.legend(h2, l2, ncol=min(len(l2), 4), loc="upper center",
                bbox_to_anchor=(0.5, -0.04), fontsize=6.5)
    for p in style.save(fig2, out, "A9_attribution_regions"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
