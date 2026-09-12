"""Instrument B — stage-wise linear CKA on the U-Net internals (§4, figure F3).

    PYTHONPATH=src python -m sr.probes.cka --cache-dir probe_cache

Reads only the extraction cache. Emits into `--out-dir` (default `figures/probes/`):

    cka_pairs.csv       every arm-seed x arm-seed x stage x convention CKA
    cka_contrasts.csv   the §1 contrasts and the within-arm noise floor
    F3_cka.pdf/.png     main text: own-pipeline | common-input, shared axes
    A3_cka_heatmap.*    appendix: arm x arm at the stage of maximum divergence

THE ESTIMATOR
-------------
Linear CKA (Kornblith et al. 2019) on column-centred activation matrices:

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F)

computed EXACTLY over the whole cached sample rather than as the minibatch
average of Nguyen et al. (2021) that §4 names. The minibatch estimator exists
to bound memory when the activations cannot be held at once; the extraction's
position subsample was chosen so that they can (~25.6k examples against at most
512 channels), and the exact statistic is the thing the minibatch one
approximates. `--minibatch N` computes the averaged form as well, for anyone who
wants the comparison.

Examples are (chip, spatial position) pairs, drawn at indices seeded from the
stage and the chip and from nothing about the arm — so the two matrices in
every pair describe the SAME positions of the SAME chips, and the CKA is a
comparison of representations rather than of samples.

THE NOISE FLOOR IS THE WHOLE ARGUMENT
-------------------------------------
CKA has no absolute scale: 0.6 at one stage means nothing on its own. What it
can support is "these two arms differ MORE than two seeds of one arm differ",
so the within-arm cross-seed CKA is computed at every stage and drawn as a grey
band. No between-arm statement should be made where the contrast sits inside
that band. Arms with one seed contribute no floor of their own and borrow the
pooled one — their contrasts are provisional, which the hollow markers say.

This bears repeating because the deep stages invite exactly the wrong reading.
On this data the contrast falls to ~1e-4 by `enc_layer4`, and the activations
there have a heavy magnitude tail (row norms up to ~400x the median on ordinary,
fully-valid chips), which linear CKA weights heavily. Neither fact settles
anything on its own: a near-zero deep-stage CKA is a finding only if the
within-arm floor at that stage is meaningfully higher.

There is NO "same network, different examples" self-check to fall back on. CKA
compares two representations OF THE SAME EXAMPLES; feeding it disjoint halves
of one network's own rows is not an identity that should return 1.0, and
reading it as one produces a bug report where there is no bug. The only
available reference is two seeds of one arm on the same examples — the floor.

TWO CONVENTIONS, AND WHAT THEIR GAP MEANS
-----------------------------------------
`own` feeds each U-Net its own SR output — the deployed representation, whose
divergence includes everything the generator changed. `common` feeds every
U-Net the same r0 bicubic tensor, so what is left is what the WEIGHTS learned
differently. Divergence present in `own` and absent in `common` was inherited
from the input; divergence surviving into `common` is internal to the U-Net,
which is the §6.2 falsification.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from sr.probes import cache, style
from sr.probes.style import FIGURES_DIR

CONVENTIONS = ("own", "common")
# A contrast is a (treatment, reference) pair. The line takes the TREATMENT
# arm's colour and linestyle: a contrast is not an arm, and colouring by the
# thing being tested keeps §9's encoding meaningful here too.
#
# Two sets, because §1 and §6.2 ask different questions of the same numbers.
# `vs-r0` (the default) is every cached arm against the anchor — the form
# §6.2's registered prediction is stated in ("divergence from r0 concentrates
# in shallow encoder stages") and the one §9's F3 lists first. `s1` is the §1
# table, where each pair ISOLATES one treatment; it is the right set once the
# question is "which factor moved the representation", but its lines share no
# common reference, so they cannot be read against each other vertically.
S1_CONTRASTS = [
    ("r1a", "r0"), ("r2a", "r1a"), ("r2b", "r0"), ("r2a", "r2b"),
    ("r4a", "r4b"), ("r4a", "r3a"), ("r4b", "r3b"),
    ("r4a", "r2a"), ("r4b", "r2b"), ("r3a", "r1a"), ("r3b", "r1b"),
]


# --------------------------------------------------------------------- maths
def _centre(x: np.ndarray) -> np.ndarray:
    return x - x.mean(0, keepdims=True)


def variance_concentration(x: np.ndarray) -> float:
    """Participation ratio of the channel variances, in [1/C, 1].

    Linear CKA is dominated by high-variance directions (Kornblith et al. 2019
    §4), so this says how much of a stage the comparison actually spans: 1.0 =
    every channel contributes equally, 1/C = one channel carries everything.

    Recorded as context, NOT as a correction or a threshold — nothing here
    reweights or filters on it. Whether a low CKA at a given stage is a real
    divergence is settled by the within-arm floor and by nothing else.
    """
    v = np.asarray(x, dtype="float64").var(0)
    t = v.sum()
    return float(t * t / (len(v) * (v * v).sum())) if t > 0 else float("nan")


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Exact linear CKA of two (n_examples, n_features) matrices."""
    x, y = _centre(np.asarray(x, dtype="float64")), _centre(np.asarray(y, dtype="float64"))
    num = np.linalg.norm(x.T @ y, "fro") ** 2
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(num / den) if den > 0 else float("nan")


def minibatch_cka(x: np.ndarray, y: np.ndarray, batch: int) -> float:
    """The Nguyen et al. (2021) averaged form, for comparison with the exact one."""
    n = min(len(x), len(y))
    vals = [linear_cka(x[i:i + batch], y[i:i + batch])
            for i in range(0, n - batch + 1, batch)]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------- cache
def stages_of(metas) -> list[str]:
    """The stage list, required identical across caches sharing a figure."""
    sets = {tuple(m["stages"]) for m in metas if m.get("stages")}
    if len(sets) != 1:
        raise SystemExit(
            f"caches hook {len(sets)} different stage lists; they cannot share "
            "an x axis — re-extract with one architecture.")
    return list(next(iter(sets)))


def _load_stage(metas, conv: str, stage: str):
    """{run: centred (n, C) matrix} for one stage, or {} if this convention was
    not extracted (e.g. `--no-common`)."""
    out = {}
    for m in metas:
        f = m["dir"] / f"cka_{conv}" / f"{stage}.npy"
        if f.exists():
            out[m["run"]] = np.load(f)
    return out


def pair_table(metas, stages, minibatch=None) -> pd.DataFrame:
    """CKA for every run pair, at every stage, under both conventions."""
    by_run = {m["run"]: m for m in metas}
    rows = []
    for conv in CONVENTIONS:
        for stage in stages:
            mats = _load_stage(metas, conv, stage)
            if len(mats) < 2:
                continue
            ns = {v.shape[0] for v in mats.values()}
            if len(ns) > 1:
                raise SystemExit(
                    f"{conv}/{stage}: caches hold {sorted(ns)} examples. The CKA "
                    "examples must be paired, so a differing count means the "
                    "caches were extracted over different chips.")
            n_ex = next(iter(ns))
            c_max = max(v.shape[1] for v in mats.values())
            if n_ex < 4 * c_max:
                print(f"  WARN {conv}/{stage}: {n_ex} examples against {c_max} "
                      f"channels. Linear CKA on fewer examples than ~4x the "
                      "channel count is dominated by rank deficiency, not by "
                      "the representations — raise --positions or drop --limit.")
            for a, b in itertools.combinations(sorted(mats), 2):
                r = {"convention": conv, "stage": stage,
                     "run_a": a, "run_b": b,
                     "arm_a": by_run[a]["arm"], "arm_b": by_run[b]["arm"],
                     "seed_a": by_run[a]["seed"], "seed_b": by_run[b]["seed"],
                     "cka": linear_cka(mats[a], mats[b]),
                     "var_conc_a": variance_concentration(mats[a]),
                     "var_conc_b": variance_concentration(mats[b])}
                if minibatch:
                    r["cka_minibatch"] = minibatch_cka(mats[a], mats[b], minibatch)
                rows.append(r)
    if not rows:
        raise SystemExit("no CKA activations in the cache — extraction was run "
                         "on an architecture with no encoder to hook.")
    return pd.DataFrame(rows)


def noise_floor(pairs: pd.DataFrame) -> pd.DataFrame:
    """Within-arm cross-seed CKA per convention per stage: the floor band."""
    same = pairs[pairs["arm_a"] == pairs["arm_b"]]
    if same.empty:
        return pd.DataFrame(columns=["convention", "stage", "lo", "hi", "mean", "n", "arms"])
    g = same.groupby(["convention", "stage"])["cka"]
    out = g.agg(lo="min", hi="max", mean="mean", n="size").reset_index()
    out["arms"] = ",".join(sorted(same["arm_a"].unique()))
    return out


def resolve_contrasts(spec, arms) -> list[tuple[str, str]]:
    """`vs-r0` / `s1` / explicit `treat:ref` tokens -> [(treatment, reference)]."""
    if spec == ["s1"]:
        return S1_CONTRASTS
    if spec == ["vs-r0"]:
        if "r0" not in arms:
            raise SystemExit(
                "--contrasts vs-r0 needs an r0 cache to reference. Extract it, "
                "or pass --contrasts s1 for the §1 isolating pairs.")
        return [(a, "r0") for a in sorted(arms, key=style.sort_key) if a != "r0"]
    out = []
    for tok in spec:
        if ":" not in tok:
            raise SystemExit(f"--contrasts: {tok!r} is not 'treat:ref', 's1' or "
                             "'vs-r0'")
        t, r = tok.split(":", 1)
        out.append((t, r))
    return out


def contrast_table(pairs: pd.DataFrame, contrasts) -> pd.DataFrame:
    """The requested contrasts, pooled over every cross-seed pair of the arms."""
    rows = []
    for treat, ref in contrasts:
        sel = pairs[((pairs["arm_a"] == treat) & (pairs["arm_b"] == ref))
                    | ((pairs["arm_a"] == ref) & (pairs["arm_b"] == treat))]
        if sel.empty:
            continue
        for (conv, stage), g in sel.groupby(["convention", "stage"]):
            rows.append({"contrast": f"{treat}-{ref}", "treatment": treat,
                         "reference": ref, "convention": conv, "stage": stage,
                         "cka": g["cka"].mean(), "lo": g["cka"].min(),
                         "hi": g["cka"].max(), "n_pairs": len(g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figures
def _stage_label(s: str) -> str:
    return (s.replace("enc_stem", "conv1").replace("enc_layer", "enc")
            .replace("dec_block", "dec"))


def _panel(ax, contrasts, floor, stages, conv, seeds, title):
    xs = np.arange(len(stages))
    sub = contrasts[contrasts["convention"] == conv]
    f = floor[floor["convention"] == conv].set_index("stage").reindex(stages)
    if f["lo"].notna().any():
        who = f["arms"].dropna().iloc[0]
        npair = int(f["n"].max())
        # With a single seed pair the floor is a LINE, not a band: fill_between
        # over lo == hi draws nothing at all, silently hiding the one reference
        # the whole panel is read against. Draw the line always, the band only
        # when there is a spread to show.
        ax.plot(xs, f["mean"], color="0.45", lw=1.0, ls=(0, (4, 2)), zorder=1,
                label=f"within-arm seed floor ({who}, "
                      f"{npair} pair{'s' if npair != 1 else ''})")
        if (f["hi"] - f["lo"]).fillna(0).gt(0).any():
            ax.fill_between(xs, f["lo"], f["hi"], color="0.72", alpha=0.45,
                            lw=0, zorder=0)
    for name, g in sub.groupby("contrast"):
        g = g.set_index("stage").reindex(stages)
        treat = g["treatment"].dropna().iloc[0]
        n = seeds.get(treat, 1)
        ax.plot(xs, g["cka"], color=style.color(treat), ls=style.linestyle(treat),
                lw=1.2, zorder=3, label=name,
                marker=style.marker(treat), markersize=4.0, markeredgewidth=0.8,
                markerfacecolor="none" if n < style.MIN_SEEDS_FOR_ERRORBAR
                else style.color(treat))
        if (g["n_pairs"] > 1).any():
            ax.fill_between(xs, g["lo"], g["hi"], color=style.color(treat),
                            alpha=0.13, lw=0, zorder=2)
    # The encoder/decoder boundary: CKA either side of it is measured on
    # different kinds of feature, and the eye should not read across it.
    n_enc = sum(1 for s in stages if s.startswith("enc"))
    if 0 < n_enc < len(stages):
        ax.axvline(n_enc - 0.5, color="0.55", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([_stage_label(s) for s in stages], rotation=45, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, loc="left")
    ax.set_xlabel("U-Net stage")


def heatmap(pairs, stages, out_dir, stem, conv="own"):
    """Appendix: arm x arm CKA at the stage where the arms diverge most."""
    import matplotlib.pyplot as plt

    p = pairs[(pairs["convention"] == conv) & (pairs["arm_a"] != pairs["arm_b"])]
    if p.empty:
        return []
    stage = p.groupby("stage")["cka"].mean().idxmin()          # most divergent
    s = pairs[(pairs["convention"] == conv) & (pairs["stage"] == stage)]
    arms = sorted(set(s["arm_a"]) | set(s["arm_b"]), key=style.sort_key)
    m = np.full((len(arms), len(arms)), np.nan)
    for i, a in enumerate(arms):
        for j, b in enumerate(arms):
            if i == j:
                sel = s[(s["arm_a"] == a) & (s["arm_b"] == a)]
                m[i, j] = sel["cka"].mean() if len(sel) else 1.0
            else:
                sel = s[((s["arm_a"] == a) & (s["arm_b"] == b))
                        | ((s["arm_a"] == b) & (s["arm_b"] == a))]
                if len(sel):
                    m[i, j] = sel["cka"].mean()
    fig, ax = plt.subplots(figsize=(style.SINGLE_COL_IN * 1.35,
                                    style.SINGLE_COL_IN * 1.2))
    im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(arms)), arms, rotation=45, ha="right")
    ax.set_yticks(range(len(arms)), arms)
    for i in range(len(arms)):
        for j in range(len(arms)):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="w" if m[i, j] < 0.6 else "k")
    ax.set_title(f"linear CKA at {_stage_label(stage)} ({conv}-pipeline)\n"
                 "diagonal = within-arm seed floor", loc="left", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046, label="CKA")
    fig.tight_layout()
    return style.save(fig, out_dir, stem)


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="probe_cache")
    ap.add_argument("--out-dir", default=FIGURES_DIR)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm-key globs to keep, e.g. r0 r1a r1b 'r2a@*' 'r2b@*'")
    ap.add_argument("--contrasts", nargs="*", default=["vs-r0"],
                    help="'vs-r0' (default; §6.2's form), 's1' (the §1 "
                         "isolating table), or explicit 'treat:ref' pairs")
    ap.add_argument("--minibatch", type=int, default=None,
                    help="also compute the Nguyen et al. minibatch estimator "
                         "at this batch size (the exact form is the default)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    metas = cache.load_metas(args.cache_dir, args.arms)
    metas = [m for m in metas if (m["dir"] / "cka_own").is_dir()]
    if len(metas) < 2:
        raise SystemExit("CKA needs at least two arm-seed caches with hooked "
                         "activations; found "
                         f"{len(metas)} under {args.cache_dir}")
    seeds = cache.seed_counts(metas)
    stages = stages_of(metas)
    print(f"{len(metas)} arm-seed cache(s), {len(stages)} stages: "
          + ", ".join(f"{a}x{n}" for a, n in sorted(seeds.items())))

    pairs = pair_table(metas, stages, args.minibatch)
    floor = noise_floor(pairs)
    wanted = resolve_contrasts(args.contrasts, set(seeds))
    contrasts = contrast_table(pairs, wanted)
    missing = [f"{t}-{r}" for t, r in wanted
               if contrasts.empty or f"{t}-{r}" not in set(contrasts["contrast"])]
    if missing:
        print(f"  {len(missing)} contrast(s) not drawable (arm not cached): "
              + ", ".join(missing))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "cka_pairs.csv", index=False)
    pd.concat([contrasts.assign(kind="contrast"),
               floor.assign(kind="floor")], ignore_index=True).to_csv(
        out / "cka_contrasts.csv", index=False)

    if floor.empty:
        print("\nNO within-arm noise floor: every arm has a single seed. Between-"
              "arm CKA differences below are NOT yet interpretable (§4).")
    else:
        print("\nwithin-arm seed floor (own-pipeline), by stage")
        print(floor[floor.convention == "own"].set_index("stage")
              .reindex(stages)[["lo", "hi", "n"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))
    if not contrasts.empty:
        print("\ncontrast CKA (own-pipeline)")
        print(contrasts[contrasts.convention == "own"]
              .pivot_table(index="contrast", columns="stage", values="cka")
              .reindex(columns=stages).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out / 'cka_pairs.csv'}, {out / 'cka_contrasts.csv'}")
    if args.no_figures or contrasts.empty:
        return 0

    style.apply_rc()
    import matplotlib.pyplot as plt

    have = [c for c in CONVENTIONS if (contrasts["convention"] == c).any()]
    fig, axes = plt.subplots(1, len(have), figsize=(style.FULL_WIDTH_IN, 3.3),
                             sharey=True, squeeze=False)
    titles = {"own": "Own pipeline (deployed)",
              "common": "Common input (weights only)"}
    for ax, conv in zip(axes[0], have):
        _panel(ax, contrasts, floor, stages, conv, seeds, titles[conv])
    axes[0][0].set_ylabel("linear CKA")
    fig.tight_layout()
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=min(len(l), 4), loc="upper center",
               bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    for p in style.save(fig, out, "F3_cka"):
        print(f"wrote {p}")
    for p in heatmap(pairs, stages, out, "A3_cka_heatmap"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
