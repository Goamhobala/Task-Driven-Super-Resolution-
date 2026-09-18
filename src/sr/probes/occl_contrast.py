"""Paired OCCLUSION contrast — where joint training moved what the model uses.

    PYTHONPATH=src python -m sr.probes.occl_contrast \
        --runs-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns \
        --runs-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns/refits \
        --pairs r2a:r1a r2b:r1b r4a:r3a r4b:r3b \
        --sr-dir models/SEN2SRLite_RGBN --chips 315 --device mps

One 1x3 panel per (pair, chip): model 1's occlusion map, model 2's occlusion
map, and their difference. The registered pairs are `joint − frozen` within a
generator and a hard-constraint state — r2a−r1a and r2b−r1b on SEN2SR,
r4a−r3a and r4b−r3b on SR4RS — so a row of four figures reads as the same
question asked four times.

NOT `contrast.py`, AND NOT `saliency.py`
----------------------------------------
Three neighbours, three questions, and the difference matters when reading any
of them:

* `saliency.py` runs an occluder over ONE model and asks "what did it use".
* `contrast.py` swaps the IMAGE under one model's U-Net and asks "where does
  the other generator's output change this reader's evidence". One reader,
  two stimuli.
* here: each model is occluded IN ITS OWN DEPLOYED PIPELINE — its own SR
  output, its own `band_mean` fill, its own z-score, its own U-Net — and the
  two resulting sensitivity maps are differenced. Two readers, each on its own
  stimulus, which is what "these are two different models" actually means.

So a positive (red) pixel in panel 3 is "model 1 relies on this location more
than model 2 does", never "model 1 scores better here". Reliance, not skill —
`occl_context.py`'s claim discipline applies unchanged.

WHAT IS OCCLUDED, AND WHAT IS READ OUT
--------------------------------------
The occluder is `saliency.py`'s: a patch flattened to the checkpoint's own
`band_mean` (exactly zero after its z-score, matching instrument C), swept at
stride < patch and accumulated over the windows covering each location.

`--target margin` (default) reads the road MARGIN — mean logit over GT road
minus mean over background, the suite's scalar. It needs no pixel of interest,
so one chip gives one map per model and the 1x3 is complete on its own. The
GT mask is arm-independent, so both models are scored on exactly the same
functional; that is what makes the difference a statement about the models.

`--target pixel` reads single road pixels instead — the paper's geometry, one
figure per marked pixel. The pixels come from `saliency.spread_pixels`, which
is seeded from the fixture hash and the chip index alone, so both models are
read at the SAME pixels.

THE SCALE CAVEAT, AND WHY THERE IS A `--normalise` FLAG
-------------------------------------------------------
Two checkpoints do not share a logit scale. A raw difference map therefore
mixes "model 1 leans on a different place" with "model 1 is more confident
everywhere" — a globally sharper model paints the whole difference panel red
without looking anywhere new. Both readings are offered and neither is hidden:

* `--normalise none` (default) differences the RAW Δ maps. Each panel's own
  peak and the intact readout are printed in its title, so the scale gap the
  difference inherits is visible in the figure rather than implied by it.
* `--normalise peak` divides each map by its own positive 99.8th percentile
  first, so the difference is about WHERE, net of how strongly. The share of
  the raw gap that survives is the part that is not a global gain.

`occl_contrast_region_stats_<pair>.csv` carries both, per region, so a claim
can be checked against the one it was not drawn on.

NO AFFINE MATCHING, DELIBERATELY — as in `contrast.py`. Nothing is pasted
across a boundary here: each model is occluded inside its own image, so there
is no seam to match and the generators' radiometric difference is part of what
distinguishes the two models rather than an artefact of the intervention.

SEEDS
-----
`--pairs` prefers a seed the two arms SHARE, because a within-arm seed
difference would otherwise ride along inside the contrast. Where the arms share
none (r2b/r1b, r4a/r3a on the current run set) it says so loudly and stamps
`seeds_matched: false` into the meta; read those two panels knowing the
difference carries seed variation as well as the treatment. `r2a.42:r1a.42`
pins a seed explicitly.

fp32, `.eval()`, no autograd — the suite's contract. One model is loaded at a
time, so the pass fits the 8 GB MPS budget the suite is built for.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sr.probes.attr_extract import NDVI_TAU, region_masks
from sr.probes.contrast import region_stats
from sr.probes.extract import (BAND_NAMES, CKPT_GLOB, RUN_GLOB, SR_DIR,
                               discover, load_fixture, load_model, read_chip,
                               run_meta)
from sr.probes.make_fixtures import FIXTURE_DIR
from sr.probes.occl_context_extract import Decoder, margin_of
from sr.probes.saliency import PATCH, STRIDE, spread_pixels, starts
from sr.probes.style import FIGURES_DIR

# The registered pairs: `joint − frozen`, within a generator and within a
# hard-constraint state. Both factors have to be held or the contrast is not
# "what joint training did" — it is that plus a generator swap.
DEFAULT_PAIRS = ("r2a:r1a", "r2b:r1b", "r4a:r3a", "r4b:r3b")
PEAK_PCT = 99.8


# ------------------------------------------------------------------- the pass
def occl_field(dec, y, readout, patch, stride, batch, torch, band=None,
               log=None):
    """(T, H, W) mean Δreadout per location, and the intact readout (T,).

    `saliency.saliency_maps` generalised from "the logit at these pixels" to
    any readout of the logit field, so the margin target and the pixel target
    share one accumulate-and-divide rather than drifting apart in two copies.
    `readout` maps (B, H, W) logits -> (B, T); with a pixel gather it
    reproduces `saliency_maps` exactly (pinned in tests).

    Sign follows the paper and the rest of the suite: Δ = intact − occluded, so
    positive means occluding there COST the model evidence.
    """
    h = y.shape[-1]
    with torch.no_grad():
        r0 = np.asarray(readout(dec.logits(y.unsqueeze(0))).cpu(), dtype="float64")[0]
    acc = np.zeros((len(r0), h, h), dtype="float64")
    cov = np.zeros((h, h), dtype="float64")
    wins = [(r, c) for r in starts(h, patch, stride)
            for c in starts(h, patch, stride)]
    bm = dec.m.band_mean.reshape(-1)
    fill = bm.reshape(-1, 1, 1) if band is None else bm[band]
    sl = slice(None) if band is None else slice(band, band + 1)
    t0 = time.perf_counter()
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        v = y.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
        for k, (r, c) in enumerate(chunk):
            v[k, sl, r:r + patch, c:c + patch] = fill
        with torch.no_grad():
            rv = np.asarray(readout(dec.logits(v)).cpu(), dtype="float64")
        for k, (r, c) in enumerate(chunk):
            acc[:, r:r + patch, c:c + patch] += (r0 - rv[k])[:, None, None]
            cov[r:r + patch, c:c + patch] += 1
        if log and (i // batch) % log == 0:
            done = min(i + batch, len(wins))
            el = time.perf_counter() - t0
            print(f"      {done}/{len(wins)} windows  {el:.0f}s "
                  f"({el / max(done, 1) * len(wins):.0f}s projected)", flush=True)
    return acc / np.maximum(cov, 1), r0


def margin_readout(road_t, torch):
    """(B,H,W) logits -> (B, 1) road margin. The default target.

    `road_t` is the 2-D GT mask, not a flat one: `margin_of` indexes (B, H, W)
    with it directly, which is the convention `occl_context_extract` sets.
    """
    return lambda lg: margin_of(lg, road_t).reshape(-1, 1)


def pixel_readout(pix, y, torch):
    """(B,H,W) logits -> (B, len(pix)) logits at flat indices `pix`."""
    idx = torch.as_tensor(np.asarray(pix), device=y.device)
    return lambda lg: lg.reshape(lg.shape[0], -1)[:, idx]


def peak_of(m) -> float:
    """The positive scale of one map — its `PEAK_PCT` percentile, clipped at 0.

    The percentile rather than the max: a single window can land on a road
    junction and put the whole panel's scale out of reach of everything else.
    """
    return float(np.percentile(np.clip(m, 0, None), PEAK_PCT))


def normalise_pair(a, b, how):
    """The two maps as they are differenced, plus the divisor used for each.

    `peak` puts both maps on their own positive scale before subtracting, which
    is what separates "looks somewhere else" from "is louder everywhere".
    """
    if how == "none":
        return a, b, 1.0, 1.0
    pa, pb = peak_of(a) or 1.0, peak_of(b) or 1.0
    return a / pa, b / pb, pa, pb


# -------------------------------------------------------------- pair plumbing
def parse_pair(spec: str):
    """`r2a:r1a` or `r2a.42:r1a.42` -> ((arm, seed|None), (arm, seed|None))."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise SystemExit(f"--pairs item {spec!r} is not TREAT:BASE")

    def one(tok):
        arm, _, seed = tok.partition(".")
        if not arm:
            raise SystemExit(f"--pairs item {spec!r} has an empty arm")
        if seed and not seed.isdigit():
            raise SystemExit(f"--pairs item {spec!r}: seed {seed!r} is not a number")
        return arm, (int(seed) if seed else None)

    return one(parts[0]), one(parts[1])


def resolve_side(runs, arm, seed):
    """Run dirs of one arm, as {seed: path}. Refuses an arm with no run."""
    cand = {run_meta(r)["seed"]: r for r in runs if run_meta(r)["arm"] == arm}
    if not cand:
        raise SystemExit(
            f"no discovered run is arm {arm!r} — check --runs-dir / --include")
    if seed is not None and seed not in cand:
        raise SystemExit(f"arm {arm} has no seed {seed}; available: "
                         f"{sorted(k for k in cand if k is not None)}")
    return cand


def resolve_pair(runs, treat, base):
    """(treat_run, base_run, seeds_matched) for one parsed pair.

    Prefers a seed the two arms SHARE, because an unmatched pair carries
    within-arm seed variation inside a contrast that is supposed to be about
    the treatment. Falls back to the lowest seed on each side and reports that
    it did — the caller warns, and the meta records it.
    """
    (a_arm, a_seed), (b_arm, b_seed) = treat, base
    a, b = resolve_side(runs, a_arm, a_seed), resolve_side(runs, b_arm, b_seed)
    if a_seed is None and b_seed is None:
        shared = sorted(s for s in set(a) & set(b) if s is not None)
        if shared:
            return a[shared[0]], b[shared[0]], True
    lo = lambda d, s: d[s] if s is not None else d[sorted(d, key=lambda k: (k is None, k))[0]]
    ra, rb = lo(a, a_seed), lo(b, b_seed)
    return ra, rb, run_meta(ra)["seed"] == run_meta(rb)["seed"]


# ------------------------------------------------------------------- the figure
def plot_triptych(axes, m_a, m_b, diff, road, labels, vmax, dmax):
    """model 1 | model 2 | model 1 − model 2.

    The first two share ONE sequential scale and are clipped at zero — the
    convention `saliency.plot_panels` fixes, where negative means occluding
    there HELPED and is not reliance. Drawing them on separate scales would
    make the panels look alike however far apart the two models are, and would
    then contradict the difference panel beside them.

    The third is diverging and centred, because both signs occur and a
    sequential map would hide the half of the finding where model 2 leans
    harder.
    """
    ims = []
    for ax, m, lab in zip(axes[:2], (m_a, m_b), labels[:2]):
        ims.append(ax.imshow(np.clip(m, 0, None), cmap="hot_r", vmin=0, vmax=vmax))
        ax.set_title(lab, fontsize=7.5)
    ims.append(axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax))
    axes[2].set_title(labels[2], fontsize=7.5)
    for ax in axes:
        # GT road as a contour on top, never a wash underneath: the reader
        # needs the structure to judge whether either model reached for the
        # road at all.
        if road.any():
            ax.contour(road.astype(float), levels=[0.5], colors=["#3b7bbf"],
                       linewidths=0.45, alpha=0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    return ims[0], ims[2]


def map_stem(chip_idx, target, fill, patch, stride, arm, seed, tag=""):
    """Cache name. Geometry is IN the name, so `--reuse-maps` cannot silently
    mix a patch-16 map into a patch-32 figure."""
    return (f"chip{chip_idx:04d}_{target}{tag}_{fill}_p{patch}s{stride}_"
            f"{arm}_seed{seed}")


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS),
                    help="TREAT:BASE per pair, arms resolved against "
                         "--runs-dir. `r2a.42:r1a.42` pins seeds.")
    ap.add_argument("--run", default=None,
                    help="explicit treatment run dir (with --ref-run), "
                         "instead of --pairs")
    ap.add_argument("--ref-run", default=None,
                    help="explicit baseline run dir, the one SUBTRACTED")
    ap.add_argument("--runs-dir", action="append", default=None,
                    help="repeatable; SRruns and SRruns/refits are two dirs")
    ap.add_argument("--include", default=RUN_GLOB)
    ap.add_argument("--ckpt-glob", default=CKPT_GLOB)
    ap.add_argument("--sr-dir", default=SR_DIR)
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--out-dir", default=str(Path(FIGURES_DIR) / "occl_contrast"))
    ap.add_argument("--chips", type=int, nargs="+", required=True)
    ap.add_argument("--target", default="margin", choices=("margin", "pixel"),
                    help="margin: the suite's road margin, one map per chip "
                         "and no pixel to choose. pixel: the paper's geometry, "
                         "one figure per marked road pixel.")
    ap.add_argument("--n-pixels", type=int, default=1,
                    help="--target pixel only; both models read the SAME "
                         "pixels (seeded from the fixture, not the arm)")
    ap.add_argument("--fill", default="all", choices=("all",) + BAND_NAMES,
                    help="'all' flattens every band in the window; a band name "
                         "flattens only that one, giving the per-band contrast")
    ap.add_argument("--patch", type=int, default=PATCH)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--normalise", default="none", choices=("none", "peak"),
                    help="what the difference panel is drawn on — see module "
                         "docstring. The CSV carries both regardless.")
    ap.add_argument("--tau", type=float, default=NDVI_TAU)
    ap.add_argument("--clip", type=float, default=99.0,
                    help="percentile setting the difference panel's symmetric "
                         "colour limit")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--reuse-maps", action="store_true",
                    help="redraw from the .npy maps already in --out-dir. "
                         "Restyling a figure should not cost 8000 forwards.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if (args.run is None) != (args.ref_run is None):
        raise SystemExit("--run and --ref-run come as a pair")

    fixture = load_fixture(Path(args.fixture_dir))
    meta_fx, chips, _px, fx_hash = fixture
    if args.dataset_dir:
        meta_fx["dataset_dir"] = str(args.dataset_dir)
    crop, up = meta_fx["crop"], meta_fx["upscale"]
    size = crop * up

    # ---- which two checkpoints, per pair
    pairs = []
    if args.run:
        a, b = Path(args.run), Path(args.ref_run)
        pairs.append((a, b, run_meta(a)["seed"] == run_meta(b)["seed"]))
    else:
        runs = discover(args.runs_dir or [], args.include, args.ckpt_glob)
        if not runs:
            raise SystemExit("no runs discovered — check --runs-dir / --include")
        for spec in args.pairs:
            t, base = parse_pair(spec)
            pairs.append(resolve_pair(runs, t, base))

    n_win = len(starts(size, args.patch, args.stride)) ** 2
    total = n_win * len(args.chips) * 2 * len(pairs)
    # ~3.8 fwd/s is MEASURED on this project's 8 GB M-series Mac at batch 4-8,
    # 512 px chips, resnet34 U-Net (smoke run 2026-09-12). `saliency.py`'s
    # banner quotes 17 fwd/s, which this machine does not reach — a 4x
    # underestimate is the difference between "go for lunch" and "go to bed".
    print(f"{len(pairs)} pair(s) x {len(args.chips)} chip(s) x 2 models x "
          f"{n_win} windows (patch {args.patch}, stride {args.stride}) = "
          f"{total} forwards; ~{total / 3.8 / 60:.0f} min at ~3.8 fwd/s "
          f"(measured on MPS; a cluster GPU is far faster). "
          f"Target {args.target}, fill {args.fill}.")
    for t, b, matched in pairs:
        mt, mb = run_meta(t), run_meta(b)
        flag = "" if matched else "   <-- SEEDS NOT MATCHED"
        print(f"  {mt['arm']} (seed {mt['seed']})  −  "
              f"{mb['arm']} (seed {mb['seed']}){flag}")
        if not matched:
            print(f"      {mt['arm']} and {mb['arm']} share no seed on this run "
                  "set; the difference carries seed variation as well as the "
                  "treatment. Read it qualitatively.")
    for i in args.chips:
        print(f"  chip {i:>3} road_px={chips[i]['road_px']:>6} {chips[i]['tile']}")
    if args.dry_run:
        return 0

    if args.device is None:
        import torch
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    import torch
    from sr.probes import style
    from sr.viz_models import find_ckpt

    band = None if args.fill == "all" else BAND_NAMES.index(args.fill)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style.apply_rc()
    import matplotlib.pyplot as plt

    rows, wrote = [], []
    for treat_run, base_run, matched in pairs:
        arm_t, arm_b = run_meta(treat_run), run_meta(base_run)
        pair = f"{arm_t['arm']}_{arm_b['arm']}"
        print(f"\n=== {pair}  ({treat_run.name}  −  {base_run.name})", flush=True)

        # One model in memory at a time: two loaded generators plus their
        # U-Nets do not fit the 8 GB MPS budget the rest of the suite is
        # built for, and nothing here needs them simultaneously.
        got, chip_ctx = {}, {}
        for side, run, meta in (("a", treat_run, arm_t), ("b", base_run, arm_b)):
            ckpt = find_ckpt(run, args.ckpt_glob)
            if ckpt is None:
                raise SystemExit(f"{run} holds no {args.ckpt_glob}")
            dec = Decoder(load_model(ckpt, Path(args.sr_dir), args.device), torch)
            got[(side, "ckpt")] = ckpt
            print(f"  {meta['arm']} seed {meta['seed']}  <- {ckpt.name}", flush=True)
            bands = tuple(dec.m.hparams.bands)
            for i in args.chips:
                chip = chips[i]
                x_np, mask = read_chip(Path(meta_fx["dataset_dir"]), chip, crop,
                                       up, bands, meta_fx["mask_dirname"])
                if not mask.any():
                    # The margin is undefined without road, and a marked pixel
                    # has nothing to mark. The suite drops these chips too.
                    print(f"    chip {i} has no GT road — skipped")
                    continue
                if i in chip_ctx and chip_ctx[i]["bands"] != bands:
                    raise SystemExit(
                        f"{arm_t['arm']} reads bands {chip_ctx[i]['bands']} and "
                        f"{meta['arm']} reads {bands}: the two models are not "
                        "looking at the same image, so nothing here differences.")
                masks = region_masks(x_np, mask, up, args.tau)
                pix = (spread_pixels(mask, args.n_pixels, fx_hash, i)
                       if args.target == "pixel" else None)
                chip_ctx.setdefault(i, dict(masks=masks, pix=pix, bands=bands))

                y = dec.sr(torch.from_numpy(np.ascontiguousarray(x_np))[None]
                           .to(args.device))[0]
                if args.target == "margin":
                    road_t = torch.from_numpy(
                        np.ascontiguousarray(masks["road"])).to(y.device)
                    readout, tags = margin_readout(road_t, torch), [""]
                else:
                    readout = pixel_readout(pix, y, torch)
                    tags = [f"_px{int(p) // size:03d}_{int(p) % size:03d}"
                            for p in pix]
                stems = [map_stem(i, args.target, args.fill, args.patch,
                                  args.stride, meta["arm"], meta["seed"], t)
                         for t in tags]
                cached = [out / f"{s}.npy" for s in stems]
                if args.reuse_maps and all(p.exists() for p in cached):
                    maps = np.stack([np.load(p) for p in cached]).astype("float64")
                    with torch.no_grad():
                        r0 = np.asarray(readout(dec.logits(y.unsqueeze(0))).cpu(),
                                        dtype="float64")[0]
                    print(f"    chip {i}: reused {len(cached)} cached map(s)")
                else:
                    print(f"    chip {i} ({chip['tile']})", flush=True)
                    maps, r0 = occl_field(dec, y, readout, args.patch,
                                          args.stride, args.batch, torch,
                                          band=band, log=args.log_every)
                    for m, p in zip(maps, cached):
                        np.save(p, m.astype("float32"))
                got[(side, i)] = (maps, r0, tags)
                got[(side, i, "y")] = y.detach().float().cpu().numpy()
            del dec
            if args.device == "cuda":
                torch.cuda.empty_cache()
            elif args.device == "mps":
                torch.mps.empty_cache()

        # ---- one triptych per chip (x per marked pixel, in pixel mode)
        for i in args.chips:
            if ("a", i) not in got or ("b", i) not in got:
                continue
            maps_a, r0_a, tags = got[("a", i)]
            maps_b, r0_b, _ = got[("b", i)]
            ctx = chip_ctx[i]
            for k, tag in enumerate(tags):
                m_a, m_b = maps_a[k], maps_b[k]
                na, nb, pa, pb = normalise_pair(m_a, m_b, args.normalise)
                diff = na - nb
                vmax = max(peak_of(m_a), peak_of(m_b)) or 1.0
                dmax = float(np.percentile(np.abs(diff), args.clip)) or 1.0

                unit = "Δ margin" if args.target == "margin" else "Δ logit"
                # Three short lines, not two long ones: a third of the full
                # column is ~2.2 in and `style.label` alone nearly fills it, so
                # appending the seed to it runs the titles into each other.
                labels = [
                    f"{style.label(arm_t['arm'])}\nseed {arm_t['seed']}   "
                    f"intact {r0_a[k]:+.2f}\npeak {peak_of(m_a):+.3f}",
                    f"{style.label(arm_b['arm'])}\nseed {arm_b['seed']}   "
                    f"intact {r0_b[k]:+.2f}\npeak {peak_of(m_b):+.3f}",
                    f"{arm_t['arm']} − {arm_b['arm']}\n"
                    + ("each ÷ its own peak\n" if args.normalise == "peak"
                       else "raw difference\n")
                    + f"mean {diff.mean():+.3f}   |mean| {np.abs(diff).mean():.3f}",
                ]
                fig, axg = plt.subplots(
                    1, 3, figsize=(style.FULL_WIDTH_IN * 0.94, 3.15))
                axes = np.atleast_1d(axg).ravel()
                im_s, im_d = plot_triptych(axes, m_a, m_b, diff,
                                           ctx["masks"]["road"], labels,
                                           vmax, dmax)
                # add_axes colourbars and a tight bbox do not agree; contrast.py
                # settles it the same way.
                fig.subplots_adjust(top=0.74, bottom=0.16, wspace=0.08)
                cs = fig.add_axes([0.125, 0.075, 0.36, 0.026])
                cd = fig.add_axes([0.655, 0.075, 0.225, 0.026])
                cb = fig.colorbar(im_s, cax=cs, orientation="horizontal")
                cb.set_label(f"{unit}  (intact − occluded), clipped at 0",
                             fontsize=6.5)
                cb2 = fig.colorbar(im_d, cax=cd, orientation="horizontal")
                cb2.set_label("difference in reliance", fontsize=6.5)
                for c in (cb, cb2):
                    c.ax.tick_params(labelsize=6)
                px_txt = (f" · pixel ({int(tag.split('_')[2])}, "
                          f"{int(tag.split('_')[3])})" if tag else "")
                fig.suptitle(
                    f"{pair} · chip {i} · {chips[i]['tile']}{px_txt} · "
                    f"target {args.target} · fill {args.fill} · patch "
                    f"{args.patch}/stride {args.stride}"
                    + ("" if matched else " · SEEDS NOT MATCHED"),
                    x=0.012, y=0.985, ha="left", fontsize=7)
                stem = (f"OC_chip{i:04d}_{pair}_{args.target}{tag}_"
                        f"{args.fill}_{args.normalise}")
                for ext in ("pdf", "png"):
                    p = out / f"{stem}.{ext}"
                    fig.savefig(p, bbox_inches=None)
                    wrote.append(p)
                    print(f"    wrote {p}")
                plt.close(fig)

                # Both normalisations in the CSV whichever one was drawn: a
                # claim read off one panel has to be checkable against the
                # other, and re-running the pass to find out is 8000 forwards.
                raw_a, raw_b, _, _ = normalise_pair(m_a, m_b, "none")
                pk_a, pk_b, _, _ = normalise_pair(m_a, m_b, "peak")
                common = dict(chip=i, pair=pair, target=args.target, fill=args.fill,
                              patch=args.patch, stride=args.stride,
                              tile=chips[i]["tile"], pixel=tag.lstrip("_") or None,
                              arm=arm_t["arm"], seed=arm_t["seed"],
                              base_arm=arm_b["arm"], base_seed=arm_b["seed"],
                              seeds_matched=matched,
                              peak_treat=pa if args.normalise == "peak" else peak_of(m_a),
                              peak_base=pb if args.normalise == "peak" else peak_of(m_b),
                              intact_treat=float(r0_a[k]),
                              intact_base=float(r0_b[k]))
                rows += region_stats(m_a, ctx["masks"], quantity="map_treat", **common)
                rows += region_stats(m_b, ctx["masks"], quantity="map_base", **common)
                rows += region_stats(raw_a - raw_b, ctx["masks"],
                                     quantity="diff_raw", **common)
                rows += region_stats(pk_a - pk_b, ctx["masks"],
                                     quantity="diff_peak", **common)

    if not rows:
        raise SystemExit("no chip produced a figure — every requested chip is "
                         "road-free, so there is no margin and no pixel to read")
    df = pd.DataFrame(rows)
    tag = "_".join(sorted(df["pair"].unique()))
    csv = out / f"occl_contrast_region_stats_{tag}.csv"
    df.to_csv(csv, index=False)
    (out / f"meta_{tag}.json").write_text(json.dumps({
        "instrument": "occl_contrast", "fixture_hash": fx_hash,
        "pairs": [{"treat": t.name, "base": b.name, "seeds_matched": m}
                  for t, b, m in pairs],
        "chips": list(args.chips), "target": args.target, "fill": args.fill,
        "n_pixels": args.n_pixels, "patch": args.patch, "stride": args.stride,
        "n_windows": n_win, "normalise": args.normalise, "ndvi_tau": args.tau,
        "clip_percentile": args.clip, "peak_percentile": PEAK_PCT,
        "sr_dir": args.sr_dir, "ckpt_glob": args.ckpt_glob,
        "device": args.device}, indent=1))
    print(f"\nwrote {csv}  ({len(wrote)} figure file(s))")
    print(df[df["quantity"] == f"diff_{'peak' if args.normalise == 'peak' else 'raw'}"]
          .pivot_table(index=["pair", "chip"], columns="region", values="mean")
          .to_string(float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
