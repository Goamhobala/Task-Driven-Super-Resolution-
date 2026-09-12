"""Occlusion suite v3 — the forward-only pass (docs/occlusion_suite_plan.md).

    PYTHONPATH=src python -m sr.probes.occl_context_extract \
        --runs-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns \
        --include '*r2grid*' --ref-run <the r0 run dir> \
        --sr-dir models/SEN2SRLite_RGBN --cache-dir probe_cache \
        --n-chips 96 --exemplar-chips 7 41 88 --device mps

Extends instrument C. NO AUTOGRAD ANYWHERE — that is what puts the whole
suite on 8 GB MPS as well as the cluster, and it is why `freeze_sr`'s
`no_grad` wrap is harmless here and is left alone (instrument E had to clear
it; this pass must not).

One SR forward per chip, then many U-Net forwards: every perturbation happens
in `y`-space (the raw-unit SR output the z-score consumes), so the generator
never runs twice. The perturbed tensor always goes through the arm's OWN
`band_mean`/`band_std` — the normaliser is part of the arm, never part of the
stimulus (extract.py's rule).

WRITES `<cache-dir>/<run>/occl2/`
---------------------------------
    conditions.pq  chip x condition: margin, dmargin vs intact, fill params
    patches.pq     chip x patch x patch-size: dmargin (the sliding pass)
    context.pq     chip x patch x sampled road pixel: dlogit at that pixel
    maps/          dmargin patch grids, registered exemplar chips only
    meta.json      every parameter in the spec's §2, plus timings

THE CONDITIONS (spec §2)
------------------------
A `intact`                     — the reference margin.
B `band_<b>@<region>`          — band b flattened to the arm's own
                                 `band_mean[b]` INSIDE one region only. The
                                 causal "does band b matter via region R"
                                 readout (NIR-via-vegetation). 12 forwards.
C `cf_<src>@<region>`          — all bands replaced by the r0 bicubic (or
                                 frozen-SEN2SR) output inside R. "Reliance on
                                 SR-added content in R." 3 (+3) forwards.
D sliding window               — patch fill at `band_mean`, stride = patch,
                                 256 patches on the 512 px grid. Per patch:
                                 dmargin, and dlogit at K sampled road pixels.
E sliding window, `y_bic` fill — exemplar chips only.

The target is the MARGIN — mean logit over GT road minus mean logit over
background — and road-free chips are skipped, so this pass covers exactly the
chips `occlusion.parquet` scores (AP is undefined on a road-free chip).

AFFINE MATCHING (mandatory for C and E)
---------------------------------------
Before a counterfactual patch is pasted it is rescaled per band to the
destination's local mean and std over the pasted region. Bare-lane (HC-off)
arms run at drifted radiometry, and a raw paste there creates a seam: the
delta would then measure seam disruption, not information removal. Matching
substitutes STRUCTURE while keeping radiometry.

WHAT THE HC-ON NULL ACTUALLY IS (corrected 2026-09-03, from a real arm)
-----------------------------------------------------------------------
The first version of this pass asserted that on an HC-on arm the affine match
is a near no-op, reasoning that the constraint takes the low band from
`bicubic(lr)` and r0's upsampler is the same call. Measured on r2a/seed42:

    global per-band mean(arm) vs mean(r0)   agree to 2e-5 - 7e-5 relative
    global std(arm) / std(r0)               1.37 - 2.00
    ROAD-region std(arm) / std(r0)          10.4 - 21.8

So the null holds for the DC/low band — exactly the 0.00000 per-band mean shift
instrument D measured — and not at all for local variance, which is precisely
what the SR's added high band changes, and most extremely along a thin road
where bicubic is nearly flat. The fatal check is therefore the global per-band
mean agreement (`dc_shift`, `--affine-tol`), which a wrong reference or wrong
units WOULD break; the match's own displacement and its per-region std ratio
are recorded as diagnostics instead of asserted.

Note what a 10-20x std ratio means for the counterfactual: matching lifts
bicubic's near-flat road structure — interpolation artefacts included — to the
arm's local contrast. The ratio is in `conditions.parquet` per row and the pass
warns when its median is extreme, because a `cf_*` claim resting on a 20x
amplification deserves to be read with that number in view.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sr.probes.attr_extract import NDVI_TAU, choose_chips, region_masks
from sr.probes.extract import (BAND_NAMES, CKPT_GLOB, SR_DIR, discover,
                               load_fixture, load_model, read_chip, run_meta)
from sr.probes.make_fixtures import FIXTURE_DIR

REGIONS = ("road", "veg", "other")
PATCH = 32
K_PIXELS = 32
# Robustness protocol (spec §4): every headline number is re-run at these patch
# sizes on a sub-subsample. Stability across them replaces IG's completeness
# axiom as the validity argument, so it is a default, not an opt-in.
ROBUST_PATCHES = (16, 64)
ROBUST_CHIPS = 24


# ------------------------------------------------------------------ fills
def affine_match(src, dst, mask, eps=1e-6):
    """`src` rescaled per band to `dst`'s mean/std over `mask`. Shapes (C,H,W).

    Structure from the source, radiometry from the destination. Returns the
    matched SOURCE (full tensor); the caller pastes it inside `mask`.
    """
    m = mask.unsqueeze(0)                       # (1,H,W) broadcast over bands
    n = m.sum().clamp(min=1)
    mu_s = (src * m).sum(dim=(-2, -1), keepdim=True) / n
    mu_d = (dst * m).sum(dim=(-2, -1), keepdim=True) / n
    sd_s = (((src - mu_s) ** 2 * m).sum(dim=(-2, -1), keepdim=True) / n).sqrt()
    sd_d = (((dst - mu_d) ** 2 * m).sum(dim=(-2, -1), keepdim=True) / n).sqrt()
    return (src - mu_s) / sd_s.clamp(min=eps) * sd_d + mu_d


def paste(dst, src, mask):
    """`dst` with `src`'s values inside `mask` (bool (H,W)); dst is not mutated."""
    out = dst.clone()
    m = mask.unsqueeze(0).expand_as(dst)
    out[m] = src.expand_as(dst)[m]
    return out


def patch_slices(size, patch, stride):
    """Top-left corners of the sliding window, covering the whole grid.

    The last row/column is clamped to the edge rather than dropped, so a patch
    size that does not divide the grid still covers it — otherwise the
    robustness re-runs at 16 and 64 would silently measure different areas.
    """
    xs = list(range(0, max(size - patch, 0) + 1, stride))
    if xs[-1] + patch < size:
        xs.append(size - patch)
    return xs


# ------------------------------------------------------------------- model
class Decoder:
    """One loaded checkpoint, used only as `z-score -> U-Net -> logits`."""

    def __init__(self, model, torch):
        self.m, self.torch = model, torch
        self.scale = float(model.hparams.reflectance_scale)
        self.hc_on = bool(getattr(model, "_sr_hc_on", False))

    def sr(self, x):
        with self.torch.no_grad():
            return self.m._sr_forward(x)

    def logits(self, y):
        """`y` raw units (B,C,H,W) -> (B,H,W) logits under this ckpt's z-score."""
        t = self.torch
        with t.no_grad(), t.autocast(device_type=y.device.type, enabled=False):
            x = (y - self.m.band_mean) / self.m.band_std
            return self.m.model(x).float()[:, 0]


def margin_of(logits, road_t):
    """(B,) — mean logit over GT road minus mean logit over background."""
    return logits[:, road_t].mean(dim=1) - logits[:, ~road_t].mean(dim=1)


def sample_road_pixels(mask, k, fixture_hash, chip_idx):
    """K GT road pixels of one chip, drawn deterministically from the fixture.

    Seeded from the fixture hash and the chip index alone, so every arm and
    every seed reads the SAME pixels of the same chip and the weighted-context
    distributions are paired.
    """
    flat = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(int(fixture_hash[:8], 16) + chip_idx)
    if flat.size <= k:
        return np.sort(flat)
    return np.sort(rng.choice(flat, size=k, replace=False))


# -------------------------------------------------------------------- pass
def _batched_margins(dec, variants, road_t, pix_idx, batch, torch):
    """Forward y-variants; return (margins, logits at `pix_idx`).

    `variants` is consumed as an ITERABLE, `batch` at a time: the sliding pass
    has 256 windows and materialising them all would be ~1 GB of y-copies
    before a single forward runs — precisely the memory this suite exists to
    avoid needing.
    """
    import itertools

    it = iter(variants)
    ms, ls = [], []
    while True:
        chunk = list(itertools.islice(it, batch))
        if not chunk:
            break
        lg = dec.logits(torch.stack(chunk))
        ms.append(margin_of(lg, road_t).cpu().numpy())
        if pix_idx is not None:
            ls.append(lg.reshape(lg.shape[0], -1)[:, pix_idx].cpu().numpy())
        del chunk, lg
    return (np.concatenate(ms) if ms else np.zeros(0),
            np.concatenate(ls) if (pix_idx is not None and ls) else None)


def region_conditions(y, band_mean, masks_t, srcs, torch):
    """[(name, tensor, affine_shift)] for conditions B and C of one chip.

    The band fills also run over an `all` region — the whole image. That is
    instrument C's own intervention, in margin space, and it is what the
    consistency panel is built on: without it the suite would have no total for
    the three region-conditioned numbers to sit against, and the regions do not
    simply add (the network is not linear in the fill).
    """
    out = []
    band_masks = {**masks_t, "all": torch.ones_like(next(iter(masks_t.values())))}
    for b, band in enumerate(BAND_NAMES):
        for rname, m in band_masks.items():
            if not bool(m.any()):
                continue
            v = y.clone()
            v[b][m] = band_mean[b]
            out.append((f"band_{band}@{rname}", v, 0.0, float("nan")))
    # The all-band mean fill: the whole-image analog of the sliding pass's own
    # fill. One forward per region, and it is what makes the sliding pass
    # checkable — the patch deltas should sum toward the `all` cell's delta
    # (spec §6.5, checked in the analysis rather than asserted per chip).
    for rname, m in band_masks.items():
        if not bool(m.any()):
            continue
        v = y.clone()
        for b in range(y.shape[0]):
            v[b][m] = band_mean[b]
        out.append((f"allbands@{rname}", v, 0.0, float("nan")))
    for sname, ysrc in srcs.items():
        for rname, m in masks_t.items():
            if not bool(m.any()):
                continue
            matched = affine_match(ysrc, y, m)
            # Diagnostics, not assertions (see the module docstring): how far
            # the match moved the donor, and the local contrast ratio it
            # applied. Both are legitimately large where the SR added
            # structure the bicubic donor does not have.
            denom = (ysrc * m).pow(2).mean().sqrt().clamp(min=1e-9)
            shift = float(((matched - ysrc) * m).pow(2).mean().sqrt() / denom)
            n = m.sum().clamp(min=1)
            mm = m.unsqueeze(0)
            sd = lambda t: ((((t - (t * mm).sum(dim=(-2, -1), keepdim=True) / n)
                              ** 2) * mm).sum(dim=(-2, -1)) / n).sqrt()
            ratio = float((sd(y) / sd(ysrc).clamp(min=1e-9)).median())
            out.append((f"cf_{sname}@{rname}", paste(y, matched, m), shift, ratio))
    return out


def window_coords(size, patch, stride):
    """[(row, col)] of every window, row-major — the order the maps reshape in."""
    xs = patch_slices(size, patch, stride)
    return [(r, c) for r in xs for c in xs]


def sliding(y, fill, patch, stride, torch):
    """Generator of y-variants, one per window, in `window_coords` order.

    `fill` is either a per-band constant of shape (C,1,1) — which broadcasts
    into the window — or a full (C,H,W) donor, which is sliced. Slicing a
    (C,1,1) constant would yield an empty patch for every window but the first,
    so the two cases are separated rather than handled by one expression.
    """
    broadcast = fill.shape[-1] == 1
    for r, c in window_coords(y.shape[-1], patch, stride):
        v = y.clone()
        v[:, r:r + patch, c:c + patch] = (
            fill if broadcast else fill[..., r:r + patch, c:c + patch])
        yield v


def extract_run(run: Path, args, fixture, refs) -> dict:
    """The whole suite for one arm-seed."""
    import pandas as pd
    import torch
    from sr.viz_models import find_ckpt, resolve_theta

    meta_fx, chips, _px, fx_hash = fixture
    ckpt = find_ckpt(run, args.ckpt_glob)
    out_dir = Path(args.cache_dir) / run.name / "occl2"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch, ckpt_step = raw.get("epoch"), raw.get("global_step")
    del raw

    dec = Decoder(load_model(ckpt, Path(args.sr_dir), args.device), torch)
    for name, r in refs.items():
        if abs(r.scale - dec.scale) > 1e-9:
            raise SystemExit(
                f"{run.name} reflectance_scale={dec.scale} but the {name!r} "
                f"source is {r.scale}. The scale is coupled to the baked band "
                "statistics, so the substituted patch would be in another unit.")

    theta, note = resolve_theta(run, args.select_on, args.default_theta)
    theta_prov = ("sweep" if note.startswith(args.select_on)
                  else "recorded" if note == "recorded θ*" else "fallback")
    crop, up = meta_fx["crop"], meta_fx["upscale"]
    bands = tuple(dec.m.hparams.bands)
    if bands != (1, 2, 3, 4):
        raise SystemExit(f"{run.name}: bands={bands}; {BAND_NAMES} assumes V2 order.")

    idxs = choose_chips(chips, args.n_chips, args.chips)
    bad = [c for c in args.exemplar_chips if c not in idxs]
    if bad:
        raise SystemExit(
            f"--exemplar-chips {bad} are not in the subsample. Exemplars are "
            "registered before any map is looked at (region_classes_plan §2), "
            "so this is a typo, not a reason to widen the subsample.")
    if args.exemplar_chips:
        (out_dir / "maps").mkdir(exist_ok=True)
    robust_chips = set(idxs[:args.robust_chips]) if args.robust_patches else set()

    band_mean = dec.m.band_mean.reshape(-1)
    cond_rows, patch_rows, ctx_rows, skipped = [], [], [], []
    t0 = time.perf_counter()

    for n_done, i in enumerate(idxs):
        chip = chips[i]
        x_np, mask = read_chip(Path(meta_fx["dataset_dir"]), chip, crop, up,
                              bands, meta_fx["mask_dirname"])
        if not mask.any():
            skipped.append(i)          # no margin; occlusion.pq drops it too
            continue
        x = torch.from_numpy(np.ascontiguousarray(x_np))[None].to(args.device)
        road_t = torch.from_numpy(mask).to(args.device)
        masks_np = region_masks(x_np, mask, up, args.tau)
        masks_t = {k: torch.from_numpy(v).to(args.device) for k, v in masks_np.items()}
        pix = sample_road_pixels(mask, args.k_pixels, fx_hash, i)
        pix_t = torch.from_numpy(pix).to(args.device)

        y = dec.sr(x)[0]
        srcs = {k: r.sr(x)[0] for k, r in refs.items()}

        # The real HC-on null: global per-band means. The constraint takes the
        # low band from bicubic(lr) and r0's upsampler is the same call, so an
        # HC-on arm and the bicubic donor share their DC exactly (instrument D
        # measures a 0.00000 per-band mean shift; sr_pad crop leakage is the
        # only residue). A wrong reference or a unit mismatch breaks this;
        # local variance does not, and asserting on that was the earlier bug.
        rms = float(y.pow(2).mean().sqrt())
        dc = {k: float((y.mean(dim=(-2, -1)) - v.mean(dim=(-2, -1))).abs().max()
                       / max(rms, 1e-9)) for k, v in srcs.items()}
        if dec.hc_on and dc.get("bic", 0.0) > args.affine_tol:
            raise SystemExit(
                f"{run.name} chip {i}: the arm and the bicubic donor differ in "
                f"their GLOBAL per-band means by {dc['bic']:.4f} of RMS "
                f"(> --affine-tol {args.affine_tol}) on an HC-ON arm, whose "
                "constraint takes the low band from bicubic(lr). That is a "
                "wiring error — wrong reference run, or a unit mismatch — not a "
                "tolerance to widen.")

        # --- A: intact
        lg0 = dec.logits(y.unsqueeze(0))
        m0 = float(margin_of(lg0, road_t)[0])
        l0 = lg0.reshape(1, -1)[:, pix_t].cpu().numpy()[0]
        cond_rows.append(dict(chip=i, condition="intact", region=None, band=None,
                              fill=None, margin=m0, dmargin=0.0, affine_shift=0.0,
                              std_ratio=float("nan"), dc_shift=dc.get("bic", float("nan")),
                              road_px=int(mask.sum()), stratum=chip.get("stratum")))

        # --- B + C: region-conditioned fills
        conds = region_conditions(y, band_mean, masks_t, srcs, torch)
        if conds:
            ms, _ = _batched_margins(dec, [c[1] for c in conds], road_t, None,
                                     args.batch, torch)
            for (name, _v, shift, ratio), mv in zip(conds, ms):
                head, region = name.split("@")[0], name.split("@")[1]
                cond_rows.append(dict(
                    chip=i, condition=name, region=region,
                    band=head[5:] if head.startswith("band_") else None,
                    fill=("mean" if head.startswith(("band_", "allbands"))
                          else head[3:]),
                    margin=float(mv), dmargin=float(mv) - m0, affine_shift=shift,
                    std_ratio=ratio, dc_shift=dc.get(head[3:], float("nan")),
                    road_px=int(mask.sum()), stratum=chip.get("stratum")))

        # --- D (+E on exemplars): the sliding pass, at every patch size asked
        sizes = [args.patch] + ([p for p in args.robust_patches]
                                if i in robust_chips else [])
        fills = {"mean": band_mean.reshape(-1, 1, 1)}
        if i in args.exemplar_chips and "bic" in srcs:
            fills["bic"] = srcs["bic"]
        for psize in sizes:
            for fname, fill in fills.items():
                if fname != "mean" and psize != args.patch:
                    continue          # the robustness sweep is on the mean fill
                coords = window_coords(y.shape[-1], psize, psize)
                ms, ls = _batched_margins(
                    dec, sliding(y, fill, psize, psize, torch), road_t,
                    pix_t, args.batch, torch)
                for (r, c), mv, lv in zip(coords, ms, ls):
                    patch_rows.append(dict(chip=i, patch=psize, fill=fname,
                                           row=r, col=c, margin=float(mv),
                                           dmargin=float(mv) - m0))
                    if fname == "mean":
                        # EVERY patch size the chip runs, not just the headline:
                        # weighted context IS a headline number, and spec §4
                        # requires it re-computed at 16 and 64. Gating these
                        # rows on the headline size left the robustness table
                        # with a single column and nothing to be stable across.
                        for kx, p in enumerate(pix):
                            ctx_rows.append(dict(chip=i, patch=psize, pix=int(p),
                                                 pix_k=kx, row=r, col=c,
                                                 dlogit=float(l0[kx] - lv[kx])))
                if i in args.exemplar_chips:
                    n = len(patch_slices(y.shape[-1], psize, psize))
                    grid = np.asarray(ms, dtype="float32").reshape(n, n) - m0
                    np.save(out_dir / "maps" / f"{i:04d}_{fname}_p{psize}.npy", grid)

        if (n_done + 1) % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"    {n_done + 1}/{len(idxs)} chips  {el:.0f}s  "
                  f"({el / (n_done + 1):.1f} s/chip)", flush=True)

    elapsed = time.perf_counter() - t0
    cdf = pd.DataFrame(cond_rows)
    ratios = cdf["std_ratio"].dropna() if "std_ratio" in cdf else pd.Series(dtype=float)
    if len(ratios) and float(ratios.median()) > args.ratio_warn:
        print(f"  NOTE: the affine match lifts the donor's local contrast by a "
              f"median {float(ratios.median()):.1f}x (max "
              f"{float(ratios.max()):.1f}x). It is substituting structure the "
              "donor barely has — bicubic is nearly flat along a thin road — so "
              "a cf_* claim should be read with `std_ratio` in view.")
    cdf.to_parquet(out_dir / "conditions.parquet", index=False)
    pd.DataFrame(patch_rows).to_parquet(out_dir / "patches.parquet", index=False)
    pd.DataFrame(ctx_rows).to_parquet(out_dir / "context.parquet", index=False)

    meta = {
        **run_meta(run),
        "instrument": "occl_context",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(ckpt), "epoch": ckpt_epoch, "global_step": ckpt_step,
        "fixture_hash": fx_hash, "fixture_meta": meta_fx,
        "chips": idxs, "n_chips": len(idxs), "chips_skipped_no_road": skipped,
        "chips_evaluated": [i for i in idxs if i not in skipped],
        "exemplar_chips": list(args.exemplar_chips),
        "patch": args.patch, "stride": args.patch,
        "robust_patches": list(args.robust_patches),
        "robust_chips": sorted(robust_chips),
        "k_pixels": args.k_pixels, "ndvi_tau": args.tau, "regions": list(REGIONS),
        "target": "margin", "bands": list(BAND_NAMES),
        "fills": ["mean"] + sorted(refs),
        "sources": {k: str(v) for k, v in args._source_ckpts.items()},
        "affine_tol": args.affine_tol, "hc_on": dec.hc_on,
        "dc_shift_max": (float(cdf["dc_shift"].max())
                         if "dc_shift" in cdf and cdf["dc_shift"].notna().any()
                         else None),
        "std_ratio_median": float(ratios.median()) if len(ratios) else None,
        "reflectance_scale": dec.scale,
        "band_mean": [float(v) for v in dec.m.band_mean.reshape(-1)],
        "band_std": [float(v) for v in dec.m.band_std.reshape(-1)],
        # Recorded, unused: this instrument is theta-free (GT masks, margins,
        # never a thresholded prediction), so plan §9's provenance filter does
        # not apply to it.
        "theta": theta, "theta_provenance": theta_prov, "theta_note": note,
        "batch": args.batch, "device": args.device,
        "hparams": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v)) for k, v in dict(dec.m.hparams).items()},
        "seconds": round(elapsed, 1),
    }
    lost = [c for c in args.exemplar_chips if c in skipped]
    if lost:
        print(f"  WARNING: registered exemplar chip(s) {lost} have no GT road, "
              "so they were skipped and NO map was written for them. Re-register "
              "road-bearing chips (--propose-exemplars only offers those).")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {out_dir}  ({elapsed:.0f}s, {len(cond_rows)} condition rows, "
          f"{len(patch_rows)} patch rows, {len(ctx_rows)} context rows, "
          f"{len(skipped)} road-free chips skipped)", flush=True)
    del dec
    return meta


def propose_exemplars(chips, subset=None, seed=20260903):
    """One chip per stratum, deterministically — the pre-registration helper.

    Printed for the user to paste into `region_classes_plan.md` §2 BEFORE any
    map is looked at. Deliberately not applied automatically: the point of the
    rule is that a human commits to the chips in the doc, not that a script
    picks them at run time.

    Drawn from `subset` — the chips this pass will actually attribute — not
    from the whole fixture. A proposal outside the subsample would be refused
    by the extractor's own exemplar guard, which is a confusing way to learn
    that the helper and the pass disagreed about the population.

    ROAD-BEARING ONLY. 116 of the fixture's 400 chips have no road, by design,
    and this pass skips them (no margin). The first smoke run registered chip
    29 as an exemplar, and it had `road_px=0`: the pass completed, wrote no
    map, and said nothing about why. A chip that cannot produce the panel it
    was registered for is not a candidate.
    """
    pool = range(len(chips)) if subset is None else subset
    by = {}
    for i in pool:
        if not chips[i].get("road_px"):
            continue
        by.setdefault(str(chips[i].get("stratum", "?")), []).append(int(i))
    rng = np.random.default_rng(seed)
    return {s: int(rng.choice(by[s])) for s in sorted(by)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", action="append", default=None)
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--include", default="*gap_ce_anorm_recalpost*")
    ap.add_argument("--ckpt-glob", default=CKPT_GLOB)
    ap.add_argument("--sr-dir", default=SR_DIR)
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--dataset-dir", default=None,
                    help="override the fixture's baked-in dataset_dir")
    ap.add_argument("--cache-dir", default="probe_cache",
                    help="extract.py's cache; this pass writes <run>/occl2/ in it")
    ap.add_argument("--ref-run", default=None,
                    help="the r0 run dir supplying the bicubic counterfactual")
    ap.add_argument("--frozen-src", default=None,
                    help="an r1 run dir supplying the frozen-SEN2SR counterfactual")
    ap.add_argument("--patch", type=int, default=PATCH,
                    help="sliding window size; stride equals it")
    ap.add_argument("--robust-patches", type=int, nargs="*", default=list(ROBUST_PATCHES),
                    help="spec §4: patch sizes every headline number is re-run at")
    ap.add_argument("--robust-chips", type=int, default=ROBUST_CHIPS)
    ap.add_argument("--k-pixels", type=int, default=K_PIXELS,
                    help="GT road pixels sampled per chip for weighted context")
    ap.add_argument("--tau", type=float, default=NDVI_TAU)
    ap.add_argument("--n-chips", type=int, default=96)
    ap.add_argument("--chips", type=int, nargs="*", default=None)
    ap.add_argument("--exemplar-chips", type=int, nargs="*", default=[],
                    help="registered in region_classes_plan §2 BEFORE any map "
                         "is read; --propose-exemplars suggests one per stratum")
    ap.add_argument("--propose-exemplars", action="store_true",
                    help="print a deterministic chip per stratum and exit")
    ap.add_argument("--affine-tol", type=float, default=0.05,
                    help="fatal GLOBAL per-band mean gap between an HC-ON arm "
                         "and the bicubic donor (the low-band null)")
    ap.add_argument("--ratio-warn", type=float, default=5.0,
                    help="median local-contrast ratio above which the affine "
                         "match is flagged in the log")
    ap.add_argument("--batch", type=int, default=8,
                    help="y-variants per U-Net forward (8 fits 8 GB MPS)")
    ap.add_argument("--select-on", default="iou")
    ap.add_argument("--default-theta", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    fixture = load_fixture(Path(args.fixture_dir))
    meta_fx, chips = fixture[0], fixture[1]
    if args.dataset_dir:
        meta_fx["dataset_dir"] = str(args.dataset_dir)
    if args.propose_exemplars:
        print(f"deterministic exemplar proposal from the {args.n_chips}-chip "
              "subsample (one per stratum) — register these in "
              "docs/region_classes_plan.md §2 before reading a map:")
        for s, i in propose_exemplars(
                chips, choose_chips(chips, args.n_chips, args.chips)).items():
            print(f"  {s:<12} chip {i}")
        return 0
    if len(args.exemplar_chips) > 8:
        raise SystemExit("at most 8 exemplar chips.")

    runs = [Path(r) for r in (args.run or [])]
    runs += discover(args.runs_dir or [], args.include, args.ckpt_glob)
    if not runs:
        raise SystemExit("no runs discovered — check --runs-dir / --include")

    from sr.viz_models import find_ckpt

    args._source_ckpts = {}
    for key, rd in (("bic", args.ref_run), ("frz", args.frozen_src)):
        if rd:
            ck = find_ckpt(Path(rd), args.ckpt_glob)
            if ck is None:
                raise SystemExit(f"{rd} holds no {args.ckpt_glob}")
            args._source_ckpts[key] = ck
    if not args._source_ckpts:
        print("no --ref-run: the counterfactual conditions (C, E) are SKIPPED; "
              "only the mean fills run.")

    idxs = choose_chips(chips, args.n_chips, args.chips)
    n_reg = len(BAND_NAMES) * len(REGIONS) + len(args._source_ckpts) * len(REGIONS)
    n_slide = len(patch_slices(meta_fx["crop"] * meta_fx["upscale"],
                               args.patch, args.patch)) ** 2
    print(f"{len(runs)} run(s), {len(idxs)} chips, patch {args.patch}: "
          f"{1 + n_reg + n_slide} forwards/chip "
          f"(+{len(args.robust_patches)} sweeps on {args.robust_chips} chips)")
    for r in runs:
        m = run_meta(r)
        print(f"  {m['arm']:<5} seed {str(m['seed']):<5} {r.name}")
    probe = Path(meta_fx["dataset_dir"]) / chips[0]["image_path"]
    if not probe.exists():
        raise SystemExit(
            f"fixture chip 0 is not readable:\n  {probe}\nPass --dataset-dir "
            "pointing at the directory holding test/imagery/ on this machine.")
    print(f"fixture {fixture[3]}: {meta_fx['n_chips']} chips, τ={args.tau}", flush=True)
    if args.dry_run:
        return 0

    if args.device is None:
        import torch
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {args.device}", flush=True)

    import torch
    refs = {k: Decoder(load_model(ck, Path(args.sr_dir), args.device), torch)
            for k, ck in args._source_ckpts.items()}

    for r in runs:
        print(f"\n--- {r.name}", flush=True)
        extract_run(r, args, fixture, refs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
