"""Instrument E — integrated-gradient attribution (docs/attribution_plan.md v2).

    PYTHONPATH=src python -m sr.probes.attr_extract \
        --runs-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns \
        --include '*r2a*' --ref-run <the r0 run dir> \
        --sr-dir models/SEN2SRLite_RGBN --out-dir attr_cache \
        --n-chips 96 --exemplar-chips 7 41 --device cuda

The ONE gradient pass in the suite: `extract.py` and `bandswap_extract.py`
stay no-grad, this one does not. Two spaces, and the order matters:

  E1 (`--space y`)  IG on the U-NET ONLY, attributing the road margin wrt `y`,
                    the raw-unit tensor the z-score consumes. Baselines: the
                    flat `band_mean` image (occlusion's own fill, exactly),
                    r0's bicubic output, and optionally a frozen-SEN2SR output.
  E2 (`--space x`)  IG through the FULL stack (SR -> adapter -> U-Net) to the
                    10 m input, from a flat baseline at the chip's own per-band
                    input mean.

E1 leads because it is the leg that anchors to occlusion. Plan v1 had that
backwards on the strength of a matched-baseline claim the code does not
support: `extract.py` substitutes `band_mean[b]` into `y`, the SR OUTPUT, and
`band_mean` is the post-recalibration mean of SR outputs in raw units. A flat
10 m input does not come out of `_sr_forward` at `band_mean` — with the hard
constraint on it comes out flat at the INPUT's level, because the constraint
takes the low band from `bicubic(lr)`. The matched baseline lives in y-space.

WHAT THE NUMBER IS
------------------
The attributed scalar is the chip's road MARGIN,

    f = mean logit over GT road px - mean logit over background px

not a road-only logit sum. The main-text claim is agreement with occlusion,
whose carrier is dAP — a RANKING statistic over the whole chip, including the
background an arm suppresses. A road-only sum cannot see false-positive
suppression, so the two would be different functionals pre-registered as
agreeing. `--target road_sum` keeps the road-only version as a sensitivity
check; whichever ran is stamped in meta.json.

Chips with no road have no margin and are SKIPPED — occlusion's AP-undefined
rule, so the two instruments aggregate over the same chips.

THE COMPLETENESS GUARD
----------------------
Per chip, `attr.sum()` must equal `f(x) - f(baseline)`. The residual is
reported RELATIVE to |df| (an absolute tolerance fires on every high-logit
chip and never on a flat one), logged always, and fatal above `--tol`. A
violated axiom is a bug — wrong wiring through the adapter, autocast leakage,
a generator still inside `torch.no_grad` — not a footnote.

`freeze_sr=True` is exactly that bug: `_sr_forward` wraps a frozen generator in
`torch.no_grad()`, which SEVERS the path from the 10 m input to the logits —
`autograd.grad` then raises rather than returning zeros (tested). The flag
governs only that wrap and we want INPUT gradients, not parameter gradients, so
the pass clears it on the loaded instance and says so (`freeze_sr_overridden`
in meta.json). The forward is numerically identical either way.

REGION CLASSES ARE ARM-INDEPENDENT
----------------------------------
road (the fixture's GT mask, never a prediction — this is what keeps the
instrument theta-free), veg (NDVI > tau from the RAW 10 m input, minus road),
other. Plan v1 derived NDVI from the SR output, which would have made the
regions a function of the arm being attributed and the per-region masses
incomparable across arms. "built-up" is dropped: the fixture's `stratum` is a
per-CHIP tile attribute, not a pixel mask, so it is carried per row for
grouping instead.

fp32 throughout, autocast explicitly disabled, models in `.eval()` (which also
disables the adaptive-norm EMA) — same contract as `extract.py`.
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

from sr.probes.extract import (BAND_NAMES, CKPT_GLOB, SR_DIR, discover,
                               load_fixture, load_model, read_chip, run_meta)
from sr.probes.make_fixtures import FIXTURE_DIR

REGIONS = ("road", "veg", "other")
# Subsample seed. Fixed and stamped into meta.json: which chips were attributed
# is part of the measurement, and the consistency scatter has to filter
# occlusion.parquet down to the same set.
CHIP_SEED = 20260902
NDVI_TAU = 0.3


# --------------------------------------------------------------------- IG
def integrated_gradients(f, x, baseline, steps=32, chunk=8):
    """IG of `f` at `x` against `baseline`, both (C, H, W). Returns (C, H, W).

    `f` maps a (B, C, H, W) batch of path points to a (B,) vector of scalars —
    the rows are independent, so summing before `autograd.grad` gives each row
    its own gradient.

    Three deliberate choices, all of which v1's snippet got wrong:

    * the MIDPOINT rule, `(i + 0.5) / steps`, not `linspace(0, 1, steps)`. The
      endpoint rule is the worst available quadrature and its error lands
      directly in the completeness residual, which is the number we make fatal.
    * `.detach().requires_grad_(True)`: the path is a non-leaf the moment
      either endpoint carries grad, and `requires_grad_` on a non-leaf raises.
    * chunking, because a 32-deep path through a 512 px U-Net with the graph
      retained does not fit a 24 GB card. The result is chunk-invariant.
    """
    import torch

    d = x - baseline
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    total = torch.zeros_like(x)
    for i in range(0, steps, chunk):
        a = alphas[i:i + chunk].view(-1, *([1] * x.dim()))
        path = (baseline.unsqueeze(0) + a * d.unsqueeze(0)).detach().requires_grad_(True)
        (g,) = torch.autograd.grad(f(path).sum(), path)
        total = total + g.sum(0)
    return d * (total / steps)


def completeness(attr, f_x, f_base, eps=1e-8):
    """(delta, residual, relative residual) for the axiom's per-chip check."""
    df = float(f_x) - float(f_base)
    r = float(attr.sum()) - df
    return df, r, abs(r) / (abs(df) + eps)


# ---------------------------------------------------------------- regions
def region_masks(x_np, road_hr, upscale, tau=NDVI_TAU):
    """{name: (H, W) bool} on the HR grid — a partition, arm-independent.

    NDVI comes from the RAW 10 m input (bands NIR=3, R=0 of the V2 order) and
    is nearest-upsampled, so no arm's generator can move a region boundary.
    Road wins the overlap: it is the class every claim is conditioned on.
    """
    nir, red = x_np[3].astype("float64"), x_np[0].astype("float64")
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    veg = np.repeat(np.repeat(ndvi > tau, upscale, 0), upscale, 1)
    if veg.shape != road_hr.shape:
        raise ValueError(f"NDVI upsampled to {veg.shape}, mask is {road_hr.shape}")
    road = np.asarray(road_hr, dtype=bool)
    veg = veg & ~road
    return {"road": road, "veg": veg, "other": ~(road | veg)}


def regions_to_lr(masks, upscale):
    """The same partition on the LR grid: per-cell class fraction, then argmax.

    E2 attributes at 10 m while the classes live at 2.5 m, and at 10 m nearly
    every road pixel is mixed. The fractions are returned alongside so that
    mixing is visible in the cache rather than hidden inside a majority vote.
    """
    names = list(masks)
    h, w = masks[names[0]].shape
    fr = {k: m.reshape(h // upscale, upscale, w // upscale, upscale)
           .mean(axis=(1, 3)) for k, m in masks.items()}
    stack = np.stack([fr[k] for k in names])
    win = stack.argmax(axis=0)
    return {k: win == i for i, k in enumerate(names)}, fr


# ------------------------------------------------------------------ tables
def mass_rows(attr_np, masks, band_names, **extra):
    """One row per band x region: signed and absolute mass, and the fractions.

    BOTH are stored because completeness decomposes only over the SIGNED sum —
    a table of |attr| fractions is not certified by the residual, and the
    figure has to say which it draws.
    """
    tot_s = float(attr_np.sum())
    tot_a = float(np.abs(attr_np).sum())
    rows = []
    for b, band in enumerate(band_names):
        a = attr_np[b]
        for region in REGIONS:
            m = masks[region]
            s, absm = float(a[m].sum()), float(np.abs(a[m]).sum())
            rows.append(dict(band=band, region=region, n_px=int(m.sum()),
                             mass=s, absmass=absm,
                             mass_frac=s / tot_s if tot_s else np.nan,
                             absmass_frac=absm / tot_a if tot_a else np.nan,
                             **extra))
    return rows


# ------------------------------------------------------------------ the arm
class Arm:
    """One loaded checkpoint, exposing the two attributable functions."""

    def __init__(self, model, torch):
        self.m, self.torch = model, torch
        self.scale = float(model.hparams.reflectance_scale)
        self.freeze_overridden = bool(getattr(model.hparams, "freeze_sr", False))
        if self.freeze_overridden:
            # Only governs the `torch.no_grad()` wrap in `_sr_forward`; the
            # forward is numerically identical with it cleared, and without
            # clearing it the graph from x to the logits is severed and
            # `autograd.grad` raises.
            model.hparams.freeze_sr = False
        for p in model.parameters():
            p.requires_grad_(False)      # input gradients only

    def sr(self, x, grad=False):
        """raw 10 m input -> the pre-adapter tensor, in raw units."""
        t = self.torch
        ctx = t.enable_grad() if grad else t.no_grad()
        with ctx:
            return self.m._sr_forward(x)

    def unet_logits(self, y):
        """`y` in RAW units -> (B, 1, H, W) logits under this ckpt's z-score.

        The normaliser is part of the arm's front-end, not part of the stimulus
        (extract.py's common-input rule), so every injected baseline goes
        through THIS checkpoint's band statistics.
        """
        t = self.torch
        with t.autocast(device_type=y.device.type, enabled=False):
            return self.m.model((y - self.m.band_mean) / self.m.band_std).float()

    def full_logits(self, x):
        """raw 10 m input -> logits, the whole deployed stack."""
        t = self.torch
        with t.autocast(device_type=x.device.type, enabled=False):
            return self.m(x).float()


def make_target(road_t, kind, torch):
    """logits (B,1,H,W) -> (B,) — the attributed scalar.

    `margin` is monotone in what AP ranks, which is what occlusion measures;
    `road_sum` is plan v1's road-only functional, kept as a sensitivity check.
    """
    bg_t = ~road_t

    def margin(logits):
        l = logits[:, 0]
        return l[:, road_t].mean(dim=1) - l[:, bg_t].mean(dim=1)

    def road_sum(logits):
        return logits[:, 0][:, road_t].sum(dim=1)

    return {"margin": margin, "road_sum": road_sum}[kind]


# ------------------------------------------------------------------ chip set
def choose_chips(chips, n_chips, explicit, seed=CHIP_SEED):
    """The attributed subsample: stratified, deterministic, and recorded.

    Stratified on the fixture's own `stratum` so the subsample keeps the
    fixture's urban/rural mix rather than whatever the first N chips happen to
    be. `--chips` overrides with an explicit list (handpicked tiles).
    """
    if explicit:
        return sorted(set(int(i) for i in explicit))
    if n_chips is None or n_chips >= len(chips):
        return list(range(len(chips)))
    strata = {}
    for i, c in enumerate(chips):
        strata.setdefault(str(c.get("stratum", "?")), []).append(i)
    order = sorted(strata)
    # Largest-remainder apportionment: the quotas sum to exactly `n_chips`, so
    # nothing has to be truncated afterwards. Truncating a concatenated list
    # would have dropped whole strata off the end of the alphabet.
    exact = {s: n_chips * len(strata[s]) / len(chips) for s in order}
    quota = {s: min(int(exact[s]), len(strata[s])) for s in order}
    short = n_chips - sum(quota.values())
    for s in sorted(order, key=lambda s: (-(exact[s] - int(exact[s])), s)):
        if short <= 0:
            break
        if quota[s] < len(strata[s]):
            quota[s] += 1
            short -= 1
    rng = np.random.default_rng(seed)
    out = []
    for s in order:
        pool = np.array(strata[s])
        out += list(rng.choice(pool, size=quota[s], replace=False))
    return sorted(int(i) for i in out)


# -------------------------------------------------------------------- pass
def extract_run(run: Path, args, fixture, refs) -> dict:
    """Attribution pass for one arm-seed. `refs` holds the baseline suppliers."""
    import pandas as pd
    import torch
    from sr.viz_models import find_ckpt, resolve_theta

    meta_fx, chips, _px, fx_hash = fixture
    ckpt = find_ckpt(run, args.ckpt_glob)
    out_dir = Path(args.out_dir) / run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch, ckpt_step = raw.get("epoch"), raw.get("global_step")
    del raw

    arm = Arm(load_model(ckpt, Path(args.sr_dir), args.device), torch)
    for name, r in refs.items():
        if abs(r.scale - arm.scale) > 1e-9:
            raise SystemExit(
                f"{run.name} reflectance_scale={arm.scale} but the {name!r} "
                f"baseline supplier is {r.scale}. The scale is coupled to the "
                "baked band statistics, so the two tensors are not in one unit "
                "and the attribution would be against a baseline in another "
                "space. Re-express the reference first (viz_grid's rule).")
    if arm.freeze_overridden:
        print("  freeze_sr was True -> cleared for this pass; without that "
              "`_sr_forward`'s no_grad severs the graph and autograd raises.")

    theta, note = resolve_theta(run, args.select_on, args.default_theta)
    theta_prov = ("sweep" if note.startswith(args.select_on)
                  else "recorded" if note == "recorded θ*" else "fallback")
    crop, up = meta_fx["crop"], meta_fx["upscale"]
    bands = tuple(arm.m.hparams.bands)
    if bands != (1, 2, 3, 4):
        raise SystemExit(f"{run.name}: bands={bands}; {BAND_NAMES} assumes V2 order.")

    idxs = choose_chips(chips, args.n_chips, args.chips)
    bad = [c for c in args.exemplar_chips if c not in idxs]
    if bad:
        raise SystemExit(
            f"--exemplar-chips {bad} are not in the attributed subsample. "
            "Exemplars are pre-registered BEFORE any map is looked at, so this "
            "is a typo, not a reason to widen the subsample silently.")
    if args.exemplar_chips:
        (out_dir / "attr_maps").mkdir(exist_ok=True)

    rows, chip_rows, skipped = [], [], []
    t0 = time.perf_counter()

    for n_done, i in enumerate(idxs):
        chip = chips[i]
        x_np, mask = read_chip(Path(meta_fx["dataset_dir"]), chip, crop, up,
                               bands, meta_fx["mask_dirname"])
        if not mask.any():
            # No road -> no margin. Occlusion drops the same chips (AP is
            # undefined there), so both instruments aggregate over one set.
            skipped.append(i)
            continue
        x = torch.from_numpy(np.ascontiguousarray(x_np))[None].to(args.device)
        road_t = torch.from_numpy(mask).to(args.device)
        target = make_target(road_t, args.target, torch)
        masks_hr = region_masks(x_np, mask, up, args.tau)
        masks_lr, frac_lr = regions_to_lr(masks_hr, up)

        y = arm.sr(x)[0]                                   # (C, H, W) raw units
        jobs = []
        if args.space in ("y", "both"):
            f_y = lambda p: target(arm.unet_logits(p))
            for b in args.baselines:
                if b == "mean":
                    base = arm.m.band_mean.reshape(-1, 1, 1).expand_as(y).contiguous()
                elif b in refs:
                    base = refs[b].sr(x)[0]
                else:
                    continue
                jobs.append(("y", b, f_y, y.detach(), base.detach(), masks_hr))
        if args.space in ("x", "both"):
            f_x = lambda p: target(arm.full_logits(p))
            x0 = x[0]
            base = x0.mean(dim=(-2, -1), keepdim=True).expand_as(x0).contiguous()
            jobs.append(("x", "chipmean", f_x, x0.detach(), base.detach(), masks_lr))

        for space, bname, f, xt, base, masks in jobs:
            with torch.no_grad():
                f_x_v = float(f(xt.unsqueeze(0))[0])
                f_b_v = float(f(base.unsqueeze(0))[0])
            attr = integrated_gradients(f, xt, base, args.steps, args.chunk)
            df, resid, rel = completeness(attr, f_x_v, f_b_v)
            if rel > args.tol:
                raise SystemExit(
                    f"{run.name} chip {i} {space}/{bname}: completeness violated "
                    f"— attr.sum()={float(attr.sum()):.4g} vs df={df:.4g} "
                    f"(relative residual {rel:.3f} > --tol {args.tol}). That is "
                    "a wiring bug (a no_grad in the path, autocast leakage) or "
                    "too few steps; do not aggregate past it.")
            a_np = attr.detach().float().cpu().numpy()
            rows += mass_rows(a_np, masks, list(BAND_NAMES), chip=i, space=space,
                              baseline=bname)
            chip_rows.append(dict(chip=i, space=space, baseline=bname,
                                  f_x=f_x_v, f_base=f_b_v, df=df,
                                  attr_sum=float(a_np.sum()),
                                  absmass=float(np.abs(a_np).sum()),
                                  resid=resid, resid_rel=rel,
                                  road_px=int(mask.sum()),
                                  veg_px=int(masks_hr["veg"].sum()),
                                  road_frac_lr=float(frac_lr["road"].mean()),
                                  stratum=chip.get("stratum")))
            if i in args.exemplar_chips:
                np.save(out_dir / "attr_maps" / f"{space}_{bname}_{i:04d}.npy",
                        a_np.astype("float32"))

        if (n_done + 1) % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"    {n_done + 1}/{len(idxs)} chips  {el:.0f}s  "
                  f"({el / (n_done + 1):.2f} s/chip)", flush=True)

    elapsed = time.perf_counter() - t0
    df_mass = pd.DataFrame(rows)
    df_chip = pd.DataFrame(chip_rows)
    df_mass.to_parquet(out_dir / "attr_mass.parquet", index=False)
    df_chip.to_parquet(out_dir / "attr_chip.parquet", index=False)

    meta = {
        **run_meta(run),
        "instrument": "attribution",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(ckpt), "epoch": ckpt_epoch, "global_step": ckpt_step,
        "fixture_hash": fx_hash, "fixture_meta": meta_fx,
        "chips": idxs, "n_chips": len(idxs),
        "chips_skipped_no_road": skipped,
        "chip_seed": CHIP_SEED if not args.chips else None,
        "exemplar_chips": list(args.exemplar_chips),
        "space": args.space, "baselines": list(args.baselines),
        "target": args.target, "steps": args.steps, "chunk": args.chunk,
        "tol": args.tol, "ndvi_tau": args.tau, "regions": list(REGIONS),
        "resid_rel_max": float(df_chip["resid_rel"].max()) if len(df_chip) else None,
        "resid_rel_mean": float(df_chip["resid_rel"].mean()) if len(df_chip) else None,
        # theta is recorded but UNUSED: attribution is threshold-free, and the
        # region classes come from GT rather than from predictions, so plan §9's
        # theta-provenance filter does not apply to this instrument.
        "theta": theta, "theta_provenance": theta_prov, "theta_note": note,
        "reflectance_scale": arm.scale,
        "freeze_sr_overridden": arm.freeze_overridden,
        "band_mean": [float(v) for v in arm.m.band_mean.reshape(-1)],
        "band_std": [float(v) for v in arm.m.band_std.reshape(-1)],
        "band_names": list(BAND_NAMES),
        "baseline_runs": {k: str(v) for k, v in
                          getattr(args, "_baseline_runs", {}).items()},
        "device": args.device,
        "hparams": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v)) for k, v in dict(arm.m.hparams).items()},
        "seconds": round(elapsed, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    if len(df_chip):
        print(f"  wrote {out_dir}  ({elapsed:.0f}s, {len(rows)} mass rows, "
              f"{len(skipped)} road-free chips skipped, max relative "
              f"completeness residual {meta['resid_rel_max']:.2e})", flush=True)
    else:
        print(f"  wrote {out_dir} — no chip in the subsample had road", flush=True)
    del arm
    return meta


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
    ap.add_argument("--out-dir", default="attr_cache")
    ap.add_argument("--ref-run", default=None,
                    help="r0 run dir supplying the 'r0' (bicubic) baseline")
    ap.add_argument("--frozen-run", default=None,
                    help="an r1 run dir supplying the 'frozen' SEN2SR baseline")
    ap.add_argument("--space", default="both", choices=("y", "x", "both"),
                    help="y = E1 (U-Net only), x = E2 (full stack)")
    ap.add_argument("--baselines", nargs="*", default=["mean"],
                    choices=("mean", "r0", "frozen"),
                    help="y-space baselines; 'mean' is occlusion's own fill")
    ap.add_argument("--target", default="margin", choices=("margin", "road_sum"))
    ap.add_argument("--steps", type=int, default=32,
                    help="IG path points (midpoint rule)")
    ap.add_argument("--chunk", type=int, default=4,
                    help="path points per forward+backward (memory vs speed)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="fatal RELATIVE completeness residual")
    ap.add_argument("--tau", type=float, default=NDVI_TAU,
                    help="NDVI threshold defining the vegetated region")
    ap.add_argument("--n-chips", type=int, default=96,
                    help="stratified deterministic subsample of the fixture")
    ap.add_argument("--chips", type=int, nargs="*", default=None,
                    help="explicit chip indices, overriding --n-chips")
    ap.add_argument("--exemplar-chips", type=int, nargs="*", default=[],
                    help="chips whose FULL maps are written; pre-register them "
                         "here before looking at any map (<=8)")
    ap.add_argument("--select-on", default="iou",
                    help="sweep.json criterion θ* is re-argmaxed on "
                         "(recorded only; this instrument is θ-free)")
    ap.add_argument("--default-theta", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if len(args.exemplar_chips) > 8:
        raise SystemExit("at most 8 exemplar chips — the maps are ~4 MB each and "
                         "a large gallery is an invitation to curate post hoc.")
    if "r0" in args.baselines and not args.ref_run:
        raise SystemExit("--baselines r0 needs --ref-run <the r0 run dir>")
    if "frozen" in args.baselines and not args.frozen_run:
        raise SystemExit("--baselines frozen needs --frozen-run <an r1 run dir>")

    runs = [Path(r) for r in (args.run or [])]
    runs += discover(args.runs_dir or [], args.include, args.ckpt_glob)
    if not runs:
        raise SystemExit("no runs discovered — check --runs-dir / --include")

    from sr.viz_models import find_ckpt

    args._baseline_runs = {}
    for key, rd in (("r0", args.ref_run), ("frozen", args.frozen_run)):
        if key in args.baselines:
            ck = find_ckpt(Path(rd), args.ckpt_glob)
            if ck is None:
                raise SystemExit(f"{rd} holds no {args.ckpt_glob}")
            args._baseline_runs[key] = ck

    print(f"{len(runs)} run(s), space={args.space}, baselines="
          f"{','.join(args.baselines)}, target={args.target}, "
          f"steps={args.steps}:")
    for r in runs:
        m = run_meta(r)
        print(f"  {m['arm']:<5} seed {str(m['seed']):<5} {r.name}")

    fixture = load_fixture(Path(args.fixture_dir))
    meta_fx, chips = fixture[0], fixture[1]
    if args.dataset_dir:
        meta_fx["dataset_dir"] = str(args.dataset_dir)
    probe = Path(meta_fx["dataset_dir"]) / chips[0]["image_path"]
    if not probe.exists():
        raise SystemExit(
            f"fixture chip 0 is not readable:\n  {probe}\nThe fixture was built "
            f"on another machine (dataset_dir={meta_fx['dataset_dir']!r}). Pass "
            "--dataset-dir pointing at the directory holding test/imagery/ here.")
    idxs = choose_chips(chips, args.n_chips, args.chips)
    print(f"fixture {fixture[3]}: {meta_fx['n_chips']} chips, "
          f"attributing {len(idxs)}: {idxs[:12]}{' ...' if len(idxs) > 12 else ''}")
    n_b = len(args.baselines) if args.space in ("y", "both") else 0
    n_b += 1 if args.space in ("x", "both") else 0
    print(f"{n_b} attribution(s)/chip x {args.steps} steps "
          f"≈ {n_b * args.steps * 3} forward-equivalents/chip", flush=True)

    if args.dry_run:
        return 0

    if args.device is None:
        import torch
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.device == "mps" and args.space in ("x", "both"):
        print("  NOTE: on MPS the FFT backward in SEN2SR's hard constraint falls "
              "back to CPU (PYTORCH_ENABLE_MPS_FALLBACK=1) — E2 will be slow, "
              "not wrong. Prefer cuda for the x space.")
    print(f"device: {args.device}", flush=True)

    import torch
    refs = {}
    for key, ck in args._baseline_runs.items():
        refs[key] = Arm(load_model(ck, Path(args.sr_dir), args.device), torch)

    for r in runs:
        print(f"\n--- {r.name}", flush=True)
        extract_run(r, args, fixture, refs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
