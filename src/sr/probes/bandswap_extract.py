"""Instrument D — the band-swap counterfactual and the frequency-ablation curve.

    PYTHONPATH=src python -m sr.probes.bandswap_extract \
        --runs-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns \
        --include '*r2grid*' --ref-run <the r0 run dir> \
        --sr-dir models/SEN2SRLite_RGBN --out-dir bandswap_cache --device cuda

Splices each arm's SR output against r0's bicubic in the Fourier plane and
feeds the result to a frozen U-Net, so a performance difference can be
attributed to a band rather than to "sharpness" as a conjecture.

WHY THE TWO LANES ASK DIFFERENT QUESTIONS
-----------------------------------------
`HardConstraint.forward` takes its low band entirely from `bicubic(lr)`, and
r0's `BicubicUpsampler` is the same `interpolate(mode="bicubic",
antialias=True)` call. So for an HC-**on** arm the splice is already in the
deployed forward: `y_on == lo(y_r0) + hi(y_on)` identically. Measured over the
sigma=35 mask on rural chips, 99.94% of an on-cell's departure from r0 sits
above the cut and its per-band mean shift is 0.00000 (the 0.06% residue is
`sr_pad=8` crop leakage — the splice runs on a 576 px grid and is then
cropped). Re-splicing an on-arm is therefore a NULL: it should reproduce the
deployed output, and the `swap_hi` condition doubles as this instrument's
self-test.

For an HC-**off** arm the splice does real work. The off cells' drift from r0
is 92-96% LOW band at the top of the lr_sr ladder, with per-band means walking
by ~0.13 reflectance, so `swap_hi` there is a genuine counterfactual: "what if
the constraint had been applied at eval time". If the off lane's advantage
survives the swap it lived in the high band; if it dies with the low band it
was radiometry.

THE FOUR THINGS THIS PASS MUST GET RIGHT
----------------------------------------
**The splice is the deployed operator, not a re-derivation.** The mask is the
shipped `hard_constraint.safetensor` read byte-for-byte, in the same fftshifted
convention `HardConstraint` uses (DC at centre). The ideal-radius sweep is the
same algebra with `low_pass_mask` swapped for a disk, so the sigma=35 point on
the curve and the HC's own cut are the same operation.

**Whose U-Net.** Conditions are named `<input>@<decoder>`. The deployed pairing
is the arm's input on the arm's decoder; the transfer cells hold one factor
fixed while the other moves. Every input passes through the CONSUMING
checkpoint's own z-score — the normaliser is part of an arm's front-end
(extract.py's common-input convention), and freezing it instead would compare a
network against an input distribution it never saw.

**Mean-matching is a separate condition, never a correction.** Splicing r0's
low band under an off-cell whose `band_mean` was recalibrated to its own
drifted output produces a real z-score offset. That offset IS part of the
treatment, so `swap_hi` keeps it; `swap_hi_mm` adds back the per-band mean the
splice removed, and the gap between them separates DC level from low-band
structure. Reporting only the mean-matched version would hide the radiometry.

**AP, not IoU, carries the claim.** ΔAP is threshold-free, so it is immune to
the theta-provenance filter that plan §9 imposes on every other readout. The
confusion counts at theta* are written too, but a theta-dependent reading must
drop any run whose `theta_provenance` is `fallback`.

fp32 throughout, autocast never enabled, models in `.eval()` — same contract as
`extract.py`, and for the same reasons.
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

from sr.probes.extract import (BAND_NAMES, CKPT_GLOB, SR_DIR, chip_readout,
                               discover, load_fixture, load_model, read_chip,
                               run_meta)
from sr.probes.make_fixtures import FIXTURE_DIR

# Radii of the ideal low-pass disks, in cycles over the 512 px HR grid. At
# 2.5 m ground sampling a radius r passes everything COARSER than 1280/r m, so
# the sweep walks from "r0's DC only" straight through road width (r=128 is a
# 10 m wavelength) to "r0 almost everywhere". 35 is included because it is the
# HC's own Gaussian sigma and pins the curve to the deployed operator.
RADII = (0, 8, 16, 24, 35, 48, 64, 96, 128, 192, 256)
HC_MASK = "hard_constraint.safetensor"


# --------------------------------------------------------------- the splice
def load_hc_mask(sr_dir: Path):
    """The shipped sigma=35 low-pass mask, in `HardConstraint`'s own convention.

    Read rather than re-derived: the deployed cutoff is the value Table 4 of
    Aybar et al. optimised, and `sen2sr_loader.build_hard_constraint` loads this
    same file for the model itself. A re-derivation that differed by a pixel
    would make the curve's sigma=35 point not quite the HC's cut.
    """
    import safetensors.torch

    p = Path(sr_dir) / HC_MASK
    if not p.exists():
        raise SystemExit(f"{p} missing — the splice needs the shipped HC mask.")
    return safetensors.torch.load_file(p)["weights"].float()


def ideal_mask(size: int, radius: int, device, torch):
    """A fftshifted ideal low-pass disk of `radius`, matching `ideal_filter`.

    `distance <= cutoff` is upstream's own boundary convention (sen2sr.models
    .tricks.ideal_filter), so radius 0 passes the DC bin alone rather than
    nothing at all.
    """
    c = size // 2
    g = torch.arange(size, device=device, dtype=torch.float32) - c
    d = (g[:, None] ** 2 + g[None, :] ** 2).sqrt()
    return (d <= radius).float()


def splice(lo_src, hi_src, mask, torch):
    """Low band of `lo_src` + high band of `hi_src`, the HardConstraint algebra.

    Identical to `HardConstraint.forward` with `lr_up` replaced by an arbitrary
    donor: fftn -> fftshift -> `M*lo + (1-M)*hi` -> ifftshift -> real(ifft2).
    Kept as one function so the sigma=35 condition and every radius on the
    ablation curve are literally the same code path.
    """
    F_lo = torch.fft.fftshift(torch.fft.fftn(lo_src, dim=(-2, -1)), dim=(-2, -1))
    F_hi = torch.fft.fftshift(torch.fft.fftn(hi_src, dim=(-2, -1)), dim=(-2, -1))
    F = F_lo * mask + F_hi * (1 - mask)
    return torch.real(torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1))))


def mean_match(spliced, target, torch):
    """Restore `target`'s per-band spatial mean into `spliced`.

    A pure DC correction: it moves only the (0,0) Fourier bin, so the low-band
    STRUCTURE the splice imported from r0 is untouched and the difference
    against the un-matched condition is exactly the radiometric level.
    """
    d = target.mean(dim=(-2, -1), keepdim=True) - spliced.mean(dim=(-2, -1), keepdim=True)
    return spliced + d


# ---------------------------------------------------------------- the pass
class Decoder:
    """One loaded checkpoint, used only as `z-score -> U-Net -> probs`."""

    def __init__(self, model, torch):
        self.m, self.torch = model, torch
        self.scale = float(model.hparams.reflectance_scale)

    def sr(self, x):
        with self.torch.no_grad():
            return self.m._sr_forward(x)

    def probs(self, y):
        """`y` in RAW units -> (B,1,H,W) probabilities under this ckpt's z-score."""
        t = self.torch
        with t.no_grad(), t.autocast(device_type=y.device.type, enabled=False):
            x = (y - self.m.band_mean) / self.m.band_std
            return t.sigmoid(self.m.model(x).float())


def conditions(y_a, y_0, masks, torch, mean_matched=True):
    """[(name, tensor)] — every input variant for one chip, in raw units.

    Names are `<what the input is>`; the decoder is recorded separately. The
    two anchors come first so an analysis can normalise a curve against them
    without a lookup table.
    """
    out = [("own", y_a), ("r0", y_0)]
    M = masks["sigma35"]
    out.append(("swap_hi", splice(y_0, y_a, M, torch)))   # r0 low + arm high
    out.append(("swap_lo", splice(y_a, y_0, M, torch)))   # arm low + r0 high
    if mean_matched:
        out.append(("swap_hi_mm", mean_match(out[2][1], y_a, torch)))
        out.append(("swap_lo_mm", mean_match(out[3][1], y_a, torch)))
    for r in RADII:
        out.append((f"swap_hi_r{r}", splice(y_0, y_a, masks[f"r{r}"], torch)))
    return out


def extract_run(run: Path, args, fixture, ref_ckpt: Path, ref_model=None) -> dict:
    """Band-swap pass for one arm-seed against the r0 reference."""
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

    arm = Decoder(load_model(ckpt, Path(args.sr_dir), args.device), torch)
    ref = ref_model or Decoder(load_model(ref_ckpt, Path(args.sr_dir), args.device), torch)
    if abs(ref.scale - arm.scale) > 1e-9:
        raise SystemExit(
            f"{run.name} reflectance_scale={arm.scale} but the r0 reference is "
            f"{ref.scale}. The scale is coupled to the baked band statistics, so "
            "the two outputs are not in one unit and the splice would be "
            "meaningless. Re-express the reference first (viz_grid's rule).")

    theta, note = resolve_theta(run, args.select_on, args.default_theta)
    theta_prov = ("sweep" if note.startswith(args.select_on)
                  else "recorded" if note == "recorded θ*" else "fallback")
    theta_r0, note_r0 = resolve_theta(Path(args.ref_run), args.select_on, args.default_theta)

    crop, up = meta_fx["crop"], meta_fx["upscale"]
    bands = tuple(arm.m.hparams.bands)
    if bands != (1, 2, 3, 4):
        raise SystemExit(f"{run.name}: bands={bands}; {BAND_NAMES} assumes V2 order.")

    hr = crop * up
    M = load_hc_mask(args.sr_dir).to(args.device)
    if M.shape[-1] != hr:
        raise SystemExit(
            f"HC mask is {tuple(M.shape)} but the chip's HR grid is {hr}px. The "
            "shipped mask is 512x512 and the fixture's 128px crop at x4 is what "
            "matches it — a resize here would silently move the cutoff.")
    masks = {"sigma35": M}
    masks.update({f"r{r}": ideal_mask(hr, r, args.device, torch) for r in RADII})

    n = len(chips) if args.limit is None else min(args.limit, len(chips))
    rows = []
    t0 = time.perf_counter()

    for i in range(n):
        chip = chips[i]
        x_np, mask = read_chip(Path(meta_fx["dataset_dir"]), chip, crop, up,
                               bands, meta_fx["mask_dirname"])
        x = torch.from_numpy(np.ascontiguousarray(x_np))[None].to(args.device)

        y_a, y_0 = arm.sr(x), ref.sr(x)
        conds = conditions(y_a, y_0, masks, torch, mean_matched=not args.no_mean_match)

        # --- input-space residual against the deployed tensor, per condition.
        # The AP-free half of the self-test: for an HC-on arm `swap_hi` should
        # reproduce `own`, and whether it does is a property of the TENSOR, not
        # of a downstream ranking statistic on one chip. `sr_pad=8` means the
        # deployed splice ran on a 576px grid and was cropped, so the identity
        # is approximate at the borders and this column is how approximate.
        na = float(y_a.pow(2).mean().sqrt())
        resid = {name: float((t - y_a).pow(2).mean().sqrt()) / na
                 for name, t in conds}

        # --- every input on the ARM's decoder (the deployed pairing + swaps)
        for j in range(0, len(conds), args.batch):
            chunk = conds[j:j + args.batch]
            p = arm.probs(torch.cat([c[1] for c in chunk], dim=0))
            for k, (name, _) in enumerate(chunk):
                ap, tp, fp, fn, tn = chip_readout(p[k, 0].cpu().numpy(), mask,
                                                  theta, args.ap_bins)
                rows.append(dict(chip=i, condition=name, decoder="arm", ap=ap,
                                 tp=tp, fp=fp, fn=fn, tn=tn,
                                 resid=resid[name], road_px=int(mask.sum())))

        # --- the transfer row: the same two anchors on r0's decoder. This is
        # the cell extract.py never computed, and it is what separates "the
        # front-end supplied the high band" from "the weights learned to use
        # it": r0's U-Net never trained on an adapted input.
        p = ref.probs(torch.cat([y_a, y_0], dim=0))
        for k, name in enumerate(("own", "r0")):
            ap, tp, fp, fn, tn = chip_readout(p[k, 0].cpu().numpy(), mask,
                                              theta_r0, args.ap_bins)
            rows.append(dict(chip=i, condition=name, decoder="r0", ap=ap, tp=tp,
                             fp=fp, fn=fn, tn=tn, resid=resid[name],
                             road_px=int(mask.sum())))

        if (i + 1) % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"    {i + 1}/{n} chips  {el:.0f}s  ({el / (i + 1):.2f} s/chip)",
                  flush=True)

    elapsed = time.perf_counter() - t0
    pd.DataFrame(rows).to_parquet(out_dir / "bandswap.parquet", index=False)

    meta = {
        **run_meta(run),
        "instrument": "bandswap",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(ckpt), "epoch": ckpt_epoch, "global_step": ckpt_step,
        "fixture_hash": fx_hash, "fixture_meta": meta_fx, "n_chips": n,
        "theta": theta, "theta_provenance": theta_prov, "theta_note": note,
        "theta_r0": theta_r0, "theta_r0_note": note_r0,
        "theta_select_on": args.select_on,
        "reflectance_scale": arm.scale,
        "band_mean": [float(v) for v in arm.m.band_mean.reshape(-1)],
        "band_std": [float(v) for v in arm.m.band_std.reshape(-1)],
        "band_names": list(BAND_NAMES),
        "radii": list(RADII), "hc_mask": str(Path(args.sr_dir) / HC_MASK),
        "mean_matched": not args.no_mean_match,
        "reference_run": Path(args.ref_run).name, "reference_ckpt": str(ref_ckpt),
        "conditions": [c for c, _ in conditions(
            torch.zeros(1, 4, hr, hr, device=args.device),
            torch.zeros(1, 4, hr, hr, device=args.device), masks, torch,
            mean_matched=not args.no_mean_match)],
        "ap_bins": args.ap_bins, "device": args.device,
        "hparams": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v)) for k, v in dict(arm.m.hparams).items()},
        "seconds": round(elapsed, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {out_dir}  ({elapsed:.0f}s, {len(rows)} rows, "
          f"theta={theta:g} [{theta_prov}])", flush=True)
    del arm
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", action="append", default=None)
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--include", default="*r2grid*",
                    help="glob a run-dir name must match (default: the lr_sr grid)")
    ap.add_argument("--ckpt-glob", default=CKPT_GLOB)
    ap.add_argument("--sr-dir", default=SR_DIR)
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--dataset-dir", default=None,
                    help="override the fixture's baked-in dataset_dir (the "
                         "dir holding test/imagery) -- needed on any machine "
                         "other than the one make_fixtures ran on")
    ap.add_argument("--out-dir", default="bandswap_cache")
    ap.add_argument("--ref-run", required=True,
                    help="the r0 run dir supplying the low-band donor")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=6,
                    help="conditions per U-Net forward (memory vs throughput)")
    ap.add_argument("--ap-bins", type=int, default=101)
    ap.add_argument("--select-on", default="iou")
    ap.add_argument("--default-theta", type=float, default=0.5)
    ap.add_argument("--no-mean-match", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    runs = [Path(r) for r in (args.run or [])]
    runs += discover(args.runs_dir or [], args.include, args.ckpt_glob)
    if not runs:
        raise SystemExit("no runs discovered — check --runs-dir / --include")

    from sr.viz_models import find_ckpt, resolve_theta

    ref_ckpt = find_ckpt(Path(args.ref_run), args.ckpt_glob)
    if ref_ckpt is None:
        raise SystemExit(f"--ref-run {args.ref_run} holds no {args.ckpt_glob}")

    print(f"{len(runs)} run(s), reference {Path(args.ref_run).name}:")
    for r in runs:
        th, note = resolve_theta(r, args.select_on, args.default_theta)
        m = run_meta(r)
        print(f"  {m['arm']:<5} seed {str(m['seed']):<5} theta*={th:<6g} "
              f"[{note:<12}] {r.name}")
    n_cond = len(RADII) + (6 if not args.no_mean_match else 4)
    print(f"{n_cond} conditions/chip on the arm decoder + 2 on r0's")

    fixture = load_fixture(Path(args.fixture_dir))
    meta_fx, chips = fixture[0], fixture[1]
    # The fixture bakes in the machine it was BUILT on ("dataset_dir"), so the
    # same frozen chip list on another box points at a path that does not
    # exist. Overriding it here keeps the fixture itself immutable -- its hash
    # is what puts arms on one axis and must not change per machine.
    if args.dataset_dir:
        meta_fx["dataset_dir"] = str(args.dataset_dir)
    probe = Path(meta_fx["dataset_dir"]) / chips[0]["image_path"]
    if not probe.exists():
        raise SystemExit(
            f"fixture chip 0 is not readable:\n  {probe}\n"
            f"The fixture was built on another machine (dataset_dir="
            f"{meta_fx['dataset_dir']!r}). Pass --dataset-dir pointing at the "
            "directory that holds test/imagery/ here.")
    print(f"fixture {fixture[3]}: {meta_fx['n_chips']} chips, "
          f"{meta_fx['split']} split, data at {meta_fx['dataset_dir']}", flush=True)

    if args.dry_run:
        return 0

    if args.device is None:
        import torch
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {args.device}", flush=True)

    # The reference is loaded ONCE and reused: it is the same checkpoint for
    # every arm, and reloading it per arm would repeat the slowest step in the
    # pass for no reason.
    import torch
    ref_model = Decoder(load_model(ref_ckpt, Path(args.sr_dir), args.device), torch)

    for r in runs:
        print(f"\n--- {r.name}", flush=True)
        extract_run(r, args, fixture, ref_ckpt, ref_model=ref_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
