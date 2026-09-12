"""Paired contrast maps — where a band, or the SR itself, changes the evidence.

    PYTHONPATH=src python -m sr.probes.contrast \
        --run <arm run dir> --ref-run <r0 run dir> \
        --sr-dir models/SEN2SRLite_RGBN --chips 315 --device mps

The companion to `saliency.py`, asking the question from the other end. A
saliency map fixes ONE pixel of interest and sweeps an occluder over the image:
"what did the model use to classify this pixel". A contrast map fixes ONE
treatment and reads the whole logit field: "where in this image does that
treatment change the road evidence". No pixel has to be chosen, nothing is
swept, and the whole thing costs a handful of forwards per chip instead of
thousands.

TWO CONTRASTS, THE TWO QUESTIONS
--------------------------------
**band** — baseline is the image with band b flattened to the checkpoint's own
`band_mean` everywhere (so the channel is exactly zero after its z-score);
treatment is the intact image. Δ = intact − occluded, so with `--bands NIR` the
map is *what NIR adds over RGB alone*, spatially resolved. Every band can be
run; each costs one forward.

**model** — baseline is whatever `--ref-run` supplies, treatment is the arm's
own SR output, BOTH read by the arm's own U-Net with its own z-score (the
normaliser is part of the reader, never part of the stimulus). With an r0
reference the map is *what the SR added over plain upsampling*; with an r1a
reference it is *what joint training added over the frozen generator* — the
r2−r1 contrast. The reference supplies an image and nothing else.

Both are the spatial version of conditions the occlusion suite already scores
as scalars (`band_<b>@all` and `cf_bic@<region>` in `occl_context_extract`):
same interventions, same fills, same sign convention, so a map here and a bar
there are two views of one number. What the map adds is WHERE.

NO AFFINE MATCHING HERE, DELIBERATELY
--------------------------------------
The suite affine-matches a counterfactual PATCH to the destination's local
statistics, because pasting a raw patch into an otherwise-untouched image
creates a seam and the delta would measure the seam. Here the whole image is
swapped, so there is no seam and no boundary to match across — and the
radiometric difference between the two generators is part of the treatment,
not an artefact of it. For an HC-on arm the two share their low band anyway
(measured: global per-band means agree to 2e-5 of RMS). For an HC-OFF arm they
do not, and that drift is a real component of the map: read those with
`docs/attribution_plan.md`'s reflectance-scale caveat in mind.

SIGN
----
Positive (red) = removing that band, or falling back to bicubic, COSTS road
evidence there. Negative (blue) = the model does better without it. A diverging
map centred at zero, because both directions happen and a sequential colormap
would hide half the finding.

fp32, `.eval()`, no autograd — the suite's contract.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

import numpy as np

from sr.probes.attr_extract import NDVI_TAU, region_masks
from sr.probes.extract import (BAND_NAMES, CKPT_GLOB, SR_DIR, load_fixture,
                               load_model, read_chip, run_meta)
from sr.probes.make_fixtures import FIXTURE_DIR
from sr.probes.occl_context_extract import Decoder, REGIONS
from sr.probes.saliency import rgb_limits, rgb_of
from sr.probes.style import FIGURES_DIR


def contrasts(dec, y, y_ref, bands, torch, ref_name="ref"):
    """{name: (H, W) Δlogit}, plus the intact logit field.

    One forward for the intact field, one per band, one for the model swap.
    """
    with torch.no_grad():
        intact = dec.logits(y.unsqueeze(0))[0]
        out = {}
        bm = dec.m.band_mean.reshape(-1)
        for name in bands:
            b = BAND_NAMES.index(name)
            v = y.clone()
            v[b] = bm[b]
            out[f"band_{name}"] = (intact - dec.logits(v.unsqueeze(0))[0])
        if y_ref is not None:
            # Named after the ARM that supplied it, not after what that arm
            # happens to be: r0 makes this "what the SR added over plain
            # upsampling", r1a makes it "what JOINT TRAINING added over the
            # frozen generator" — the r2−r1 contrast. Same arithmetic,
            # different question, and every output name has to say which, since
            # both land in one directory.
            out[f"model_sr_vs_{ref_name}"] = (
                intact - dec.logits(y_ref.unsqueeze(0))[0])
    return {k: v.float().cpu().numpy() for k, v in out.items()}, \
        intact.float().cpu().numpy()


def arm_pair(arm, ref_name):
    """`r2a_r1a` — treatment arm, then the arm supplying the baseline image.

    One directory holds every pair, so the pair belongs in every filename. An
    arm with no model contrast is just its own key.
    """
    return f"{arm}_{ref_name}" if ref_name else str(arm)


def region_stats(dmap, masks, **extra):
    """Mean Δ, mean |Δ|, and the share of pixels helped, per region.

    `frac_positive` is the part a mean hides: a band can be neutral on average
    while mattering strongly in both directions, and on a road/background split
    that is a different finding from "it does nothing".
    """
    rows = []
    for region in REGIONS:
        m = masks[region]
        if not m.any():
            continue
        v = dmap[m]
        rows.append(dict(region=region, n_px=int(m.sum()), mean=float(v.mean()),
                         mean_abs=float(np.abs(v).mean()),
                         frac_positive=float((v > 0).mean()),
                         p99=float(np.percentile(v, 99)),
                         p01=float(np.percentile(v, 1)), **extra))
    return rows


def plot_contrasts(axes, panels, road, vmax):
    """Draw an ordered list of (kind, payload, title) panels.

    ONE diverging scale across every contrast map, band and model alike. The
    two quantities do sit in different regimes — the model contrast runs larger
    than any single band — so the smaller maps compress; the trade is that a
    reader can compare a band panel against the model panel directly instead of
    checking which of two colourbars applies to which panel.
    """
    im = None
    for ax, (kind, payload, title) in zip(axes, panels):
        if kind == "rgb":
            ax.imshow(payload)
        elif kind == "mask":
            # Prediction as a filled field with the GT outline on top, so a
            # miss reads as contour-without-fill and a false positive as
            # fill-without-contour.
            ax.imshow(np.where(payload, 0.25, 1.0), cmap="gray", vmin=0, vmax=1)
        else:
            im = ax.imshow(payload, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        if kind != "rgb" and road.any():
            ax.contour(road.astype(float), levels=[0.5], colors=["#2c6fbb"],
                       linewidths=0.45, alpha=0.8)
        ax.set_title(title, fontsize=7.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    return im


def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref-run", default=None,
                    help="run dir supplying the model contrast's BASELINE "
                         "image: r0 for 'what the SR added over bicubic', r1a "
                         "for 'what joint training added over frozen SEN2SR'. "
                         "Omit to skip the model contrast.")
    ap.add_argument("--ckpt-glob", default=CKPT_GLOB)
    ap.add_argument("--sr-dir", default=SR_DIR)
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--out-dir", default=str(Path(FIGURES_DIR) / "contrast"))
    ap.add_argument("--chips", type=int, nargs="+", required=True)
    ap.add_argument("--bands", nargs="*", default=list(BAND_NAMES),
                    choices=BAND_NAMES,
                    help="each is one forward; NIR alone answers the "
                         "'RGB baseline, NIR treatment' question")
    ap.add_argument("--tau", type=float, default=NDVI_TAU)
    ap.add_argument("--select-on", default="iou",
                    help="sweep criterion θ* is re-argmaxed on, for the "
                         "predicted-mask panel only")
    ap.add_argument("--default-theta", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=99.0,
                    help="percentile setting the symmetric colour limit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    fixture = load_fixture(Path(args.fixture_dir))
    meta_fx, chips, _px, fx_hash = fixture
    if args.dataset_dir:
        meta_fx["dataset_dir"] = str(args.dataset_dir)
    n_fwd = 1 + len(args.bands) + (1 if args.ref_run else 0)
    print(f"{len(args.chips)} chip(s) x {n_fwd} forwards = "
          f"{len(args.chips) * n_fwd} total — seconds, not minutes")
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
    from sr.viz_models import find_ckpt, resolve_theta

    ckpt = find_ckpt(Path(args.run), args.ckpt_glob)
    dec = Decoder(load_model(ckpt, Path(args.sr_dir), args.device), torch)
    arm = run_meta(Path(args.run))
    ref, ref_name = None, None
    if args.ref_run:
        ref_meta = run_meta(Path(args.ref_run))
        ref_name = ref_meta["arm"]
        ref = Decoder(load_model(find_ckpt(Path(args.ref_run), args.ckpt_glob),
                                 Path(args.sr_dir), args.device), torch)
        print(f"model contrast baseline: {ref_meta['arm']} "
              f"(seed {ref_meta['seed']}) -> pair "
              f"{arm_pair(arm['arm'], ref_name)}")
        if abs(ref.scale - dec.scale) > 1e-9:
            raise SystemExit(
                f"reflectance_scale {dec.scale} vs {ref.scale}: the two outputs "
                "are not in one unit, so the model contrast would be a unit "
                "conversion rather than a treatment.")
    theta, theta_note = resolve_theta(Path(args.run), args.select_on,
                                      args.default_theta)
    print(f"predicted-mask panel drawn at θ*={theta:g} [{theta_note}] — the ONE "
          "θ-dependent element here; every contrast map is threshold-free.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style.apply_rc()
    import matplotlib.pyplot as plt

    rows = []
    for i in args.chips:
        chip = chips[i]
        x_np, mask = read_chip(Path(meta_fx["dataset_dir"]), chip,
                               meta_fx["crop"], meta_fx["upscale"],
                               tuple(dec.m.hparams.bands), meta_fx["mask_dirname"])
        x = torch.from_numpy(np.ascontiguousarray(x_np))[None].to(args.device)
        y = dec.sr(x)[0]
        y_ref = ref.sr(x)[0] if ref else None
        maps, intact = contrasts(dec, y, y_ref, args.bands, torch,
                                 ref_name or "ref")
        pair = arm_pair(arm["arm"], ref_name)
        masks = region_masks(x_np, mask, meta_fx["upscale"], args.tau)
        for name, m in maps.items():
            # A band map does not depend on the baseline, so it keeps the arm's
            # name; only the model contrast carries the pair.
            stem = (f"chip{i:04d}_{pair}_model" if name.startswith("model")
                    else f"chip{i:04d}_{arm['arm']}_{name}")
            np.save(out / f"{stem}.npy", m.astype("float32"))
            rows += region_stats(m, masks, chip=i, contrast=name, pair=pair,
                                 baseline=ref_name, run=arm["run"],
                                 arm=arm["arm"], seed=arm["seed"],
                                 tile=chip["tile"])

        names = list(maps)
        band_names = [n for n in names if n.startswith("band_")]
        model_key = f"model_sr_vs_{ref_name}" if ref_name else None
        # One scale over every map drawn, so the panels compare directly.
        vmax = max((float(np.percentile(np.abs(maps[n]), args.clip))
                    for n in names), default=1.0) or 1.0

        lims = rgb_limits(y.float().cpu().numpy())
        pred = 1.0 / (1.0 + np.exp(-intact)) > theta
        panels = [("rgb", rgb_of(y.float().cpu().numpy(), lims=lims),
                   f"{arm['arm']} SR output")]
        if y_ref is not None:
            # The baseline drawn on the TREATMENT's stretch. On its own
            # percentiles it would look like the same picture however far apart
            # the two generators actually are.
            panels.append(("rgb", rgb_of(y_ref.float().cpu().numpy(), lims=lims),
                           f"baseline: {ref_name} output"))
        panels.append(("mask", pred, f"predicted mask  (θ*={theta:g})"))
        if model_key:
            panels.append(("map", maps[model_key], f"SR − {ref_name}\n"
                                                   f"mean {maps[model_key].mean():+.3f}"))
        for n in band_names:
            panels.append(("map", maps[n],
                           f"{n.split('_')[1]} − without it\n"
                           f"mean {maps[n].mean():+.3f}"))

        n_ax = len(panels)
        ncol = 4 if n_ax > 6 else min(3, n_ax)
        nrow = int(np.ceil(n_ax / ncol))
        fig, axg = plt.subplots(nrow, ncol,
                                figsize=(style.FULL_WIDTH_IN * 0.94, 2.35 * nrow))
        axes = np.atleast_1d(axg).ravel()
        im = plot_contrasts(axes, panels, masks["road"], vmax)
        for ax in axes[n_ax:]:
            ax.axis("off")
        fig.subplots_adjust(top=0.93, bottom=0.13, hspace=0.25)
        cax = fig.add_axes([0.30, 0.055, 0.40, 0.022])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Δ logit   (treatment − baseline)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"C_chip{i:04d}_{pair}.{ext}", bbox_inches=None)
        for q in (out / f"C_chip{i:04d}_{pair}.pdf",
                  out / f"C_chip{i:04d}_{pair}.png"):
            print(f"  wrote {q}")
        plt.close(fig)

    pair = arm_pair(arm["arm"], ref_name)
    df = pd.DataFrame(rows)
    # Per pair, not one shared file: two runs into the same directory would
    # otherwise overwrite each other's numbers.
    df.to_csv(out / f"contrast_region_stats_{pair}.csv", index=False)
    (out / f"meta_{pair}.json").write_text(json.dumps({
        **arm, "instrument": "contrast", "checkpoint": str(ckpt),
        "reference_run": args.ref_run, "reference_name": ref_name, "fixture_hash": fx_hash,
        "chips": list(args.chips), "bands": list(args.bands),
        "ndvi_tau": args.tau, "clip_percentile": args.clip,
        "theta": theta, "theta_note": theta_note,
        "device": args.device}, indent=1))
    print(f"\nwrote {out / f'contrast_region_stats_{pair}.csv'}")
    print(df.pivot_table(index=["chip", "contrast"], columns="region",
                         values="mean").to_string(float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
