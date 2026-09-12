"""Per-pixel occlusion saliency maps — the qualitative companion to the suite.

    PYTHONPATH=src python -m sr.probes.saliency \
        --run <arm run dir> --sr-dir models/SEN2SRLite_RGBN \
        --chips 315 --n-pixels 3 --patch 16 --stride 8 --device mps

Reproduces Figure 1 of O'Sullivan & Dev (IGARSS 2025) on this project's road
task: for ONE pixel of interest, which parts of the input the model used to
classify it. The spectral image with the pixel marked, then one
occlusion-sensitivity map per fill — the all-band fill the paper uses, and one
per band, which is what turns a picture of WHERE the model looked into a
picture of where it looked IN EACH BAND. Each panel carries its own weighted
context, so the qualitative panel and the global metric are read together.

WHY THIS IS NOT A FLAG ON `occl_context_extract`
-------------------------------------------------
Different geometry, deliberately. The suite partitions the grid (stride =
patch) because that makes the saliency map patch-constant, which is what lets
weighted context collapse to per-patch distance sums over 96 chips cheaply.
A figure wants the opposite trade: OVERLAPPING windows (stride < patch),
accumulated into the map over the area each patch covers — the paper's step 4 —
which is smooth enough to look at and far too expensive to run over a
subsample. At patch 16 / stride 8 that is 63x63 = 3969 forwards for one chip;
at the paper's patch 8 / stride 1 it would be 255,025.

One pass serves EVERY pixel of interest on the chip: occluding a patch changes
the whole logit field at once, so the marginal cost of another marked pixel is
a gather, not a forward. Pick several. Each FILL, by contrast, is its own pass:
`--fills all R G B NIR` costs five times one map.

WHAT IS PLOTTED
---------------
Δ = intact logit at p − occluded logit at p, accumulated over every window
covering a location and divided by the cover count, then normalised to [0,1]
(`--norm minmax`, the literal reading of the paper; `--norm relu` clips the
patches whose removal HELPED). Positive = occluding there cost the model
evidence at p: reliance, in the paper's sign convention.

The GT road mask is drawn under the map in grey, as the paper greys its
land/water classes — it is the structure the reader needs to judge whether the
model reached for the road it was classifying or for something else.

TWO SCALES, ON PURPOSE
----------------------
Weighted context needs its map on [0,1], normalised per map, which is what the
paper specifies and what makes W comparable. But normalising each BAND's map
independently would draw a band the model barely uses as hot as the band it
leans on — the per-band comparison would be destroyed by the very step the
metric requires. So the panels are drawn on ONE scale shared across the fills
of a given pixel (the largest |Δ| among them), each annotated with its own raw
peak and its own W. The colour compares; the number follows the paper.

fp32, `.eval()`, no autograd — the suite's contract. `freeze_sr`'s `no_grad`
wrap is harmless and is left alone.
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
from sr.probes.extract import (BAND_NAMES, CKPT_GLOB, SR_DIR, load_fixture,
                               load_model, read_chip, run_meta)
from sr.probes.make_fixtures import FIXTURE_DIR
from sr.probes.occl_context_extract import Decoder, sample_road_pixels
from sr.probes.style import FIGURES_DIR

PATCH, STRIDE = 16, 8


# ------------------------------------------------------------------ geometry
def starts(size: int, patch: int, stride: int) -> list[int]:
    """Window origins, with the last one clamped so the grid is fully covered."""
    xs = list(range(0, max(size - patch, 0) + 1, stride))
    if xs and xs[-1] + patch < size:
        xs.append(size - patch)
    return xs


def spread_pixels(mask, n, fixture_hash, chip_idx, pool=96, margin=64):
    """`n` road pixels, deterministic and spread out.

    Drawn from the SAME `sample_road_pixels` pool the suite samples, so a pixel
    shown here is a pixel the weighted-context distribution was computed over —
    then thinned by farthest-point selection, because three pixels a few metres
    apart make three copies of one figure.
    """
    # Keep the mark off the border: a pixel in the top row has half its
    # context outside the chip, and the figure is about the context.
    h, w = mask.shape
    inner = np.zeros_like(mask)
    inner[margin:h - margin, margin:w - margin] = True
    cand = sample_road_pixels(mask & inner if (mask & inner).any() else mask,
                              pool, fixture_hash, chip_idx)
    pts = np.stack([cand // w, cand % w], axis=1).astype("float64")
    chosen = [0]
    d = np.hypot(*(pts - pts[0]).T)
    while len(chosen) < min(n, len(cand)):
        k = int(d.argmax())
        chosen.append(k)
        d = np.minimum(d, np.hypot(*(pts - pts[k]).T))
    return cand[sorted(chosen)]


def weighted_context_dense(smap, py, px) -> float:
    """Eq. 1 on a full-resolution map (the suite's fast path assumes patch-constant)."""
    h, w = smap.shape
    g_y = np.arange(h, dtype="float64")[:, None] - py
    g_x = np.arange(w, dtype="float64")[None, :] - px
    d = np.sqrt(g_y ** 2 + g_x ** 2)
    return float((smap * d).sum() / d.sum())


def normalise(a, norm="minmax"):
    """The paper's "normalised to [0,1]", both readings (see occl_context.py)."""
    if norm == "relu":
        a = np.clip(a, 0, None)
        return a / a.max() if a.max() > 0 else a
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)


# -------------------------------------------------------------------- the pass
def saliency_maps(dec, y, pix, patch, stride, batch, torch, band=None, log=None):
    """(len(pix), H, W) mean Δlogit per location, one dense pass for all pixels.

    `band=None` flattens every band inside the window (the paper's fill);
    `band=b` flattens only band b, leaving the other three intact — the
    per-band map. Both use the checkpoint's own `band_mean`, so the occluded
    channel is exactly zero after its z-score, matching instrument C.
    """
    h = y.shape[-1]
    pix_t = torch.as_tensor(np.asarray(pix), device=y.device)
    with torch.no_grad():
        l0 = dec.logits(y.unsqueeze(0)).reshape(1, -1)[:, pix_t][0].cpu().numpy()
    acc = np.zeros((len(pix), h, h), dtype="float64")
    cov = np.zeros((h, h), dtype="float64")
    wins = [(r, c) for r in starts(h, patch, stride) for c in starts(h, patch, stride)]
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
            lg = dec.logits(v).reshape(len(chunk), -1)[:, pix_t].cpu().numpy()
        for k, (r, c) in enumerate(chunk):
            acc[:, r:r + patch, c:c + patch] += (l0 - lg[k])[:, None, None]
            cov[r:r + patch, c:c + patch] += 1
        if log and (i // batch) % log == 0:
            done = min(i + batch, len(wins))
            el = time.perf_counter() - t0
            print(f"    {done}/{len(wins)} windows  {el:.0f}s "
                  f"({el / max(done, 1) * len(wins):.0f}s projected)", flush=True)
    return acc / np.maximum(cov, 1), l0


# --------------------------------------------------------------------- figure
def rgb_limits(y_np, lo=2, hi=98):
    """[(lo, hi)] per band — the stretch, factored out so it can be SHARED.

    Two images drawn with their own percentiles look alike no matter how far
    apart they are, because each is normalised to its own range. Anything
    comparing a treatment image against a baseline one has to fix the stretch
    on one of them and reuse it.
    """
    return [(float(np.percentile(y_np[b], lo)),
             float(np.percentile(y_np[b], hi))) for b in range(3)]


def rgb_of(y_np, lo=2, hi=98, lims=None):
    """True colour from bands (R, G, B), stretched PER BAND.

    A single stretch across all three keeps the scene's colour cast, which on a
    savanna tile means a dark red-brown wash that hides the roads the figure is
    about. Per-band is the standard remote-sensing composite. Pass `lims` from
    `rgb_limits` to draw a second image on the first one's scale.
    """
    lims = lims or rgb_limits(y_np, lo, hi)
    return np.stack([np.clip((y_np[b] - a) / max(z - a, 1e-9), 0, 1)
                     for b, (a, z) in enumerate(lims)], axis=-1)


NULL_FRACTION = 0.10


def plot_panels(axes, rgb, maps, road, py, px, fills, wcs, vmax, peak_all):
    """Spectral image + one panel per fill, on ONE shared colour scale.

    `maps` are RAW Δlogit, clipped at zero and drawn against a shared `vmax`.
    Two reasons not to draw the [0,1]-normalised maps W is computed from:
    every panel would peak at 1.0 by construction, so a band the model barely
    uses would look exactly as hot as the one it leans on; and min-max puts the
    bulk of a map at mid-scale whenever one patch is strongly negative, which
    paints the whole background yellow instead of the paper's white. Clipping
    at zero is the honest reading here anyway — negative means occluding there
    HELPED, which is not reliance.

    A band whose peak is under `NULL_FRACTION` of the all-band peak has its W
    parenthesised: weighted context is computed on a min-max normalised map, so
    a band the model barely uses normalises its own NOISE to [0,1] and scores a
    large W for having no structure at all. That number is an artefact, and the
    figure has to say so rather than invite the reader to rank on it.
    """
    ax = axes[0]
    ax.imshow(rgb)
    ax.plot([px], [py], marker="o", markerfacecolor="none",
            markeredgecolor="#e8262f", markersize=9, markeredgewidth=1.6)
    ax.set_title("Spectral image\n ", fontsize=8)
    im = None
    for ax, name, m, w in zip(axes[1:], fills, maps, wcs):
        im = ax.imshow(np.clip(m, 0, None), cmap="hot_r", vmin=0, vmax=vmax)
        # GT road as a contour ON TOP, not a wash underneath: the paper greys
        # its land/water classes, but a grey underlay beneath an opaque
        # colormap is invisible, which is what the first render showed.
        if road.any():
            ax.contour(road.astype(float), levels=[0.5], colors=["#3b7bbf"],
                       linewidths=0.45, alpha=0.75)
        # An open ring, not a filled cross: at patch 16 the hotspot is a few
        # pixels across and a solid marker sits exactly on top of the thing the
        # panel exists to show.
        ax.plot([px], [py], marker="o", markerfacecolor="none",
                markeredgecolor="#1c4f86", markersize=8, markeredgewidth=1.3)
        peak = float(m.max())
        near_null = name != "all" and peak < NULL_FRACTION * peak_all
        wtxt = f"({w:.3f})" if near_null else f"{w:.3f}"
        label = "all bands" if name == "all" else f"{name} only"
        ax.set_title(f"{label}   peak {peak:+.2f}\nW {wtxt}", fontsize=7.5,
                     color="0.45" if near_null else "black")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)          # every panel the same square
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    return im


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the arm's run dir")
    ap.add_argument("--ckpt-glob", default=CKPT_GLOB)
    ap.add_argument("--sr-dir", default=SR_DIR)
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--out-dir", default=str(Path(FIGURES_DIR) / "saliency"))
    ap.add_argument("--chips", type=int, nargs="+", required=True,
                    help="fixture chip indices to map")
    ap.add_argument("--pixels", type=int, nargs="*", default=None,
                    help="explicit flat pixel indices (overrides --n-pixels)")
    ap.add_argument("--n-pixels", type=int, default=3)
    ap.add_argument("--fills", nargs="+", default=["all"],
                    choices=("all",) + BAND_NAMES,
                    help="'all' is the paper's fill; a band name flattens ONLY "
                         "that band inside the window. Each fill is its own "
                         "dense pass.")
    ap.add_argument("--patch", type=int, default=PATCH)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--norm", default="minmax", choices=("minmax", "relu"))
    ap.add_argument("--tau", type=float, default=NDVI_TAU)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--reuse-maps", action="store_true",
                    help="re-draw from the .npy maps already in --out-dir "
                         "instead of running the pass. Restyling a figure "
                         "should not cost 4000 forwards.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    fixture = load_fixture(Path(args.fixture_dir))
    meta_fx, chips, _px, fx_hash = fixture
    if args.dataset_dir:
        meta_fx["dataset_dir"] = str(args.dataset_dir)
    crop, up = meta_fx["crop"], meta_fx["upscale"]
    size = crop * up
    n_win = len(starts(size, args.patch, args.stride)) ** 2
    total = n_win * len(args.fills) * len(args.chips)
    print(f"{len(args.chips)} chip(s) x {len(args.fills)} fill(s) x {n_win} "
          f"windows (patch {args.patch}, stride {args.stride}) = {total} "
          f"forwards; ~{total / 17 / 60:.0f} min on MPS at ~17 fwd/s. One pass "
          "serves every marked pixel on a chip; each fill is its own pass.")
    for i in args.chips:
        c = chips[i]
        print(f"  chip {i:>3} road_px={c['road_px']:>6} {c['tile']}")
    if args.dry_run:
        return 0

    if args.device is None:
        import torch
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    import matplotlib
    import torch
    from sr.probes import style
    from sr.viz_models import find_ckpt

    ckpt = find_ckpt(Path(args.run), args.ckpt_glob)
    if ckpt is None:
        raise SystemExit(f"{args.run} holds no {args.ckpt_glob}")
    dec = Decoder(load_model(ckpt, Path(args.sr_dir), args.device), torch)
    arm = run_meta(Path(args.run))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style.apply_rc()
    import matplotlib.pyplot as plt

    for i in args.chips:
        chip = chips[i]
        x_np, mask = read_chip(Path(meta_fx["dataset_dir"]), chip, crop, up,
                               tuple(dec.m.hparams.bands), meta_fx["mask_dirname"])
        if not mask.any():
            print(f"  chip {i} has no GT road — nothing to classify; skipped")
            continue
        x = torch.from_numpy(np.ascontiguousarray(x_np))[None].to(args.device)
        y = dec.sr(x)[0]
        y_np = y.detach().float().cpu().numpy()
        pix = (np.asarray(args.pixels) if args.pixels
               else spread_pixels(mask, args.n_pixels, fx_hash, i))
        print(f"\n--- chip {i} ({chip['tile']}), {len(pix)} pixel(s), "
              f"{len(args.fills)} fill(s)", flush=True)
        rgb = rgb_of(y_np)
        road = region_masks(x_np, mask, up, args.tau)["road"]

        raw, l0 = {}, None
        if args.reuse_maps:
            found = sorted(out.glob(f"chip{i:04d}_px*_{args.fills[0]}_{arm['arm']}.npy"))
            if not found:
                raise SystemExit(
                    f"--reuse-maps found no chip{i:04d}_*_{args.fills[0]}_"
                    f"{arm['arm']}.npy under {out}; run the pass first.")
            pix = np.array([int(f.stem.split("_px")[1].split("_")[0]) * size
                            + int(f.stem.split("_px")[1].split("_")[1])
                            for f in found])
            for fname in args.fills:
                raw[fname] = np.stack([
                    np.load(out / f"chip{i:04d}_px{int(p) // size:03d}_"
                                  f"{int(p) % size:03d}_{fname}_{arm['arm']}.npy")
                    for p in pix])
            with torch.no_grad():
                lg = dec.logits(y.unsqueeze(0)).reshape(1, -1)
                l0 = lg[0, torch.as_tensor(pix, device=y.device)].cpu().numpy()
            print(f"  reused {len(pix) * len(args.fills)} cached map(s)")
        else:
            for fname in args.fills:
                b = None if fname == "all" else BAND_NAMES.index(fname)
                print(f"  fill {fname}", flush=True)
                raw[fname], l0 = saliency_maps(dec, y, pix, args.patch,
                                               args.stride, args.batch, torch,
                                               band=b, log=args.log_every)

        for k, p_flat in enumerate(pix):
            py, pxx = int(p_flat) // size, int(p_flat) % size
            # W is computed on each map's OWN [0,1] normalisation, per the
            # paper; the panels are then drawn on one shared scale so the bands
            # can be compared. Normalising each panel independently would draw
            # a band the model barely uses as hot as the one it leans on.
            norm_maps = [normalise(raw[f][k], args.norm) for f in args.fills]
            wcs = [weighted_context_dense(m, py, pxx) for m in norm_maps]
            # The shared scale comes from the RAW maps. Taking it from the
            # normalised ones (as the first version did) is vacuous: they all
            # peak at 1.0 by definition.
            plot_maps = [raw[f][k] for f in args.fills]
            vmax = max(float(np.percentile(np.clip(m, 0, None), 99.8))
                       for m in plot_maps) or 1.0
            for f in args.fills:
                np.save(out / f"chip{i:04d}_px{py:03d}_{pxx:03d}_{f}_{arm['arm']}.npy",
                        raw[f][k].astype("float32"))
            n_ax = 1 + len(args.fills)
            ncol = min(3, n_ax) if n_ax <= 4 else 3
            nrow = int(np.ceil(n_ax / ncol))
            fig, axg = plt.subplots(nrow, ncol,
                                    figsize=(style.FULL_WIDTH_IN * 0.82,
                                             2.55 * nrow))
            axes = np.atleast_1d(axg).ravel()
            peak_all = float(raw[args.fills[0]][k].max()) if args.fills else 1.0
            im = plot_panels(axes, rgb, plot_maps, road, py, pxx,
                             args.fills, wcs, vmax, peak_all)
            for ax in axes[n_ax:]:
                ax.axis("off")
            # A one-row grid puts its panel titles right where the suptitle
            # sits; two rows have room. Reserve it explicitly rather than
            # relying on the tight bbox, which does not know about overlap.
            fig.subplots_adjust(top=0.74 if nrow == 1 else 0.88)
            cb = fig.colorbar(im, ax=list(axes), fraction=0.028, pad=0.02)
            cb.set_label("Δ logit  (intact − occluded)", fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
            fig.suptitle(
                f"{style.label(arm['arm'])} · chip {i} · {chip['tile']} · "
                f"pixel ({py}, {pxx}) · intact logit {l0[k]:+.2f}",
                x=0.012, y=0.995, ha="left", fontsize=7)
            for q in style.save(fig, out,
                                f"S_chip{i:04d}_px{py:03d}_{pxx:03d}_{arm['arm']}"):
                print(f"  wrote {q}")
            plt.close(fig)

    (out / "meta.json").write_text(json.dumps({
        **arm, "instrument": "saliency", "checkpoint": str(ckpt),
        "fixture_hash": fx_hash, "chips": list(args.chips),
        "patch": args.patch, "stride": args.stride, "norm": args.norm,
        "fills": list(args.fills), "n_pixels": args.n_pixels,
        "n_windows": n_win, "ndvi_tau": args.tau, "device": args.device}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
