"""WHOLE-TILE visualisation of one JointSR+UNet checkpoint, for figures.

Sibling of `viz_single` (one 128 px crop, four panels in a row) and `viz_grid`.
This one renders a COMPLETE tile — 512 px at 10 m in, 2048 px at 2.5 m out —
as standalone full-resolution PNGs suitable for dropping into a document:

    <tile>_<tag>_sr.png          the SR image the UNet actually saw (bicubic x4
                                 for an r0 ckpt), RGB, 2048 x 2048
    <tile>_<tag>_pred_iou.png    sigmoid(logits) > θ*, one per selection
    <tile>_<tag>_pred_f1.png     criterion — IoU-, F1- and buffered-F1-tuned
    <tile>_<tag>_pred_bf1.png    operating points, side by side
    <tile>_<tag>_gt.png          the 2.5 m ground truth (--no-gt drops it)
    <tile>_<tag>_compare.png     SR | each prediction | GT, one contact sheet

θ ONLY AFFECTS THE FINAL COMPARISON, so every threshold is served by ONE
forward pass: the SR panel is emitted once (it cannot differ between criteria,
and three identical files would invite a reader to hunt for a difference) and
the predictions are re-binarisations of the same probability map. Rendering
three operating points therefore costs the same GPU as rendering one.

The panels are written separately and borderless (`imsave`, one pixel per
array element) precisely because inter-panel spacing is a document-layout
problem, not a plotting one — compose them in LaTeX and the figure keeps its
native resolution.

WHY THE TILE IS SCORED IN WINDOWS, NOT IN ONE PASS
--------------------------------------------------
A whole tile cannot simply be pushed through the model:

  * SEN2SR's shipped FFT `HardConstraint` mask PINS the LR input size
    (`model._required_lr`, 128 px). Anything else is a shape error.
  * Even for the fully-convolutional upsamplers (bicubic, SR4RS), running the
    whole tile in one pass would NOT reproduce the benchmarked prediction:
    `benchmarking.runner._score_tile_sr` evaluates in `cell_m`-sized footprint
    cells (2560 m -> 256 px at 10 m), and convolution borders differ between a
    256 px cell and a 512 px tile.

So the window unit here is the same one the bench used: `_required_lr` for a
pinned model, else `--cell-px`. That makes these figures show exactly the
prediction that produced the numbers in the store, rather than a
similar-looking one. Cells are stitched non-overlapping, as in the runner.

CONTRAST IS FIXED ACROSS MODELS, ON PURPOSE
-------------------------------------------
The percentile stretch is computed once from the ORIGINAL 10 m reflectance and
then applied to every panel (`--stretch`, default 2-98). Autoscaling each
model's SR output independently would make a model look sharper or brighter
purely because its histogram moved, which is exactly the artefact these
figures get used to argue about. Pass `--stretch-from-sr` to autoscale anyway.

    python -m sr.viz_tile \
        --ckpt <run>/checkpoints/unet_s2rosa_jointsr_final.ckpt \
        --dataset-dir <ROSA> --split val \
        --tile Durban_IndianCoastal_-29p851_30p94_Urban_r0_c3 \
        --sr-dir models/SEN2SRLite_RGBN --threshold 0.775 \
        --tag r2a --out-dir figures/

`--threshold` pins a single explicit θ and overrides `--select-on` entirely.
Left alone, each criterion's θ* is re-argmaxed from the run's sweep.json — the
same value the corresponding bench scored at.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_tile(src, bands, width, height):
    """Whole tile as (C, H, W) RAW values — the training loader's own read."""
    from rasterio.windows import Window

    from sentinel2data.dataset.joint_sr_dataset import _read_native

    # _read_native zero-pads to the requested square, matching the runner's
    # edge-cell behaviour; these tiles are square already.
    return _read_native(src, bands, Window(0, 0, width, height), max(width, height))


def _windows(size: int, step: int):
    return [(r, c) for r in range(0, size, step) for c in range(0, size, step)]


def run_tile(ckpt, sr_dir, img_chw, device, cell_px=256):
    """Replay a checkpoint over a whole tile in bench-sized windows.

    Returns (sr_chw, prob_hw) at `upscale` x the input grid — PROBABILITIES,
    not a binary mask, because θ only affects the final comparison. Rendering
    the same tile at three θ* (IoU-, F1- and buffered-F1-selected) therefore
    costs ONE forward pass, and the SR panel is provably identical across them
    rather than identical by luck.

    The SR tensor is the reflectance the UNet saw (this ckpt's own SR net +
    sr_pad pad/crop, BEFORE the z-score adapter) — the same quantity
    viz_single's "SR image" panel shows.
    """
    import torch

    from sr.model import JointSRUNetLightning

    kwargs = {"map_location": "cpu"}
    if sr_dir is not None:
        kwargs["sen2sr_dir"] = str(sr_dir)
    model = JointSRUNetLightning.load_from_checkpoint(str(ckpt), **kwargs)
    model = model.eval().float().to(device)

    # Some s2rosa ckpts saved reflectance_scale as None; forward divides by it.
    if getattr(model.hparams, "reflectance_scale", 10000.0) is None:
        model.hparams.reflectance_scale = 1.0
    rs = float(model.hparams.reflectance_scale)
    pad = int(model.hparams.sr_pad)
    up = int(model.hparams.upscale)
    req = model._required_lr                  # None = fully convolutional

    # The window the model is fed. Pinned models MUST see exactly `req`; the
    # others see the bench's footprint cell so borders match the scored run.
    step = int(req) if req else int(cell_px)
    C, H, W = img_chw.shape
    if H % step or W % step:
        raise SystemExit(
            f"tile {H}x{W} is not a multiple of the model's window ({step} px). "
            "Pinned SEN2SR needs an exact multiple of 128; pass --cell-px for "
            "the unpinned upsamplers."
        )

    sr_out = np.empty((C, H * up, W * up), dtype=np.float32)
    prob = np.empty((H * up, W * up), dtype=np.float32)

    with torch.no_grad():
        for r, c in _windows(H, step):
            chunk = img_chw[:, r:r + step, c:c + step]
            t = torch.from_numpy(np.ascontiguousarray(chunk))[None].float().to(device)

            # --- SR panel: the ckpt's own pad/crop, in reflectance units -----
            t_ref = t / rs
            t_sr = torch.nn.functional.pad(t_ref, (pad,) * 4, mode="reflect") if pad else t_ref
            hr = model.sr(t_sr)
            if pad:
                q = pad * up
                hr = hr[..., q:-q, q:-q]

            # --- prediction: the faithful end-to-end forward on RAW values ---
            logits = model(t)

            R, Cc = r * up, c * up
            sr_out[:, R:R + step * up, Cc:Cc + step * up] = hr[0].cpu().numpy()
            prob[R:R + step * up, Cc:Cc + step * up] = (
                torch.sigmoid(logits)[0, 0].cpu().numpy())

    return sr_out, prob


def to_rgb(x, lo, hi):
    """[B4,B3,B2,...] -> RGB in 0-1. Same band order as viz_grid.to_rgb."""
    rgb = np.transpose(np.asarray(x)[:3], (1, 2, 0))
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-dir", required=True,
                    help="ROSA root containing <split>/imagery and <split>/<mask-dirname>")
    ap.add_argument("--split", default="val",
                    help="the tile's split (default val — the decision split)")
    ap.add_argument("--tile", required=True, help="tile stem, no .tif")
    ap.add_argument("--mask-dirname", default="mask_new_2pt5")
    ap.add_argument("--sr-dir", default=None,
                    help="SR weights dir. REQUIRED for sen2sr/sr4rs ckpts: hparams "
                         "bake the training node's path. Omit for bicubic (r0).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="θ*. Default: derive it from the run's sweep.json for "
                         "--select-on, else 0.5.")
    ap.add_argument("--select-on", default="iou_mean,f1_mean,buffered_f1_mean",
                    help="comma/space list of criteria; one prediction PNG per "
                         "criterion, all off ONE forward pass. Each θ* is "
                         "re-argmaxed from the recorded sweep curve — sweep.json's "
                         "own `best_threshold` records whichever criterion wrote "
                         "the file LAST, so trusting it would silently render a "
                         "figure at a different operating point from the table it "
                         "illustrates.")
    ap.add_argument("--cell-px", type=int, default=256,
                    help="window for UNPINNED upsamplers; 256 = the bench's 2560 m "
                         "footprint cell at 10 m. Ignored when the model pins its input.")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98),
                    help="percentile stretch, computed on the ORIGINAL 10 m reflectance")
    ap.add_argument("--stretch-from-sr", action="store_true",
                    help="autoscale on the SR output instead (breaks cross-model comparability)")
    ap.add_argument("--no-gt", action="store_true", help="skip the ground-truth PNG")
    ap.add_argument("--no-pair", action="store_true", help="skip the side-by-side PNG")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    ds = Path(args.dataset_dir)
    img_path = ds / args.split / "imagery" / f"{args.tile}.tif"
    gt_path = ds / args.split / args.mask_dirname / f"{args.tile}.tif"
    if not img_path.is_file():
        raise SystemExit(f"tile not found: {img_path}")

    crits = [c for c in args.select_on.replace(",", " ").split() if c]
    SHORT = {"iou_mean": "iou", "f1_mean": "f1", "buffered_f1_mean": "bf1",
             "buffered_precision_mean": "bp", "buffered_recall_mean": "br",
             "iou_micro": "ioumicro", "f1_micro": "f1micro"}

    thetas: dict[str, float] = {}
    if args.threshold is not None:
        thetas = {"fixed": float(args.threshold)}
    else:
        sweep = Path(args.ckpt).resolve().parent.parent / "sweep.json"
        if not sweep.is_file():
            print("WARNING: no sweep.json — falling back to θ = 0.5, which is NOT "
                  "the operating point any bench scored at.")
            thetas = {"fixed": 0.5}
        else:
            rec = json.loads(sweep.read_text())
            curve = rec.get("sweep", {})
            for c in crits:
                have = {t: v[c] for t, v in curve.items()
                        if c in v and v[c] == v[c]}
                if not have:
                    print(f"  skip {c}: not recorded in {sweep.name} "
                          f"(a buffered_* criterion needs --buffer-px at sweep time)")
                    continue
                best = max(have, key=lambda t: have[t])
                thetas[c] = float(best)
                print(f"  θ*[{c}] = {float(best):<6} ({c}={have[best]:.4f})")
            if not thetas:
                raise SystemExit(f"none of {crits} are recorded in {sweep}")

    # Bands come from the ckpt so the read matches the model's expectation.
    import torch
    hp = torch.load(str(args.ckpt), map_location="cpu",
                    weights_only=False, mmap=True).get("hyper_parameters", {})
    bands = list(hp.get("bands", (1, 2, 3, 4)))
    rs = hp.get("reflectance_scale", 10000.0) or 1.0

    with rasterio.open(img_path) as src:
        img = _read_tile(src, bands, src.width, src.height)
    print(f"tile {args.tile}: {img.shape} raw, bands={bands}, upsampler={hp.get('upsampler')}")

    sr, prob = run_tile(args.ckpt, args.sr_dir, img, args.device,
                        cell_px=args.cell_px)
    preds = {c: (prob > t).astype(np.float32) for c, t in thetas.items()}
    print(f"  SR {sr.shape}  prob {prob.shape}")
    for c, t in thetas.items():
        print(f"    θ={t:<6} [{c:<18}] road frac = {preds[c].mean():.4f}")

    # Does the SR output actually live in the same reflectance range as the
    # input? A joint-finetuned generator is under no obligation to keep its
    # output looking like an image — the task loss only needs it to be USEFUL
    # for segmentation — so a panel that renders as saturated noise is either a
    # genuinely drifted generator or merely a mis-stretched one. These two
    # numbers separate those cases without another GPU run.
    in_ref = img / float(rs)
    p_in = np.percentile(in_ref[:3], (1, 50, 99))
    p_sr = np.percentile(sr[:3], (1, 50, 99))
    print(f"  input  reflectance p1/p50/p99 = {p_in[0]:+.4f} {p_in[1]:+.4f} {p_in[2]:+.4f}")
    print(f"  SR out reflectance p1/p50/p99 = {p_sr[0]:+.4f} {p_sr[1]:+.4f} {p_sr[2]:+.4f}")
    ratio = (p_sr[2] - p_sr[0]) / max(p_in[2] - p_in[0], 1e-9)
    print(f"  SR/input dynamic-range ratio  = {ratio:.2f}"
          + ("   <- SR is off the input's scale; the shared stretch will saturate"
             if (ratio > 3 or ratio < 0.33) else ""))

    ref = sr if args.stretch_from_sr else in_ref
    lo, hi = np.percentile(ref[:3], args.stretch)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tile}" + (f"_{args.tag}" if args.tag else "")

    sr_rgb = to_rgb(sr, lo, hi)
    # ONE SR panel: it does not depend on θ, so emitting it per criterion would
    # be three identical files inviting the reader to look for a difference.
    plt.imsave(out / f"{stem}_sr.png", sr_rgb)
    print(f"  -> {out / f'{stem}_sr.png'}")
    for c, t in thetas.items():
        name = f"{stem}_pred_{SHORT.get(c, c)}.png"
        plt.imsave(out / name, preds[c], cmap="gray", vmin=0, vmax=1)
        print(f"  -> {out / name}   (θ={t}, {c})")

    gt = None
    if not args.no_gt and gt_path.is_file():
        with rasterio.open(gt_path) as g:
            gt = (g.read(1) > 0).astype(np.float32)
        plt.imsave(out / f"{stem}_gt.png", gt, cmap="gray", vmin=0, vmax=1)
        print(f"  -> {out / f'{stem}_gt.png'}")

    if not args.no_pair:
        panels = [("SR input (" + str(hp.get("upsampler")) + ")", sr_rgb, None)]
        panels += [(f"pred θ*={t}\n[{c}]", preds[c], "gray") for c, t in thetas.items()]
        if gt is not None:
            panels.append(("ground truth 2.5 m", gt, "gray"))
        fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 6.2))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, arr, cmap) in zip(axes, panels):
            ax.imshow(arr) if cmap is None else ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.set_axis_off()
        fig.suptitle(f"{args.tile}  [{args.tag or Path(args.ckpt).parent.parent.name}]",
                     y=1.04)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"{stem}_compare.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out / f'{stem}_compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
