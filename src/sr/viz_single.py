"""2x2 single-checkpoint visualisation of one JointSR+UNet model.

Renders, for ONE `JointSRUNetLightning` checkpoint on ONE 128 px crop, the four
panels that tell the whole story of the model's forward pass:

    | Original 10 m image    | Ground-truth 10 m mask |
    | SR image (UNet input)  | Predicted mask         |

Top-left is the raw 10 m crop the model consumes; bottom-left is the exact
super-resolved reflectance tensor its UNet actually saw (this ckpt's own SR net
+ `sr_pad` pad/crop, BEFORE the z-score adapter) — i.e. what the SR module made
of the top-left image. Top-right is the ground truth; bottom-right is
sigmoid(logits) > threshold. Both image panels share ONE percentile stretch
computed from the original crop, so the SR panel's contrast is comparable to the
input rather than autoscaled in isolation.

The checkpoint's forward is replayed faithfully from its saved hparams
(`upsampler`, `sr_pad`, `reflectance_scale`, ...), so padded/unpadded and
SEN2SR/SR4RS/bicubic ckpts all just work. The SR-weights directory defaults to
the right place for the ckpt's upsampler (SR4RS -> models/SR4RS_RGBN, SEN2SR ->
the examples folder); override with --sr-dir.

Zero-argument default (everything from the gitignored src/sr/examples/ folder):

    python -m sr.viz_single                         # -> Durban_r4_c3_single.png

`--pristine` adds a row for the UN-finetuned SR net (the pristine weights the
ckpt was initialised from, loaded fresh from --sr-dir), turning the left column
into: original 10 m -> un-finetuned SR -> the ckpt's finetuned SR. Both SR
panels go through the ckpt's own pad/crop/scale so they are directly comparable;
the un-finetuned row's mask panel is left empty until a frozen-SR ckpt exists:

    python -m sr.viz_single --pristine --device mps

Override any piece:

    python -m sr.viz_single --ckpt <other.ckpt> --image <tile.tif> \
        --row 224 --col 288 --threshold 0.4

Ground truth: `{tile}_mask.tif` beside the image (or --mask). A 10 m mask (dims
== tile) is bicubically upsampled x4 for display; a 2.5 m mask (dims == 4x) is
read at the scaled window. Missing -> empty GT panel with a warning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sr.model import JointSRUNetLightning
from sr.viz_grid import CROP, EXAMPLES_DIR, read_gt, read_patch, to_rgb

# Zero-arg defaults, resolved inside --examples-dir.
DEFAULT_CKPT = "unet_s2rosa_sr4rs_best.ckpt"
DEFAULT_IMAGE = "AzonalVegetation_UrbanCore_Road_-3379_2543_r4_c3.tif"

# SR-weights directory per upsampler, when --sr-dir is not given. SEN2SR keeps
# its weights in the examples folder (model.safetensor + hard_constraint.safetensor);
# SR4RS's extracted generator lives under models/ (see scripts/sr4rs/extract_sr4rs.py).
SR4RS_DIR = "models/SR4RS_RGBN"


def peek_upsampler(ckpt: Path) -> str:
    """Cheap read of the ckpt's `upsampler` hparam (mmap avoids pulling the
    400 MB state_dict off disk) so we can pick the default SR-weights dir."""
    hp = torch.load(str(ckpt), map_location="cpu", weights_only=False,
                    mmap=True).get("hyper_parameters", {})
    return hp.get("upsampler", "sen2sr")


def default_sr_dir(upsampler: str, examples: Path) -> Path:
    return Path(SR4RS_DIR) if upsampler == "sr4rs" else examples


def load_pristine_sr(upsampler: str, sr_dir: Path, sr_pad: int):
    """The UN-finetuned SR net: the exact module the ckpt was initialised from
    (loaded fresh from `sr_dir`, before joint fine-tuning drifted its weights).
    Mirrors JointSRUNetLightning.__init__'s per-upsampler construction, incl.
    the sr_pad low-pass-mask grow for SEN2SR."""
    if upsampler == "sr4rs":
        from sr.sr4rs_torch import load_trainable_sr4rs
        return load_trainable_sr4rs(sr_dir)
    if upsampler in ("sen2sr", "sen2sr_full"):
        from sr.sen2sr_loader import (
            load_trainable_sen2sr, load_trainable_sen2sr_full, pad_low_pass_mask)
        sr = (load_trainable_sen2sr(str(sr_dir)) if upsampler == "sen2sr"
              else load_trainable_sen2sr_full(str(sr_dir)))
        if sr_pad > 0:
            pad_low_pass_mask(sr, sr_pad)
        return sr
    if upsampler == "bicubic":
        from sr.sen2sr_loader import BicubicUpsampler
        return BicubicUpsampler(4)   # parameter-free: pristine == finetuned
    raise ValueError(f"no pristine loader for upsampler {upsampler!r}")


@torch.no_grad()
def run_checkpoint(ckpt: Path, sr_dir: Path, x, threshold, device, pristine=False):
    """Replay one checkpoint's forward. Returns (unet_input_chw, pred_hw,
    pristine_sr_chw). `pristine_sr_chw` is None unless `pristine` is set.

    unet_input = the reflectance tensor entering the z-score adapter (this
    ckpt's own FINE-TUNED SR net + sr_pad pad/crop) — exactly what its UNet saw,
    up to normalisation. pristine_sr = the same crop through the UN-finetuned SR
    net (fresh from `sr_dir`), run through the IDENTICAL pad/crop/scale math so
    the two SR panels are directly comparable.
    """
    # map_location="cpu", then .to(device): deserialising straight onto MPS
    # puts the WHOLE ckpt there — including ~300 MB of Adam moments viz never
    # uses. Via CPU, only the model weights land on the device.
    model = JointSRUNetLightning.load_from_checkpoint(
        str(ckpt), map_location="cpu", sen2sr_dir=str(sr_dir)).eval().to(device)
    # `reflectance_scale` maps the dataloader's raw values to the 0-1 reflectance
    # the SR nets expect: 10000.0 for DN COGs, 1.0 for the ROSA V2 datasets whose
    # COGs already store 0-1 reflectance (see sr/configs/joint_sr.yaml). Some
    # s2rosa ckpts saved it as None; forward divides by it, so pin None -> 1.0.
    if getattr(model.hparams, "reflectance_scale", 10000.0) is None:
        model.hparams.reflectance_scale = 1.0
    rs = float(model.hparams.reflectance_scale)
    p = int(model.hparams.sr_pad)
    up = int(model.hparams.upscale)

    t = torch.from_numpy(x)[None].float().to(device)   # RAW values, as the dataloader feeds
    t_ref = t / rs

    def sr_img(sr_module):
        """One SR net's output for this crop, via the ckpt's own pad/crop."""
        t_sr = torch.nn.functional.pad(t_ref, (p,) * 4, mode="reflect") if p else t_ref
        hr = sr_module(t_sr)
        if p:
            q = p * up
            hr = hr[..., q:-q, q:-q]
        return hr[0].cpu().numpy()

    finetuned = sr_img(model.sr)
    logits = model(t)  # faithful end-to-end (forward divides by its own scale)
    pred = (torch.sigmoid(logits)[0, 0] > threshold).float().cpu().numpy()

    pristine_sr = None
    if pristine:
        pmod = load_pristine_sr(model.hparams.upsampler, sr_dir, p).eval().to(device)
        pristine_sr = sr_img(pmod)
        d = float(np.abs(pristine_sr - finetuned).mean())
        print(f"mean|finetuned - un-finetuned SR| = {d:.4g} (reflectance)")
    return finetuned, pred, pristine_sr


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", default=EXAMPLES_DIR,
                    help="folder holding the ckpt, tile and SEN2SR weights — "
                         "every unspecified path defaults from here")
    ap.add_argument("--ckpt", default=None,
                    help=f"JointSR+UNet checkpoint (default: {DEFAULT_CKPT})")
    ap.add_argument("--image", default=None,
                    help=f"V2 tile GeoTIFF (default: {DEFAULT_IMAGE})")
    ap.add_argument("--mask", default=None,
                    help="explicit GT raster (default: {tile}_mask.tif beside the image)")
    ap.add_argument("--sr-dir", default=None,
                    help="SR-net weights dir (default: per the ckpt's upsampler — "
                         f"{SR4RS_DIR} for sr4rs, --examples-dir for sen2sr)")
    ap.add_argument("--row", type=int, default=0, help="top of the 128 px crop")
    ap.add_argument("--col", type=int, default=0, help="left of the 128 px crop")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98),
                    metavar=("PLO", "PHI"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pristine", action="store_true",
                    help="add an UN-finetuned SR row (pristine weights from "
                         "--sr-dir) between the original and the ckpt's SR — "
                         "column becomes: original 10 m / un-finetuned SR / "
                         "finetuned SR. Its mask panel is left empty (no "
                         "frozen-SR ckpt yet).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = Path(args.examples_dir)
    ckpt = Path(args.ckpt) if args.ckpt else examples / DEFAULT_CKPT
    image = Path(args.image) if args.image else examples / DEFAULT_IMAGE
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if not image.exists():
        raise SystemExit(f"image not found: {image}")
    upsampler = peek_upsampler(ckpt)
    sr_dir = Path(args.sr_dir) if args.sr_dir else default_sr_dir(upsampler, examples)
    print(f"ckpt: {ckpt.name}  (upsampler={upsampler}, sr_dir={sr_dir})")

    x = read_patch(image, args.row, args.col)
    gt, gt_label = read_gt(image, args.mask, args.row, args.col)
    lo, hi = np.percentile(x[:3], args.stretch)
    hi = max(hi, lo + 1e-6)
    unet_in, pred, pristine = run_checkpoint(
        ckpt, sr_dir, x, args.threshold, args.device, pristine=args.pristine)

    hr_px = f"{CROP * 4}px, 2.5 m"
    if pristine is None:
        # original 2x2: original / GT ; finetuned SR / prediction
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.6))
        axes[0, 0].imshow(to_rgb(x, lo, hi), interpolation="nearest")
        axes[0, 0].set_title(f"Original 10 m image ({CROP}px)", fontsize=11)
        axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title(gt_label, fontsize=11)
        axes[1, 0].imshow(to_rgb(unet_in, lo, hi))
        axes[1, 0].set_title(f"SR image -> UNet input ({hr_px})", fontsize=11)
        axes[1, 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[1, 1].set_title(f"Predicted mask (road frac {pred.mean():.3f})", fontsize=11)
    else:
        # 3x2: original / GT ; un-finetuned SR / (empty) ; finetuned SR / pred.
        # Left column is the SR progression the user asked for; right column
        # keeps the matching masks (the un-finetuned row has no ckpt -> empty).
        fig, axes = plt.subplots(3, 2, figsize=(8.0, 12.6))
        axes[0, 0].imshow(to_rgb(x, lo, hi), interpolation="nearest")
        axes[0, 0].set_title(f"Original 10 m image ({CROP}px)", fontsize=11)
        axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title(gt_label, fontsize=11)
        axes[1, 0].imshow(to_rgb(pristine, lo, hi))
        axes[1, 0].set_title(f"Un-finetuned SR ({hr_px})", fontsize=11)
        axes[1, 1].text(0.5, 0.5, "no frozen-SR ckpt yet",
                        ha="center", va="center", fontsize=11, color="0.5",
                        transform=axes[1, 1].transAxes)
        axes[2, 0].imshow(to_rgb(unet_in, lo, hi))
        axes[2, 0].set_title(f"Finetuned SR -> UNet input ({hr_px})", fontsize=11)
        axes[2, 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2, 1].set_title(f"Predicted mask (road frac {pred.mean():.3f})", fontsize=11)

    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle(f"{ckpt.name}\n{image.stem}  crop r{args.row} c{args.col}  "
                 f"(shared {args.stretch[0]:g}-{args.stretch[1]:g}% stretch; "
                 f"threshold {args.threshold})", fontsize=10)
    fig.tight_layout()
    out = args.out or f"{image.stem}_single.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}  (pred road fraction {pred.mean():.4f})")


if __name__ == "__main__":
    main()
