"""SR-evolution grid: one panel per fine-tuning snapshot, 10 per row.

Sibling of `viz_single` / `viz_grid`, reading the SR-weights-only frames the
training run drops into `<run dir>/sr_snapshots/` (`SR_SNAPSHOT_EVERY` in
scripts/{hpc,LightningStudio}/sr/_stages.sh -> `model.sr_snapshot_every`).
Every frame is the SAME crop through the SAME SR architecture with that
epoch's weights, so reading the grid left-to-right, top-to-bottom replays how
the task loss reshaped the SR net over the refit:

    header  | Original 10 m (nearest x4) | Bicubic x4 | Un-finetuned SR | GT mask |
    row 1   | e1 | e3 | e5 | e7 | e9 | e11 | e13 | e15 | e17 | e19 |
    row 2   | e21 | ...

The header row is the fixed frame of reference (`--no-header` drops it and
tiles every frame instead): the raw input, the parameter-free baseline, the
epoch-0 `_init` snapshot (the pretrained SR before any task gradient) and the
native 2.5 m ground truth, so "did fine-tuning sharpen the roads" is judged
against both the un-enhanced input and the un-finetuned SR. All image panels
share ONE percentile stretch computed from the original crop, so a panel
getting brighter/sharper is the SR drifting, not an autoscale artefact.

THAT DEFAULT HAS A FAILURE MODE, AND `--stretch-mode` IS THE ESCAPE HATCH.
A generator adapted hard toward the segmentation loss is under no obligation to
keep its output in the input's reflectance range — it often learns very large
contrasts. Once its values leave [lo, hi], the shared stretch clips every pixel
to 0 or 1 and the panel renders as a flat block: the frame is fine, the mapping
is saturated. The per-frame log line and the panel title now both report the
CLIPPED FRACTION, so that case reads as "clip 97%" instead of looking like a
broken tile. Three modes:

  shared     (default) percentiles from the ORIGINAL 10 m crop, applied to
             every panel. Brightness is comparable across frames; a drifted
             frame saturates, and says so.
  global     percentiles pooled over the original crop AND every frame. Still
             ONE mapping — so frames stay comparable — but wide enough that
             nothing clips. Costs the frames being held in memory at once.
  per-frame  each SR frame autoscaled to its own percentiles. Always shows the
             structure; brightness is NOT comparable between frames, so a
             frame looking sharper may only mean its histogram moved. The
             header panels keep the original stretch, as the fixed reference.

A PERCENTILE PAIR ALWAYS CLIPS SOMETHING — 2-98 DISCARDS 4% BY DEFINITION.
"Don't clip, just cut the long tails" is therefore a choice of `--stretch`, not
a separate mode: widen the pair until only the outliers fall outside it. On a
drifted generator the pairing to reach for is

    --stretch-mode global --stretch 0.5 99.5

which on the ls1e-4 grid arm leaves 0.6-1.2% clipped (against 63-72% for the
default shared 2-98) while keeping ONE mapping across every frame, so a real
brightness difference between runs still reads as one. Go to 0.1/99.9 if even
that is too aggressive — the cost is contrast, since a handful of outlying
pixels then set the range the bulk of the image has to share.

Snapshots hold weights only — no UNet — so there are no mask predictions here;
for those use `viz_single`/`viz_grid` on a real checkpoint. Each panel is
titled with its epoch and the `sr_drift_rel` recorded in the frame (relative
L2 distance of the trainable SR params from their pretrained values).

Geometry comes from the snapshot itself. For SEN2SR the frozen FFT
`HardConstraint` mask pins the accepted input size: its side / `upscale` is
`crop + 2*sr_pad`, so with the conventional 128 px crop the run's `sr_pad`
(8 for r2a) is read straight off the mask — no need to remember the recipe.
SR4RS has no such mask, so its crop is free (`--crop`) and `--sr-pad` defaults
to 0 (r4b). The SR-weights dir defaults per upsampler exactly as in
`viz_single`.

`--device` defaults to `auto`: MPS when this crop's estimated peak activation
footprint fits half of Metal's recommended working set, CPU otherwise. Because
this runs on a laptop rather than the cluster, the Metal allocator is also
capped below physical memory (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`), each
frame's buffers are released as it is drawn, and an allocation failure demotes
the run to CPU mid-flight — so an over-large crop gets a slow render or a clean
error, never a swapping machine.

Zero-argument default — r2a's snapshots, from the gitignored examples folder:

    python -m sr.viz_sr                    # -> Skukuza_r2_c2_r2a_snapshots.png

Same tool for the R4b (SR4RS) evolution, or any other run's frames:

    python -m sr.viz_sr --snapshot-dir src/sr/examples/r4b_snapshots
    python -m sr.viz_sr --snapshot-dir <run>/sr_snapshots --every 2 --per-row 8

Ground truth for the header panel follows `viz_grid`'s rules (native 2.5 m
`{tile}_mask_high.tif`, else rasterised from the tile's road graph, else the
10 m mask bicubic x4); `--no-header` skips it entirely.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np

# --- MPS guard rails, set BEFORE torch initialises its Metal allocator -------
# This is a laptop, not the cluster: an SR net that over-allocates unified
# memory does not politely OOM, it drags the whole machine into swap. The high
# watermark caps the allocator at 0.7x the recommended working set (torch's own
# default is 1.7x, i.e. deliberately allowed to exceed physical RAM), so a bad
# run raises a catchable "MPS backend out of memory" and `sr_evolution` finishes
# on CPU instead. The low watermark (where cached blocks start being handed
# back) must stay <= the high one or torch rejects the pair outright, so it is
# lowered to match. The fallback var keeps any op Metal lacks on the CPU path
# rather than aborting the render. All `setdefault`, so an explicit environment
# always wins.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (must follow the env guards above)

from sr.model import REFLECTANCE_SCALE
from sr.viz_grid import (
    CROP, EXAMPLES_DIR, MASK_DATASET_DIR, nearest_x4, read_gt, read_patch,
    to_rgb)
from sr.viz_single import SR4RS_DIR, default_sr_dir, load_pristine_sr

# Zero-arg defaults, resolved inside --examples-dir.
DEFAULT_SNAPSHOT_DIR = "r2a_snapshots"
DEFAULT_IMAGE = "Durban_r4_c3.tif"   # same tile as viz_single/viz_grid

PER_ROW = 10
EPOCH_RE = re.compile(r"epoch_(\d+)")

# Widest feature map each SR net carries at the HR grid, used to size a run
# before it is launched (see `resolve_device`). SR4RS runs 256-channel convs at
# 4x — the reason `r4b` needs batch size 1-4 even on a 44 GB cluster GPU.
PEAK_CHANNELS = {"sen2sr": 48, "sen2sr_full": 64, "sr4rs": 256, "bicubic": 4}
LIVE_TENSORS = 8        # conservative count of HR maps alive at once
MPS_BUDGET = 0.5        # fraction of the Metal working set we let a run plan for


# ------------------------------------------------------------------ snapshots
def find_snapshots(snapshot_dir: Path):
    """The run's frames as (epoch, is_init, path), in training order.

    `_snapshot_sr` names them `epoch_{NNN}[_init].pt`; sorting on the parsed
    epoch (not the filename) keeps 3-digit and any future 4-digit runs ordered,
    with the epoch-0 `_init` frame first."""
    snaps = []
    for p in sorted(snapshot_dir.glob("*.pt")):
        m = EPOCH_RE.search(p.name)
        if m:
            snaps.append((int(m.group(1)), "_init" in p.name, p))
    if not snaps:
        raise SystemExit(f"no epoch_*.pt snapshots in {snapshot_dir}")
    return sorted(snaps, key=lambda s: (s[0], not s[1]))


def load_snapshot(path: Path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def resolve_geometry(state_dict, upsampler, crop_arg, pad_arg, upscale=4):
    """(crop, sr_pad) for these snapshots.

    SEN2SR's `hard_constraint.low_pass_mask` is a fixed FFT mask sized for one
    exact input: `mask_side / upscale == crop + 2*sr_pad` (the training run grew
    it via `pad_low_pass_mask`). One of the two is therefore implied by the
    other, and with both left at their defaults we recover the run's own
    padding. SR4RS is fully convolutional — nothing to infer, so the CLI
    values (128 / 0) stand."""
    mask = state_dict.get("hard_constraint.low_pass_mask")
    if mask is None:
        return (crop_arg or CROP), (pad_arg or 0)

    lr = mask.shape[-1] // upscale               # crop + 2*pad, in native px
    if crop_arg and pad_arg is not None:
        if crop_arg + 2 * pad_arg != lr:
            raise SystemExit(
                f"--crop {crop_arg} + 2*--sr-pad {pad_arg} != {lr}, the input "
                f"size baked into this snapshot's {mask.shape[-1]}px FFT mask.")
        return crop_arg, pad_arg
    if pad_arg is not None:
        crop = lr - 2 * pad_arg
        if crop <= 0:
            raise SystemExit(f"--sr-pad {pad_arg} leaves no crop ({lr}px input)")
        return crop, pad_arg
    crop = crop_arg or CROP
    pad, rem = divmod(lr - crop, 2)
    if pad < 0 or rem:
        raise SystemExit(
            f"crop {crop} does not fit this snapshot's FFT mask: it accepts "
            f"{lr}px inputs, so --crop must be {lr} - 2*sr_pad (an even "
            f"difference). Pass --crop {lr} for the unpadded size.")
    return crop, pad


def align_low_pass_mask(module, state_dict):
    """Give the module the snapshot's OWN FFT mask buffer.

    `load_pristine_sr` grows the shipped 512px mask by the `sr_pad` we pass it,
    which reproduces the training run's mask for the run's own crop — but not
    for a different `--crop`/`--sr-pad` split of the same input size (e.g.
    reading r2a's 576px mask as crop 144 / pad 0), where `load_state_dict`
    would then hit a shape mismatch. The mask is a buffer *inside the
    snapshot*, so the fix is to size the buffer to it and let the load supply
    the values: the constraint applied is then literally the one that epoch
    trained under, never a re-derived approximation."""
    m = state_dict.get("hard_constraint.low_pass_mask")
    hc = getattr(module, "hard_constraint", None)
    if m is None or hc is None or hc.low_pass_mask.shape == m.shape:
        return module
    ref = hc.low_pass_mask
    del hc.low_pass_mask
    hc.register_buffer("low_pass_mask",
                       torch.empty(m.shape, dtype=ref.dtype, device=ref.device))
    return module


# --------------------------------------------------------------------- device
def peak_bytes(upsampler, crop, pad, upscale=4):
    """Rough peak activation footprint of one forward, in bytes.

    Deliberately an OVER-estimate (widest layer x `LIVE_TENSORS` maps at the HR
    grid): the point is to refuse a run that would exhaust unified memory, and
    being pessimistic costs at most a slower CPU render."""
    hr = (crop + 2 * pad) * upscale
    return PEAK_CHANNELS.get(upsampler, 256) * hr * hr * 4 * LIVE_TENSORS


def release_mps():
    """Hand this frame's Metal buffers back. Without it the cache grows across
    a 50-frame run until the allocator (or the OS) starts thrashing."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def resolve_device(arg, upsampler, crop, pad, upscale=4):
    """Pick the device for `--device auto`, and sanity-check an explicit one.

    "auto" = MPS when this crop's estimated peak fits inside `MPS_BUDGET` of
    Metal's recommended working set, else CPU. On a small unified-memory Mac
    that keeps the light SEN2SR grids on the GPU while sending, say, SR4RS at a
    256 px crop (~8 GB of 4x feature maps) to the CPU instead of into swap.
    An explicit `--device mps` is honoured — it is the user's call — but is
    still sized, warned about, and backed by the high-watermark cap plus
    `sr_evolution`'s CPU fallback."""
    want = torch.device(arg if arg != "auto" else "cpu")
    need = peak_bytes(upsampler, crop, pad, upscale)
    if arg == "auto":
        if not torch.backends.mps.is_available():
            return torch.device("cpu")
        budget = torch.mps.recommended_max_memory() * MPS_BUDGET
        if need > budget:
            print(f"device: cpu — {upsampler} at crop {crop} needs ~"
                  f"{need / 2**30:.1f} GB, over the {budget / 2**30:.1f} GB "
                  f"MPS budget")
            return torch.device("cpu")
        print(f"device: mps (~{need / 2**30:.1f} GB estimated peak, budget "
              f"{budget / 2**30:.1f} GB)")
        return torch.device("mps")

    if want.type == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps: no Metal device available")
        rec = torch.mps.recommended_max_memory()
        if need > rec * MPS_BUDGET:
            print(f"WARN: {upsampler} at crop {crop} needs ~{need / 2**30:.1f} "
                  f"GB, over half the {rec / 2**30:.1f} GB Metal working set — "
                  "capped at 0.7x by PYTORCH_MPS_HIGH_WATERMARK_RATIO, and it "
                  "will finish on CPU if that cap trips. Use --crop to shrink "
                  "it, or --device cpu.")
    return want


# ---------------------------------------------------------------- SR frames
def snapshot_hc(snapshot):
    """(sr_hc, hc_mask) for `load_pristine_sr`, read off one snapshot.

    Snapshots written since the HC 2x2 record the RESOLVED `sr_hc`; older ones
    do not, so fall back to the state dict itself, which is self-describing —
    the mask is a persistent buffer, so its presence IS the constraint."""
    sd = snapshot["sr_state_dict"]
    mask = sd.get("hard_constraint.low_pass_mask")
    return snapshot.get("sr_hc") or ("on" if mask is not None else "off"), mask


@torch.no_grad()
def sr_evolution(snaps, upsampler, sr_dir, x, tile_scale, pad, device,
                 upscale=4, sr_hc=None, hc_mask=None):
    """Yield (epoch, is_init, drift, sr_chw) for each snapshot, in order.

    ONE SR module is built (from the pristine weights, so buffers and any
    non-trainable pieces are right) and each frame's `sr_state_dict` is loaded
    into it in place — the architecture is identical across frames, only the
    weights move. The crop goes in as 0-1 reflectance and comes back through the
    run's own reflect-pad/crop, so every panel is the exact tensor that epoch's
    UNet would have consumed (before the z-score adapter).

    A generator, not a list: 50+ frames of (4, 512, 512) float32 is ~200 MB, and
    the caller only needs one at a time to draw its panel.

    On MPS each frame's buffers are released as soon as its array is on the host
    (a 50-frame run otherwise grows the Metal cache monotonically), and the
    first allocation failure demotes the whole run to CPU rather than leaving
    the machine swapping — slower, but it finishes."""
    device = torch.device(device)
    if sr_hc is None:
        sr_hc, hc_mask = snapshot_hc(load_snapshot(snaps[0][2]))
    module = load_pristine_sr(upsampler, Path(sr_dir), pad, sr_hc=sr_hc,
                              hc_mask=hc_mask).eval().to(device)
    reflectance = torch.from_numpy(x)[None].float().to(device) / tile_scale
    t_in = (torch.nn.functional.pad(reflectance, (pad,) * 4, mode="reflect")
            if pad else reflectance)
    q = pad * upscale
    for epoch, is_init, path in snaps:
        snap = load_snapshot(path)
        # load_state_dict copies in place, so weights stay on `device`.
        align_low_pass_mask(module, snap["sr_state_dict"])
        module.load_state_dict(snap["sr_state_dict"])
        try:
            hr = module(t_in)
        except RuntimeError as err:
            if device.type == "cpu":
                raise
            print(f"WARN: {device.type} forward failed ({str(err).splitlines()[0]})"
                  " — finishing the run on CPU.")
            device = torch.device("cpu")
            module, t_in = module.to(device), t_in.to(device)
            release_mps()
            hr = module(t_in)
        if q:
            hr = hr[..., q:-q, q:-q]
        out = hr[0].cpu().numpy()
        del hr
        release_mps()
        yield epoch, is_init, snap.get("sr_drift_rel"), out


def clip_frac(x, lo, hi) -> float:
    """Fraction of the RGB pixels this stretch maps outside [0, 1].

    The number that separates "the SR output is a flat grey block" from "the
    stretch is saturated": at 0.97 the panel carries essentially no information
    and needs `--stretch-mode global` or `per-frame` to be readable at all."""
    rgb = np.asarray(x)[:3]
    return float(((rgb < lo) | (rgb > hi)).mean())


def rgb_u8(x, lo, hi):
    """Panel RGB as uint8. The grid holds every frame's array until savefig;
    at 8 bits that is ~0.8 MB a panel instead of ~6 MB, and the quantisation is
    invisible against a percentile stretch that already clips to 0-1."""
    return (to_rgb(x, lo, hi) * 255).round().astype(np.uint8)


# ------------------------------------------------------------------ rendering
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", default=EXAMPLES_DIR,
                    help="folder holding the snapshots, tile and SR weights — "
                         "every unspecified path defaults from here")
    ap.add_argument("--snapshot-dir", default=None,
                    help="folder of epoch_*.pt SR snapshots (default: "
                         f"--examples-dir/{DEFAULT_SNAPSHOT_DIR}; a training "
                         "run's own sr_snapshots/ works as-is)")
    ap.add_argument("--image", default=None,
                    help=f"V2 tile GeoTIFF (default: {DEFAULT_IMAGE})")
    ap.add_argument("--mask", default=None,
                    help="explicit GT raster for the header panel, read as-is "
                         "(overrides the native-HR logic)")
    ap.add_argument("--graph", default=None,
                    help="road-graph parquet to rasterise the native 2.5 m GT "
                         "from (default: {tile}_graph.parquet beside the image, "
                         "else --dataset-dir's masks_graph/{tile}.parquet)")
    ap.add_argument("--dataset-dir", default=MASK_DATASET_DIR,
                    help="ROSA dataset root holding {split}/masks_graph/ used to "
                         "find a tile's road-graph parquet for the native 2.5 m GT")
    ap.add_argument("--sr-dir", default=None,
                    help="SR-net weights dir the snapshots are loaded into "
                         f"(default: per the snapshot's upsampler — {SR4RS_DIR} "
                         "for sr4rs, --examples-dir for sen2sr)")
    ap.add_argument("--row", type=int, default=None,
                    help="top of the crop (default: centred in the tile)")
    ap.add_argument("--col", type=int, default=None,
                    help="left of the crop (default: centred in the tile)")
    ap.add_argument("--crop", type=int, default=None,
                    help=f"native-px crop side (default: {CROP}; SEN2SR's FFT "
                         "mask pins this, SR4RS accepts any size the box fits)")
    ap.add_argument("--sr-pad", type=int, default=None,
                    help="reflect-pad in native px (default: read off SEN2SR's "
                         "FFT mask — the run's own sr_pad — else 0)")
    ap.add_argument("--per-row", type=int, default=PER_ROW,
                    help=f"snapshots per row (default: {PER_ROW})")
    ap.add_argument("--every", type=int, default=1, metavar="N",
                    help="keep every Nth snapshot after the init frame, to thin "
                         "a long run down to a readable grid (default: all)")
    ap.add_argument("--no-header", action="store_true",
                    help="drop the reference row (original / bicubic / "
                         "un-finetuned SR / GT) and tile every frame, init "
                         "included, --per-row to a row")
    ap.add_argument("--stretch-mode", default="shared",
                    choices=("shared", "global", "per-frame"),
                    help="shared (default) = percentiles from the ORIGINAL crop, "
                         "applied to every panel (comparable, but a drifted frame "
                         "saturates); global = percentiles pooled over the original "
                         "AND every frame (one mapping, nothing clips, frames held "
                         "in memory); per-frame = each frame autoscaled (always "
                         "readable, brightness NOT comparable)")
    ap.add_argument("--clip-warn", type=float, default=0.02,
                    help="annotate a panel's title with its clipped fraction once "
                         "it exceeds this (default 0.02)")
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98),
                    metavar=("PLO", "PHI"))
    ap.add_argument("--device", default="auto",
                    help="auto (default) = MPS when this crop's estimated peak "
                         "fits half the Metal working set, else CPU; or force "
                         "cpu/mps/cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = Path(args.examples_dir)
    snapshot_dir = (Path(args.snapshot_dir) if args.snapshot_dir
                    else examples / DEFAULT_SNAPSHOT_DIR)
    image = Path(args.image) if args.image else examples / DEFAULT_IMAGE
    if not snapshot_dir.is_dir():
        raise SystemExit(f"snapshot dir not found: {snapshot_dir}")
    if not image.exists():
        raise SystemExit(f"image not found: {image}")

    snaps = find_snapshots(snapshot_dir)
    head = load_snapshot(snaps[0][2])
    upsampler = head.get("upsampler", "sen2sr")
    sr_dir = Path(args.sr_dir) if args.sr_dir else default_sr_dir(upsampler, examples)
    crop, pad = resolve_geometry(head["sr_state_dict"], upsampler,
                                 args.crop, args.sr_pad)
    # Which module architecture to replay these weights into (HC 2x2): r4a
    # snapshots carry sr_model./hard_constraint. prefixes on a SR4RS generator,
    # r2b snapshots carry no hard_constraint. keys at all.
    sr_hc, hc_mask = snapshot_hc(head)
    device = resolve_device(args.device, upsampler, crop, pad)
    print(f"{len(snaps)} snapshots in {snapshot_dir.name}  "
          f"(upsampler={upsampler}, sr_hc={sr_hc}, crop={crop}, sr_pad={pad}, "
          f"sr_dir={sr_dir})")

    # The init frame is the un-finetuned baseline: it belongs in the header row
    # rather than the evolution, which also makes the frame count divide evenly
    # (a run of 1 init + 50 epochs tiles as exactly 5 rows of 10).
    init = None
    if not args.no_header and snaps[0][1]:
        init = snaps[0]
        snaps = snaps[1:]
    if args.every > 1:
        snaps = snaps[::args.every]

    # Centre the crop in the tile unless an explicit row/col was given (matches
    # viz_single/viz_grid), so the default panel shows the middle of the scene.
    import rasterio
    with rasterio.open(image) as src:
        h, w = src.height, src.width
    row = args.row if args.row is not None else max(0, (h - crop) // 2)
    col = args.col if args.col is not None else max(0, (w - crop) // 2)

    x = read_patch(image, row, col, crop=crop)
    lo, hi = np.percentile(x[:3], args.stretch)
    hi = max(hi, lo + 1e-6)
    # Storage units of THIS tile (V2 COGs already hold 0-1 reflectance -> 1.0;
    # DN COGs hold reflectance*10000); the SR nets want 0-1 either way.
    tile_scale = 1.0 if float(np.nanmax(x)) <= 1.5 else REFLECTANCE_SCALE

    per_row = max(1, args.per_row)
    rows = math.ceil(len(snaps) / per_row) + (0 if args.no_header else 1)
    fig, axes = plt.subplots(rows, per_row, squeeze=False,
                             figsize=(1.65 * per_row, 1.85 * rows))
    for ax in axes.ravel():
        ax.set_axis_off()

    first_row = 0
    sr0 = None      # un-finetuned frame, the baseline the per-frame stats use
    if not args.no_header:
        first_row = 1
        gt, gt_label = read_gt(image, args.mask, row, col, graph_path=args.graph,
                               dataset_dir=args.dataset_dir, crop=crop)
        bicubic = torch.nn.functional.interpolate(
            torch.from_numpy(x)[None].float(), scale_factor=4, mode="bicubic",
            align_corners=False)[0].numpy()
        panels = [(rgb_u8(nearest_x4(x), lo, hi), "Original 10 m (nearest x4)",
                   {"interpolation": "nearest"}),
                  (rgb_u8(bicubic, lo, hi), "Bicubic x4", {})]
        if init is not None:
            _, _, _, sr0 = next(sr_evolution([init], upsampler, sr_dir, x,
                                             tile_scale, pad, device,
                                             sr_hc=sr_hc, hc_mask=hc_mask))
            panels.append((rgb_u8(sr0, lo, hi),
                           f"Un-finetuned SR (e{init[0]} init)", {}))
        panels.append((gt, gt_label, {"cmap": "gray", "vmin": 0, "vmax": 1}))
        # zip truncates: a narrow --per-row simply drops the rightmost reference
        # panels rather than overrunning the row.
        for ax, (img, title, kw) in zip(axes[0], panels):
            ax.imshow(img, **kw)
            ax.set_title(title, fontsize=7)
        if len(panels) > per_row:
            print(f"NOTE: --per-row {per_row} fits {per_row} of the "
                  f"{len(panels)} reference panels; "
                  f"dropped {', '.join(p[1] for p in panels[per_row:])}")

    frames = sr_evolution(snaps, upsampler, sr_dir, x, tile_scale, pad, device,
                          sr_hc=sr_hc, hc_mask=hc_mask)
    if args.stretch_mode == "global":
        # The only mode that needs every frame before drawing any of them: the
        # mapping is pooled over all of them. `--every` is what keeps this
        # affordable (one 128px crop -> 512px x 4 bands is ~4 MB a frame).
        frames = list(frames)
        pool = np.concatenate([x[:3].ravel()]
                              + [f[3][:3].ravel() for f in frames])
        lo, hi = np.percentile(pool, args.stretch)
        hi = max(hi, lo + 1e-6)
        print(f"  global stretch over {len(frames)} frames + the original: "
              f"[{lo:.4f}, {hi:.4f}]")

    for i, (epoch, is_init, drift, sr) in enumerate(frames):
        ax = axes[first_row + i // per_row, i % per_row]
        # In per-frame mode ONLY the SR frames autoscale; the header panels stay
        # on the original crop's stretch, because they are the fixed reference
        # the frames are being judged against.
        if args.stretch_mode == "per-frame":
            f_lo, f_hi = np.percentile(np.asarray(sr)[:3], args.stretch)
            f_hi = max(f_hi, f_lo + 1e-6)
        else:
            f_lo, f_hi = lo, hi
        clipped = clip_frac(sr, f_lo, f_hi)
        ax.imshow(rgb_u8(sr, f_lo, f_hi))
        tag = f"e{epoch}" + (" init" if is_init else "")
        title = tag + (f"  d={drift:.3f}" if drift is not None else "")
        # A saturated panel must announce itself, or it reads as a broken tile.
        if clipped > args.clip_warn:
            title += f"\nclip {clipped:.0%}"
        ax.set_title(title, fontsize=7,
                     color=("#b3261e" if clipped > 0.5 else "black"))
        # Per-frame numbers for the panel you are looking at. NOT the mean: under
        # SEN2SR's hard constraint the low frequencies (incl. DC) are pinned to
        # the LR input, so the mean is constant by construction and only the
        # contrast/high-frequency content moves. `d_init` is how far this frame's
        # IMAGE has travelled from the un-finetuned one, in reflectance.
        if sr0 is None:
            sr0 = sr
        rgb = np.asarray(sr)[:3]
        print(f"  {tag:10s} drift={drift if drift is None else round(drift, 4)}"
              f"  std={sr.std():.4f}  d_init={np.abs(sr - sr0).mean():.4g}"
              f"  range=[{rgb.min():+.3f},{rgb.max():+.3f}]  clip={clipped:.1%}")

    fig.suptitle(
        f"SR evolution — {snapshot_dir.name} ({upsampler}, sr_pad {pad})\n"
        f"{image.stem}  crop r{row} c{col} {crop}px -> {crop * 4}px, 2.5 m  "
        f"({args.stretch_mode} {args.stretch[0]:g}-{args.stretch[1]:g}% stretch; "
        f"d = sr_drift_rel)", fontsize=9)
    fig.tight_layout()
    out = args.out or f"{image.stem}_{snapshot_dir.name}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}  ({len(snaps)} frames, {per_row}/row)")


if __name__ == "__main__":
    main()
