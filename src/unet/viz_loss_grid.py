"""One example patch through the WHOLE loss-ablation arm plan, in a fixed grid.

A deliberately dumb sibling of ``viz_grid``: no dataset plumbing, no --runs
scanning, no --expect lists. It renders the bundled example tile (cell 0 RGB,
cell 1 ground truth) through every loss-ablation checkpoint, laid out in the
fixed arm order l1 … l10, la0, la1. Each checkpoint reports its own loss arm
(stored as an hparam), so files are slotted by what they ARE, not by filename —
``topo_f1.ckpt`` lands in l3 because its arm is ``tl_ce``. An arm with no
checkpoint on disk stays a BLANK cell, so the layout is stable as runs land.

Each prediction is binarized at that arm's own val-tuned θ* (``best_threshold``
from its sibling ``train_meta.json``; 0.5 if none), so arms with very different
operating points — e.g. focal_tversky at θ*≈0.95 — render at their real setting.
``--threshold`` forces one global θ; ``--prob`` shows raw probability instead.
Checkpoints are found recursively, so both a flat ``*.ckpt`` folder and the
``<name>/<name>.ckpt`` + ``train_meta.json`` run layout work.

Zero args does the whole thing:

    python -m unet.viz_loss_grid                    # -> loss_grid.png

Knobs (all optional): --ckpt-dir, --image, --quadrant 0-3 (256px crop instead
of the full 512 tile), --prob (probability heatmaps), --overlay (red on RGB),
--threshold, --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unet.viz_grid import read_patch

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[2] / "models" / "Loss_Ablations"

# Fixed arm plan (docs/loss_ablation.md). (tag, arm-name-for-display, matches):
# `matches(arm)` decides which slot a checkpoint's stored loss_arm falls into.
ARMS: list[tuple[str, str, "callable"]] = [
    ("l1",  "bce",           lambda a: a == "bce"),
    ("l2",  "gap_ce",        lambda a: a == "gap_ce"),
    ("l3",  "tl_ce",         lambda a: a == "tl_ce"),
    ("l4a", "t2_ce",         lambda a: a == "t2_ce"),
    ("l4b", "t4_ce",         lambda a: a == "t4_ce"),
    ("l5",  "pstar_dice",    lambda a: a == "pstar_dice"),
    ("l6",  "pstar_tversky", lambda a: a == "pstar_tversky"),
    ("l7",  "+cldice",       lambda a: a.endswith("+cldice")),
    ("l8",  "+skelrec",      lambda a: a.endswith("+skelrec")),
    ("l9",  "gap_tl_ce",     lambda a: a == "gap_tl_ce"),
    ("l10", "wbce",          lambda a: a == "wbce"),
    ("la0", "bce_dice",      lambda a: a == "bce_dice"),
    ("la1", "focal_tversky", lambda a: a == "focal_tversky"),
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR),
                    help=f"folder of *.ckpt files (default {DEFAULT_CKPT_DIR})")
    ap.add_argument("--image", default=None,
                    help=f"example GeoTIFF (default: first non-mask .tif in {EXAMPLES_DIR})")
    ap.add_argument("--mask", default=None,
                    help="GT raster (default: sibling {stem}_mask.tif)")
    ap.add_argument("--quadrant", type=int, default=None,
                    help="0-3: render a 256px quadrant instead of the full tile")
    ap.add_argument("--rgb-bands", type=int, nargs=3, default=(21, 22, 23),
                    help="1-based display-RGB bands (default 21 22 23 = the "
                         "enhanced R/G/B the arms train on; falls back to 1 2 3)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="global binarization θ override (default: each arm's "
                         "tuned θ* from its train_meta.json, else 0.5)")
    ap.add_argument("--prob", action="store_true",
                    help="probability heatmaps instead of binary masks")
    ap.add_argument("--overlay", action="store_true",
                    help="draw predictions in red over the RGB patch")
    ap.add_argument("--ncols", type=int, default=5)
    ap.add_argument("--device", default="cpu",
                    help="torch device for inference (e.g. cpu, mps, cuda); "
                         "falls back to cpu if the requested backend is unavailable")
    ap.add_argument("--out", default="loss_grid.png")
    return ap.parse_args(argv)


def resolve_device(name: str) -> str:
    """Requested device, downgraded to cpu when its backend isn't available."""
    import torch

    if name == "mps" and not torch.backends.mps.is_available():
        print("WARN: mps unavailable — falling back to cpu")
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("WARN: cuda unavailable — falling back to cpu")
        return "cpu"
    return name


def predict_arm(ckpt: Path, img_path, top, left, size, device="cpu"):
    """(loss_arm, tl_theta|None, sigmoid-probs) for one checkpoint, its own bands/norm."""
    import torch

    from sentinel2data.dataset.reading import apply_norm
    from unet.model import UNetLightning

    # map_location="cpu" then .to(device): deserialising straight onto MPS puts the
    # whole ckpt (incl. unused optimiser state) there — via CPU only the weights land.
    model = UNetLightning.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.eval().float().to(device)
    hp = model.hparams
    arm = str(hp.get("loss_arm") or "?")
    theta = hp.get("tl_theta") if arm in ("tl_ce", "t2_ce", "t4_ce", "gap_tl_ce") else None
    bands = list(hp.get("bands", (1, 2, 3)))
    raw = read_patch(img_path, bands, top, left, size)
    if bool(hp.get("normalize", True)):
        raw = apply_norm(raw, bands, hp.get("norm_mean"), hp.get("norm_std"))
    x = torch.from_numpy(np.ascontiguousarray(raw))[None].to(device)
    pad = (-x.shape[-1]) % 32, (-x.shape[-2]) % 32
    if any(pad):
        x = torch.nn.functional.pad(x, (0, pad[0], 0, pad[1]), mode="reflect")
    with torch.inference_mode():
        probs = torch.sigmoid(model(x))[0, 0, :size, :size]
    return arm, theta, probs.cpu().numpy()


def tuned_theta(ckpt: Path) -> float | None:
    """Val-tuned θ* (best_threshold) from the checkpoint's train_meta.json.

    Handles the sibling layout (models/.../<name>/{<name>.ckpt,train_meta.json})
    and the run-dir layout (<run>/checkpoints/best_f1.ckpt + <run>/train_meta.json).
    Returns None when no meta is found.
    """
    for meta in (ckpt.parent / "train_meta.json",
                 ckpt.parent.parent / "train_meta.json"):
        try:
            return float(json.loads(meta.read_text())["best_threshold"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return None


def render(cells, out_path, ncols, suptitle):
    """cells: (title, img(H,W)|(H,W,3)|None) — None => blank slot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cells)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (title, img) in enumerate(cells):
        ax = axes[k // ncols][k % ncols]
        if img is None:
            ax.set_facecolor("0.93")
            ax.axis("on")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center",
                    va="center", color="0.6", fontsize=20)
        elif img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="magma" if img.dtype.kind == "f" else "gray",
                      vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(h_pad=2.6)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    args = parse_args(argv)
    from sentinel2data.dataset.augment import _overlay_mask, _rgb_for_display

    # --- example patch --------------------------------------------------------
    if args.image:
        img_path = Path(args.image)
    else:
        tifs = sorted(t for t in EXAMPLES_DIR.glob("*.tif") if not t.stem.endswith("_mask"))
        if not tifs:
            raise SystemExit(f"no example tile in {EXAMPLES_DIR} — pass --image")
        img_path = tifs[0]
    mask_path = Path(args.mask) if args.mask else img_path.parent / f"{img_path.stem}_mask.tif"
    if not img_path.is_file():
        raise SystemExit(f"image not found: {img_path}")
    if not mask_path.is_file():
        raise SystemExit(f"mask not found: {mask_path} (pass --mask)")

    if args.quadrant is not None:
        size = 256
        top, left = (args.quadrant // 2) * size, (args.quadrant % 2) * size
    else:
        import rasterio
        with rasterio.open(img_path) as src:
            size = min(src.width, src.height)
        top = left = 0

    rgb = _rgb_for_display(read_patch(img_path, list(args.rgb_bands), top, left, size))
    gt = (read_patch(mask_path, [1], top, left, size)[0] > 0).astype("uint8")

    # --- slot each checkpoint by its stored loss arm --------------------------
    slots: dict[str, np.ndarray] = {}    # arm-tag -> probs
    slot_thr: dict[str, float | None] = {}  # arm-tag -> tuned θ* (None if no meta)
    labels: dict[str, str] = {}          # arm-tag -> concrete loss_arm (+ loss θ)
    extras: list[tuple[str, np.ndarray, float | None]] = []
    ckpts = sorted(Path(args.ckpt_dir).rglob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no *.ckpt under {args.ckpt_dir}")
    device = resolve_device(args.device)
    for ckpt in ckpts:
        try:
            arm, theta, probs = predict_arm(ckpt, img_path, top, left, size, device)
        except Exception as e:   # a broken checkpoint shouldn't kill the grid
            print(f"WARN {ckpt.name}: {e}")
            continue
        star = tuned_theta(ckpt)
        name = arm + (f" θ{theta:g}" if theta is not None else "")
        slot = next((tag for tag, _, match in ARMS
                     if match(arm) and tag not in slots), None)
        if slot is None:
            print(f"WARN {ckpt.name}: arm {arm!r} matches no l-slot (shown as extra)")
            extras.append((f"{name}\n({ckpt.stem})", probs, star))
        else:
            slots[slot], slot_thr[slot], labels[slot] = probs, star, name

    # --- render binary @ tuned θ* (or override / prob / overlay) ---------------
    def cell(title_base, probs, star):
        """One cell binarized at the override θ, else the arm's tuned θ* (θ0.5 fallback)."""
        if probs is None:
            return (title_base, None)
        if args.prob:
            return (title_base, probs.astype("float32"))
        if args.threshold is not None:
            thr, tag = args.threshold, "θ"
        elif star is not None:
            thr, tag = star, "θ*"
        else:
            thr, tag = 0.5, "θ"
        binary = (probs >= thr).astype("uint8")
        title = f"{title_base}  @{tag}{thr:g}"
        return (title, _overlay_mask(rgb, binary) if args.overlay else binary)

    cells = [(f"{img_path.stem}\nr{top} c{left}", rgb),
             ("ground truth", _overlay_mask(rgb, gt) if args.overlay else gt)]
    for tag, arm_name, _ in ARMS:
        base = f"{tag} · {labels.get(tag, arm_name)}"
        cells.append(cell(base, slots.get(tag), slot_thr.get(tag)))
    for base, probs, star in extras:
        cells.append(cell(base, probs, star))

    mode = ("prob" if args.prob else
            f"binary @ θ={args.threshold:g}" if args.threshold is not None else
            "binary @ tuned θ*")
    filled = len(slots) + len(extras)
    out = render(cells, args.out, args.ncols,
                 f"loss ablation · {img_path.stem} · [{mode}]  "
                 f"({filled}/{len(ARMS)} arms)")
    print(f"wrote {out}  ({filled} checkpoints, {len(ARMS) - len(slots)} blank slots)")


if __name__ == "__main__":
    main()
