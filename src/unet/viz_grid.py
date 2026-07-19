"""Qualitative prediction grid: ONE patch, MANY checkpoints, side by side.

Cell 0 is the RGB patch, cell 1 the ground truth; every further cell is one
checkpoint's prediction on that patch (binary at its own tuned θ*, or a
probability heatmap with --prob). A checkpoint that does not exist (arm not
trained yet, fit still running) leaves its cell BLANK, so one fixed grid
layout can be re-rendered as runs land.

Checkpoints come from either/both of:
  * --ckpt "label=path"      explicit entries, in order (repeatable)
  * --runs DIR               scan the staged engine's run dirs
                             (DIR/loss_*_seed*/checkpoints/best_f1.ckpt);
                             labels are the dir names minus the loss_ prefix
  * --expect LABEL           with --runs: fix the grid to these labels (substring
                             match on run-dir name); unmatched -> blank cell

Per-model plumbing is read from the checkpoint itself (bands, norm stats) and
its sibling train_meta.json (tuned θ*; falls back to 0.5) — so arms with
different band sets or thresholds render correctly in one grid.

Examples:
    python -m unet.viz_grid --dataset-dir /scratch/$USER/InstaRoad/ROSA_all \
        --split val --tile 0 --quadrant 3 --runs /scratch/$USER/InstaRoad/runs \
        --out pred_grid.png
    python -m unet.viz_grid ... --expect l1_bce --expect l2_gap_ce_r5 \
        --expect l3_tl_ce_l5 --expect l4a_t2_ce_l5 --expect l4b_t4_ce_l5 --prob
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- cells
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="ROSA root (splits/<split>.csv)")
    ap.add_argument("--split", default="val", help="train | val | test")
    ap.add_argument("--tile", default="0",
                    help="tile row index in the split CSV, or a substring of the "
                         "tile id (image_path stem); first match wins")
    ap.add_argument("--quadrant", type=int, default=0,
                    help="0-3: deterministic 256px quadrant of the 512 tile "
                         "(row-major, as TileCropDataset)")
    ap.add_argument("--top", type=int, default=None, help="custom crop row (overrides quadrant)")
    ap.add_argument("--left", type=int, default=None, help="custom crop col (overrides quadrant)")
    ap.add_argument("--size", type=int, default=256, help="crop edge in px")
    ap.add_argument("--mask-dirname", default=None,
                    help="alternative label dir beside masks_raster (as in training)")
    ap.add_argument("--ckpt", action="append", default=[], metavar="LABEL=PATH",
                    help="explicit checkpoint cell (repeatable; blank if missing)")
    ap.add_argument("--runs", default=None, help="run-dir root to scan (loss_*_seed*)")
    ap.add_argument("--expect", action="append", default=[], metavar="LABEL",
                    help="with --runs: fixed cell order; blank where no run matches")
    ap.add_argument("--threshold", type=float, default=None,
                    help="global binarization override (default: each model's θ* "
                         "from train_meta.json, else 0.5)")
    ap.add_argument("--prob", action="store_true",
                    help="show sigmoid probability heatmaps instead of binary masks")
    ap.add_argument("--overlay", action="store_true",
                    help="draw predictions in red over the RGB patch instead of "
                         "white-on-black masks")
    ap.add_argument("--rgb-bands", type=int, nargs=3, default=(21, 22, 23),
                    help="1-based bands for the display RGB (falls back to 1 2 3 "
                         "if the raster has fewer bands)")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out", default="pred_grid.png")
    return ap.parse_args(argv)


def collect_models(args) -> list[dict]:
    """Ordered [{label, ckpt(Path|None), theta}] — ckpt None => blank cell."""
    models = []
    for entry in args.ckpt:
        label, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--ckpt wants LABEL=PATH, got {entry!r}")
        p = Path(path)
        models.append({"label": label, "ckpt": p if p.is_file() else None,
                       "theta": _theta_for(p)})
    if args.runs:
        run_dirs = sorted(d for d in Path(args.runs).glob("loss_*")
                          if (d / "checkpoints").is_dir() or (d / "config.yaml").is_file()
                          or d.is_dir())
        if args.expect:
            for label in args.expect:
                match = next((d for d in run_dirs if label in d.name), None)
                models.append(_run_cell(label, match))
        else:
            for d in run_dirs:
                models.append(_run_cell(d.name.removeprefix("loss_"), d))
    if not models:
        raise SystemExit("no checkpoints requested: pass --ckpt and/or --runs")
    return models


def _run_cell(label: str, run_dir: Path | None) -> dict:
    if run_dir is None:
        return {"label": label, "ckpt": None, "theta": 0.5}
    ckpt = run_dir / "checkpoints" / "best_f1.ckpt"
    return {"label": label, "ckpt": ckpt if ckpt.is_file() else None,
            "theta": _theta_for(ckpt)}


def _theta_for(ckpt: Path) -> float:
    """Tuned θ* from the run dir's train_meta.json (ckpt lives in checkpoints/)."""
    meta = ckpt.parent.parent / "train_meta.json"
    try:
        return float(json.loads(meta.read_text())["best_threshold"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return 0.5


# ------------------------------------------------------------------ imagery
def read_patch(path, bands, top, left, size):
    """(C, size, size) float32 window, zero-padded at tile edges."""
    import rasterio
    from rasterio.windows import Window

    from sentinel2data.dataset.reading import read_window

    with rasterio.open(path) as src:
        bands = [b for b in bands if b <= src.count] or list(range(1, min(4, src.count + 1)))
        img = read_window(src, bands, Window(left, top, min(size, src.width - left),
                                             min(size, src.height - top)))
    c, h, w = img.shape
    if (h, w) != (size, size):
        pad = np.zeros((c, size, size), dtype="float32")
        pad[:, :h, :w] = img
        img = pad
    return img


def predict(ckpt: Path, img_path, top, left, size) -> np.ndarray:
    """Sigmoid probabilities (size, size) from one checkpoint, its own bands/norm."""
    import torch

    from sentinel2data.dataset.reading import apply_norm
    from unet.model import UNetLightning

    model = UNetLightning.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.eval().float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    hp = model.hparams
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
    return probs.cpu().numpy()


# ------------------------------------------------------------------- render
def render(cells, out_path, ncols, suptitle):
    """cells: (title, image(H,W) or (H,W,3) or None) — None => blank cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cells)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (title, img) in enumerate(cells):
        ax = axes[k // ncols][k % ncols]
        if img is None:
            ax.set_facecolor("0.92")
            ax.axis("on")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                    ha="center", va="center", color="0.6", fontsize=18)
        elif img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="viridis" if img.dtype.kind == "f" else "gray",
                      vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    args = parse_args(argv)

    import pandas as pd

    from sentinel2data.dataset.augment import _overlay_mask, _rgb_for_display

    # split CSV + optional mask remap (mirrors sentinel2data.dataset.datasets,
    # inlined so the grid renders without the lightning training stack)
    csv = Path(args.dataset_dir) / "splits" / f"{args.split}.csv"
    if not csv.exists():
        raise SystemExit(f"Split CSV not found: {csv}")
    df = pd.read_csv(csv)
    if args.mask_dirname:
        df = df.copy()
        df["mask_path"] = df["mask_path"].map(
            lambda rel: str(Path(rel).parent.parent / args.mask_dirname / Path(rel).name))
    if args.tile.isdigit():
        row = df.iloc[int(args.tile)]
    else:
        hits = df[df["image_path"].map(lambda p: args.tile in Path(p).stem)]
        if hits.empty:
            raise SystemExit(f"no {args.split} tile matches {args.tile!r}")
        row = hits.iloc[0]
    tile_id = Path(row["image_path"]).stem
    s = args.size
    top = args.top if args.top is not None else (args.quadrant // 2) * s
    left = args.left if args.left is not None else (args.quadrant % 2) * s
    img_path = Path(args.dataset_dir) / row["image_path"]
    mask_path = Path(args.dataset_dir) / row["mask_path"]

    rgb = _rgb_for_display(read_patch(img_path, list(args.rgb_bands), top, left, s))
    gt = (read_patch(mask_path, [1], top, left, s)[0] > 0).astype("uint8")

    cells = [(f"{tile_id}\nr{top} c{left}", rgb),
             ("ground truth", _overlay_mask(rgb, gt) if args.overlay
              else gt.astype("uint8"))]
    for m in collect_models(args):
        if m["ckpt"] is None:
            cells.append((m["label"], None))
            continue
        theta = args.threshold if args.threshold is not None else m["theta"]
        try:
            probs = predict(m["ckpt"], img_path, top, left, s)
        except Exception as e:  # a broken checkpoint shouldn't kill the grid
            print(f"WARN {m['label']}: {e}")
            cells.append((f"{m['label']} (load failed)", None))
            continue
        if args.prob:
            cells.append((f"{m['label']}", probs.astype("float32")))
        elif args.overlay:
            cells.append((f"{m['label']}  θ={theta}",
                          _overlay_mask(rgb, (probs >= theta).astype("uint8"))))
        else:
            cells.append((f"{m['label']}  θ={theta}",
                          (probs >= theta).astype("uint8")))

    n_blank = sum(1 for _, img in cells if img is None)
    sup = f"{args.split}/{tile_id}  quadrant ({top},{left})  " \
          f"[{'prob' if args.prob else 'binary @ θ*'}]"
    out = render(cells, args.out, args.ncols, sup)
    print(f"wrote {out}  ({len(cells) - 2} model cells, {n_blank} blank)")


if __name__ == "__main__":
    main()
