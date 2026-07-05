"""Visually test the super-resolution dataloader: step through a split showing the
bicubic-upsampled RGB crop next to its graph-rasterised road mask.

Runs the *real* :class:`UpscaleRoadTileDataset` (train, random crops) /
:class:`UpscaleTileCropDataset` (val/test, deterministic cells) with normalisation
off, so what you see is exactly what the SR dataloader emits: a ``crop_size`` native
window bicubic-upsampled to ``crop_size * upscale`` px, and the mask rasterised fresh
from that tile's masks_graph parquet (per-row ``buffer`` half-width) at the upsampled
transform.

    # interactive: arrow-key through the split
    python -m sentinel2data.dataset.preview_upsampler /path/to/dataset --split train

    # headless: write the first N items to PNGs
    python -m sentinel2data.dataset.preview_upsampler /path/to/dataset --split val --save out/

Keys (interactive):  [right]/[space]=next  [left]=prev  [q]=quit
"""
from pathlib import Path

import numpy as np

from sentinel2data.dataset.bands import ENHANCED_RGB, RGB
from sentinel2data.dataset.upscale_dataset import (
    UpscaleRoadTileDataset,
    UpscaleTileCropDataset,
)

# --mode display source: raw B4/B3/B2 vs the appended CLAHE+gamma enhanced RGB (21-23).
RGB_MODES = {"normal": RGB, "enhanced": ENHANCED_RGB}


def _stretch(chw, rgb=(0, 1, 2), pct=(2, 98)):
    """``(C,H,W)`` array/tensor -> ``(H,W,3)`` float in [0,1], per-channel
    percentile-stretched (0-valued nodata excluded), for display only."""
    arr = np.asarray(chw, dtype="float32")
    h, w = arr.shape[1], arr.shape[2]
    out = np.zeros((h, w, 3), dtype="float32")
    for k, b in enumerate(rgb):
        ch = arr[b]
        valid = ch[np.isfinite(ch) & (ch > 0)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, pct)
        if hi <= lo:
            continue
        out[..., k] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out


def _overlay(rgb_img, mask, alpha=0.45, color=(1.0, 0.0, 0.0)):
    """RGB image with the binary mask alpha-blended in ``color`` (default red)."""
    ov = rgb_img.copy()
    m = np.asarray(mask) > 0
    ov[m] = (1.0 - alpha) * ov[m] + alpha * np.asarray(color, dtype="float32")
    return np.clip(ov, 0.0, 1.0)


def _build_dataset(dataset_dir, split, bands, crop_size, upscale):
    """Real SR dataset for ``split`` with normalisation off (raw reflectance;
    the upscale datasets need no frozen stats when ``normalize=False``)."""
    bands = list(bands)
    if split == "train":
        return UpscaleRoadTileDataset(
            dataset_dir, bands=bands, crop_size=crop_size, upscale=upscale,
            normalize=False,
        )
    return UpscaleTileCropDataset(
        dataset_dir, split, bands=bands, crop_size=crop_size, upscale=upscale,
        normalize=False,
    )


def _draw(ax_i, ax_m, ax_o, ds, idx, upscale, mode, rgb, pct):
    """Render item ``idx`` (bicubic RGB | graph mask | red overlay) into the three axes."""
    image, mask, name = ds[idx]
    rgb_img = _stretch(image, rgb, pct)
    m = np.asarray(mask).squeeze()
    road = float(m.mean()) * 100.0
    edge = rgb_img.shape[0]
    prefix = "enhanced " if mode == "enhanced" else ""

    ax_i.clear()
    ax_m.clear()
    ax_o.clear()
    ax_i.imshow(rgb_img)
    ax_i.set_title(f"{prefix}bicubic RGB  {edge}px (x{upscale})")
    ax_i.axis("off")
    ax_m.imshow(m, cmap="gray", vmin=0, vmax=1)
    ax_m.set_title(f"graph mask ({road:.1f}% road)")
    ax_m.axis("off")
    ax_o.imshow(_overlay(rgb_img, m))
    ax_o.set_title("GT overlay")
    ax_o.axis("off")
    return name


def preview_upsampler(dataset_dir, split="train", mode="normal",
                      crop_size=128, upscale=4, rgb=(0, 1, 2), pct=(2, 98), start=0):
    """Interactive viewer over ``split``: bicubic RGB | graph mask | red overlay.

    ``mode`` picks the display RGB source: ``"normal"`` (bands 1-3) or ``"enhanced"``
    (CLAHE+gamma bands 21-23). Arrow-key (or space) to step; ``q`` to quit.
    """
    import matplotlib.pyplot as plt

    ds = _build_dataset(dataset_dir, split, RGB_MODES[mode], crop_size, upscale)
    n = len(ds)
    if n == 0:
        raise ValueError(f"Split '{split}' is empty in {dataset_dir}")
    state = {"idx": start % n}

    fig, (ax_i, ax_m, ax_o) = plt.subplots(1, 3, figsize=(13.5, 4.8))

    def show():
        name = _draw(ax_i, ax_m, ax_o, ds, state["idx"], upscale, mode, rgb, pct)
        fig.suptitle(f"[{state['idx'] + 1}/{n}] {split}  ·  {name}", fontsize=10)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", " ", "n"):
            state["idx"] = (state["idx"] + 1) % n
        elif event.key in ("left", "p"):
            state["idx"] = (state["idx"] - 1) % n
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    print(f"{split}: {n} items.  [right]/[space]=next  [left]=prev  [q]=quit")
    show()
    plt.tight_layout()
    plt.show()
    return ds


def save_previews(dataset_dir, split="train", out_dir="previews_upsampled", n=8,
                  mode="normal", crop_size=128, upscale=4, rgb=(0, 1, 2),
                  pct=(2, 98), start=0):
    """Headless: write the first ``n`` items of ``split`` as side-by-side PNGs.

    ``mode`` = ``"normal"`` (bands 1-3) or ``"enhanced"`` (bands 21-23). Returns the
    list of written paths (for no-display / remote runs).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = _build_dataset(dataset_dir, split, RGB_MODES[mode], crop_size, upscale)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    count = min(n, len(ds))
    for i in range(count):
        idx = (start + i) % len(ds)
        fig, (ax_i, ax_m, ax_o) = plt.subplots(1, 3, figsize=(13.5, 4.8))
        name = _draw(ax_i, ax_m, ax_o, ds, idx, upscale, mode, rgb, pct)
        fig.suptitle(f"[{idx + 1}/{len(ds)}] {split}  ·  {name}", fontsize=10)
        fig.tight_layout()
        out = out_dir / f"{split}_{idx:04d}_{Path(name).stem}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)
    print(f"Wrote {len(written)} previews to {out_dir}")
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Step through the SR (upscale) dataloader (bicubic RGB | graph mask)."
    )
    ap.add_argument("dataset_dir", help="Dataset root (has splits/<split>.csv)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--mode", default="normal", choices=["normal", "enhanced"],
                    help="Display RGB source: normal (bands 1-3) or enhanced (21-23)")
    ap.add_argument("--crop-size", type=int, default=128, help="Native-px crop window")
    ap.add_argument("--upscale", type=int, default=4, help="Bicubic upscale factor")
    ap.add_argument("--start", type=int, default=0, help="Starting item index")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="Headless: write previews here instead of an interactive window")
    ap.add_argument("--n", type=int, default=8, help="How many to save (with --save)")
    args = ap.parse_args()

    if args.save:
        save_previews(args.dataset_dir, split=args.split, out_dir=args.save,
                      n=args.n, mode=args.mode, crop_size=args.crop_size,
                      upscale=args.upscale, start=args.start)
    else:
        preview_upsampler(args.dataset_dir, split=args.split, mode=args.mode,
                          crop_size=args.crop_size, upscale=args.upscale,
                          start=args.start)
