"""
Warning Note: Completely claude generated. Would look through the code at a later date.

Visually test the dataloader: step through a split showing the RGB satellite
crop next to its ground-truth road mask.

Runs the *real* :class:`RoadTileDataset` (train, random crops) /
:class:`TileCropDataset` (val/test, deterministic quadrants) with normalisation
turned off, so what you see is exactly the crop + mask the dataloader emits (raw
reflectance, percentile-stretched only for display).

    # interactive: arrow-key through the split
    python -m sentinel2data.dataset.preview /path/to/dataset --split train

    # headless: write the first N items to PNGs
    python -m sentinel2data.dataset.preview /path/to/dataset --split val --save out/

Keys (interactive):  [right]/[space]=next  [left]=prev  [q]=quit
"""
from pathlib import Path

import numpy as np

from sentinel2data.dataset.bands import DEFAULT_BANDS
from sentinel2data.dataset.datasets import RoadTileDataset, TileCropDataset


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


def _build_dataset(dataset_dir, split, bands, image_size):
    """Real dataset for ``split`` with normalisation off (dummy stats bypass the
    norm-stats guard; they are never applied because ``normalize=False``).

    ``image_size`` only affects the train (random-crop) dataset; val/test use
    fixed 256px quadrants.
    """
    bands = list(bands)
    mean, std = [0.0] * max(bands), [1.0] * max(bands)  # length >= max band, satisfies the guard
    if split == "train":
        return RoadTileDataset(
            dataset_dir, bands=bands, image_size=image_size,
            normalize=False, norm_mean=mean, norm_std=std,
        )
    return TileCropDataset(
        dataset_dir, split, bands=bands,
        normalize=False, norm_mean=mean, norm_std=std,
    )


def _draw(ax_i, ax_m, ax_o, ds, idx, split, rgb, pct):
    """Render item ``idx`` (RGB | ground-truth | red overlay) into the three axes."""
    image, mask, name = ds[idx]
    rgb_img = _stretch(image, rgb, pct)
    m = np.asarray(mask).squeeze()
    road = float(m.mean()) * 100.0

    ax_i.clear()
    ax_m.clear()
    ax_o.clear()
    ax_i.imshow(rgb_img)
    ax_i.set_title("RGB satellite")
    ax_i.axis("off")
    ax_m.imshow(m, cmap="gray", vmin=0, vmax=1)
    ax_m.set_title(f"ground truth ({road:.1f}% road)")
    ax_m.axis("off")
    ax_o.imshow(_overlay(rgb_img, m))
    ax_o.set_title("GT overlay")
    ax_o.axis("off")
    return name


def preview_dataloader(dataset_dir, split="train", bands=DEFAULT_BANDS,
                       image_size=256, rgb=(0, 1, 2), pct=(2, 98), start=0):
    """Interactive viewer over ``split``: RGB crop | ground-truth mask.

    Arrow-key (or space) to step; ``q`` to quit. Returns the built dataset.
    """
    import matplotlib.pyplot as plt

    ds = _build_dataset(dataset_dir, split, bands, image_size)
    n = len(ds)
    if n == 0:
        raise ValueError(f"Split '{split}' is empty in {dataset_dir}")
    state = {"idx": start % n}

    fig, (ax_i, ax_m, ax_o) = plt.subplots(1, 3, figsize=(13.5, 4.8))

    def show():
        name = _draw(ax_i, ax_m, ax_o, ds, state["idx"], split, rgb, pct)
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


def save_previews(dataset_dir, split="train", out_dir="previews", n=8,
                  bands=DEFAULT_BANDS, image_size=256, rgb=(0, 1, 2), pct=(2, 98),
                  start=0):
    """Headless: write the first ``n`` items of ``split`` as side-by-side PNGs.

    Returns the list of written paths (for no-display / remote runs).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = _build_dataset(dataset_dir, split, bands, image_size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    count = min(n, len(ds))
    for i in range(count):
        idx = (start + i) % len(ds)
        fig, (ax_i, ax_m, ax_o) = plt.subplots(1, 3, figsize=(13.5, 4.8))
        name = _draw(ax_i, ax_m, ax_o, ds, idx, split, rgb, pct)
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
        description="Step through the ROSA dataloader (RGB | ground-truth mask)."
    )
    ap.add_argument("dataset_dir", help="Dataset root (has splits/<split>.csv)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--start", type=int, default=0, help="Starting item index")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="Headless: write previews here instead of an interactive window")
    ap.add_argument("--n", type=int, default=8, help="How many to save (with --save)")
    args = ap.parse_args()

    if args.save:
        save_previews(args.dataset_dir, split=args.split, out_dir=args.save,
                      n=args.n, image_size=args.image_size, start=args.start)
    else:
        preview_dataloader(args.dataset_dir, split=args.split,
                           image_size=args.image_size, start=args.start)
