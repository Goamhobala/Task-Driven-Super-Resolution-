"""Real-time data augmentation for the road datasets, built on Albumentations.

The transform is applied per-patch inside ``RoadTileDataset.__getitem__`` (train
split only), so every epoch sees freshly augmented chips at no extra disk cost.
``build_transform`` is a menu of independently toggleable augmentations:

  * flip     — D4 dihedral group (flips + 90 deg rotations). Lossless, the safe
               default; reorders pixels identically across all bands + the mask.
  * sharpen  — unsharp-mask style edge enhancement.
  * noise    — additive per-band Gaussian noise.
  * blur     — Gaussian blur. POTENTIALLY DETRIMENTAL for thin road features.
  * colour   — per-band brightness/contrast jitter (the multispectral stand-in
               for RGB colour jitter — hue/saturation don't generalise to the
               full band stack). POTENTIALLY DETRIMENTAL.

IMPORTANT — these run on *normalised* (z-scored, ~unit-std) input, because the
dataset applies the transform after ``apply_norm``. So the photometric
magnitudes below are in normalised units, not raw reflectance, and are tuned
accordingly. Flip is unaffected by this.

`build_transform` returns the `albumentations.Compose`; `visualize_augmentations`
renders a matplotlib grid so you can eyeball that image and mask stay aligned
under the augmentation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["build_transform", "AUG_FLAGS", "visualize_augmentations"]

# The toggleable photometric/geometric augmentations, in apply order. `flip` is
# kept first and separate (it is the lossless geometric default, always p=1.0
# when on); the rest default off and share the `p` probability.
AUG_FLAGS = ("flip", "sharpen", "noise", "blur", "colour")

# Maps each Albumentations transform class to its `AUG_FLAGS` name, so a replay
# record (which speaks in class names) can be turned back into our toggle names.
_AUG_BY_CLASS = {
    "D4": "flip",
    "Sharpen": "sharpen",
    "GaussNoise": "noise",
    "GaussianBlur": "blur",
    "RandomBrightnessContrast": "colour",
}

# Readable names for the 8 elements of the D4 dihedral group, as reported by
# A.D4's `group_element` param. Lets a flip-only grid label *which* symmetry each
# cell is (rot90, hflip, ...) rather than just "flip". Unknown keys pass through.
_D4_NAMES = {
    "e": "identity", "r90": "rot90", "r180": "rot180", "r270": "rot270",
    "v": "vflip", "h": "hflip", "t": "transpose", "hvt": "antitranspose",
}


def build_transform(flip: bool = True, sharpen: bool = False, noise: bool = False,
                    blur: bool = False, colour: bool = False,
                    p: float = 0.5, seed: int | None = None):
    """Build the Albumentations pipeline from a set of augmentation toggles.

    `flip` is the D4 dihedral group via `A.D4`: a single transform that picks a
    *uniformly* random element of the 8 flip/rotation symmetries (identity, 3
    rotations, 2 axis flips, 2 diagonal flips) — cleaner and unbiased vs.
    composing HFlip/VFlip/RandomRotate90/Transpose, and it subsumes the "50%
    horizontal/vertical flip" idea. The photometric toggles each fire with
    probability `p`. Magnitudes are tuned for z-scored input (see module
    docstring). Albumentations is imported lazily.
    """
    import albumentations as A

    tfms = []
    if flip:
        tfms.append(A.D4(p=1.0))
        
    if sharpen:
        tfms.append(A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=p))
    if noise:
        # std in normalised units: ~5-20% of the unit std of the z-scored bands.
        tfms.append(A.GaussNoise(std_range=(0.05, 0.2), per_channel=True, p=p))
    if blur:
        tfms.append(A.GaussianBlur(blur_limit=(3, 5), p=p))
    if colour:
        tfms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p))
    if not tfms:
        raise ValueError("build_transform: no augmentations enabled (all toggles False)")
    return A.Compose(tfms, seed=seed)


def _item_arrays(item):
    """Unpack a dataset item to ``(img (C,H,W), mask (H,W))`` numpy arrays.

    Handles both this package's ``(image, mask, name)`` datasets and plain
    ``(image, mask)`` pairs; the mask may carry a leading channel dim.
    """
    x, y = item[0], item[1]
    img = np.asarray(x)
    mask = np.asarray(y)
    if mask.ndim == 3:
        mask = mask[0]
    return img, mask


def _rgb_for_display(x: np.ndarray) -> np.ndarray:
    """Turn a (C, H, W) patch into an (H, W, 3) uint8 RGB for plotting.

    Channels 0,1,2 are R/G/B for all the common band groups (RGB, S2_10M,
    ENHANCED_RGB — see ``sentinel2data.dataset.bands``). The patch is already
    z-scored, so we percentile-stretch each channel (2-98%) to get a viewable
    image regardless of the normalisation.
    """
    rgb = np.asarray(x)[:3].astype("float32")
    out = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[i]
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        out[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return (out.transpose(1, 2, 0) * 255).astype("uint8")


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Tint road pixels red so geometric alignment is obvious in the grid."""
    out = rgb.astype("float32").copy()
    road = mask > 0
    out[road] = (1 - alpha) * out[road] + alpha * np.array([255.0, 0.0, 0.0])
    return out.astype("uint8")


def _describe_replay(replay) -> str:
    """Human-readable label of which augmentations actually fired in a draw.

    Reads an Albumentations `ReplayCompose` record and names the transforms whose
    `applied` flag is set, mapping each class back to its `AUG_FLAGS` name. For
    the D4 flip it also appends the chosen orientation (e.g. ``flip:rot90``) so a
    flip-only grid shows *which* of the 8 symmetries each cell is. Returns
    ``"none"`` when a draw happened to apply nothing (the identity).
    """
    parts = []
    for t in replay.get("transforms", []):
        if not t.get("applied"):
            continue
        cls = t["__class_fullname__"].rsplit(".", 1)[-1]
        name = _AUG_BY_CLASS.get(cls, cls)
        if cls == "D4":
            elem = (t.get("params") or {}).get("group_element")
            if elem is not None:
                name = f"{name}:{_D4_NAMES.get(elem, elem)}"
        parts.append(name)
    return "+".join(parts) if parts else "none"


def _pick_informative_patch(dataset, rng, sample=40):
    """``(idx, (img, mask))`` of the patch with the most road among a sample.

    A blank, road-free patch makes a useless augmentation example, so we peek at
    up to `sample` patches and keep the one with the highest road fraction. The
    ARRAYS are returned (not just the index) because random-crop datasets like
    ``RoadTileDataset`` return a different patch every time an index is fetched.
    """
    n = min(sample, len(dataset))
    cand = rng.choice(len(dataset), size=n, replace=False)
    best, best_arrays, best_frac = int(cand[0]), None, -1.0
    for i in cand:
        img, mask = _item_arrays(dataset[int(i)])
        frac = float(mask.mean())
        if frac > best_frac:
            best, best_arrays, best_frac = int(i), (img, mask), frac
    return best, best_arrays


def _enabled_flags(transform):
    """The `AUG_FLAGS` names present in a built `transform`, in apply order."""
    names = []
    for t in transform.transforms:
        name = _AUG_BY_CLASS.get(type(t).__name__)
        if name:
            names.append(name)
    return names


def _render_grid(cells, out_path, ncols, overlay, suptitle):
    """Plot `(rgb, mask, title)` cells into a grid PNG (headless, via Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = len(cells)
    ncols = min(ncols, total)
    nrows = (total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")  # hides ticks and any trailing unused cells
    for k, (crgb, cmask, title) in enumerate(cells):
        ax = axes[k // ncols][k % ncols]
        ax.imshow(_overlay_mask(crgb, cmask) if overlay else crgb)
        if title:
            ax.set_title(title, fontsize=10)

    if overlay:
        suptitle += " (road mask overlaid in red)"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_augmentations(dataset, transform, out_path, patch_idx=None,
                            n_aug=11, ncols=4, seed=0, labels=False, overlay=False,
                            isolate=False):
    """Save a grid of ONE patch and its augmented variations (headless, via Agg).

    Two modes:
      * default (combined) — cell 0 is the un-augmented patch; the rest are up to
        `n_aug` *distinct* draws of the full `transform`, exactly as training sees
        it (so a single cell may stack flip+noise+colour). A draw reproducing the
        original or an already-shown one is rejected, so flip-only yields the 7
        non-identity D4 orientations and then stops rather than padding with
        duplicates.
      * `isolate` — one cell per *enabled* augmentation, each applied ALONE and
        forced to fire (p=1.0), so you can eyeball each transform's effect on its
        own. This is a visualisation aid only; it does not reflect the training
        distribution (which combines them) and does not change `build_transform`.

    If `patch_idx` is None, an informative (road-rich) patch is picked
    automatically.

    Two independent annotations:
      * `labels` titles each cell with the augmentations that actually fired in
        that draw (e.g. ``flip:rot90`` or ``flip+noise``), obtained by replaying
        the transform via `A.ReplayCompose`. (In `isolate` mode every cell is
        titled with its augmentation name regardless of this flag.)
      * `overlay` tints the ground-truth road mask red on every cell, so you can
        confirm image and mask transform together.
    Both default off, giving clean augmented chips.
    """
    rng = np.random.default_rng(seed)
    if patch_idx is None:
        patch_idx, (img, mask) = _pick_informative_patch(dataset, rng)
    else:
        img, mask = _item_arrays(dataset[patch_idx])
    # img: (C, H, W) normalised; mask: (H, W)
    hwc = img.transpose(1, 2, 0)

    if isolate:
        # One cell per enabled augmentation, each applied alone at p=1.0 so its
        # effect is guaranteed visible. Rebuilt from build_transform rather than
        # the composed `transform` so the photometric toggles are forced to fire.
        cells = [(_rgb_for_display(img), mask, "original")]
        for name in _enabled_flags(transform):
            single = build_transform(**{f: f == name for f in AUG_FLAGS},
                                     p=1.0, seed=seed)
            aug = single(image=hwc, mask=mask)
            a_rgb = _rgb_for_display(aug["image"].transpose(2, 0, 1))
            cells.append((a_rgb, aug["mask"], name))
        suptitle = f"Each augmentation applied alone to patch {patch_idx}"
        return _render_grid(cells, out_path, ncols, overlay, suptitle)

    # For labelling we need to know which transforms fired per draw, which a plain
    # Compose doesn't expose — re-wrap the same transforms in a ReplayCompose.
    tf = transform
    if labels:
        import albumentations as A
        tf = A.ReplayCompose(transform.transforms)
        tf.set_random_seed(seed)  # ReplayCompose has no seed= ctor arg (AB 2.x)

    cells = [(_rgb_for_display(img), mask, "original" if labels else "")]
    seen = {img.tobytes()}         # reject draws identical to the original
    attempts = max(200, n_aug * 30)  # bounded: flip-only has only 7 distinct draws
    for _ in range(attempts):
        if len(cells) > n_aug:
            break
        aug = tf(image=hwc, mask=mask)
        key = aug["image"].tobytes()
        if key in seen:
            continue
        seen.add(key)
        a_rgb = _rgb_for_display(aug["image"].transpose(2, 0, 1))
        title = _describe_replay(aug["replay"]) if labels else ""
        cells.append((a_rgb, aug["mask"], title))

    suptitle = f"Augmentation variations of patch {patch_idx}"
    return _render_grid(cells, out_path, ncols, overlay, suptitle)


if __name__ == "__main__":
    # Visual check: load one train tile from a ROSA dataset and render the
    # augmented variations of a single patch.
    import argparse
    import subprocess
    import sys

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sentinel2data.dataset.datasets import RoadTileDataset  # noqa: E402

    ap = argparse.ArgumentParser(
        description="Visualise augmentation on a ROSA train tile. Saves a grid "
                    "PNG: cell 0 = original patch, rest = augmented draws."
    )
    ap.add_argument("--dataset-dir", required=True,
                    help="ROSA dataset root (has splits/train.csv)")
    ap.add_argument("--norm-config", default="src/unet/configs/norm_stats.yaml",
                    help="YAML with data.norm_mean/data.norm_std (frozen train stats)")
    ap.add_argument("--bands", type=int, nargs="+", default=[21, 22, 23, 4],
                    help="1-based band indices (default: enhanced RGB + NIR)")
    ap.add_argument("--out", default="augment_preview.png",
                    help="output PNG path (default: augment_preview.png)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducible patch selection and augmentation")
    ap.add_argument("--patch", type=int, default=None,
                    help="patch index to visualise (default: auto-pick the most road-rich)")
    ap.add_argument("--n-aug", type=int, default=11,
                    help="number of augmented variations to show (default: 11)")
    ap.add_argument("--labels", action="store_true",
                    help="title each cell with the augmentations that fired")
    ap.add_argument("--overlay", action="store_true",
                    help="tint the ground-truth road mask red on each cell")
    ap.add_argument("--isolate", action="store_true",
                    help="show each enabled augmentation applied alone (forced to fire)")
    # Augmentation toggles — flip on by default, the rest opt-in.
    ap.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True,
                    help="D4 flips + rotations (default: on)")
    ap.add_argument("--sharpen", action="store_true", help="edge sharpening")
    ap.add_argument("--noise", action="store_true", help="additive per-band Gaussian noise")
    ap.add_argument("--blur", action="store_true", help="Gaussian blur (potentially detrimental)")
    ap.add_argument("--colour", action="store_true",
                    help="per-band brightness/contrast jitter (potentially detrimental)")
    args = ap.parse_args()

    norm = yaml.safe_load(Path(args.norm_config).read_text())["data"]
    # No transform on the dataset itself: the grid applies the augmentation so
    # the original patch stays un-augmented in cell 0.
    ds = RoadTileDataset(args.dataset_dir, bands=tuple(args.bands),
                         norm_mean=norm["norm_mean"], norm_std=norm["norm_std"])
    print(f"train patches/epoch: {len(ds)}")

    transform = build_transform(flip=args.flip, sharpen=args.sharpen, noise=args.noise,
                                blur=args.blur, colour=args.colour, seed=args.seed)
    enabled = [n for n in AUG_FLAGS if getattr(args, n)]
    print(f"augmentations: {', '.join(enabled) or 'none'}")
    out_path = visualize_augmentations(ds, transform, args.out, patch_idx=args.patch,
                                       n_aug=args.n_aug, seed=args.seed,
                                       labels=args.labels, overlay=args.overlay,
                                       isolate=args.isolate)
    print(f"saved -> {out_path}")

    # Open with the system viewer on macOS so the grid appears immediately.
    if sys.platform == "darwin":
        subprocess.run(["open", str(out_path)], check=False)
