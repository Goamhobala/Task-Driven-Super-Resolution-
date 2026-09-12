"""Build the FROZEN probe fixtures: a chip list and a pixel sample.

Run ONCE. Everything downstream (`extract.py`, then `lda.py` / `cka.py` /
`occlusion.py`) reads these two files, so every arm is measured on byte-
identical ground and clouds are paired pixel-for-pixel across arms.

    python -m sr.probes.make_fixtures --dataset-dir <ROSA> --n-chips 400

writes, into `src/sr/probes/fixtures/`:

    probe_chips.json   the chip list  (tile, window, stratum, road_px, ...)
    probe_pixels.npz   the pixel sample (chip index, flat HR index, is_road)

WHAT A CHIP IS, AND WHY THAT UNIT
---------------------------------
One chip = one NATIVE 128 px window of a test tile -> 512 px at 2.5 m. That is
the model's OWN forward unit: SEN2SR's shipped FFT mask pins the LR input to
128 px (`model._required_lr`), the training loader crops at 128, and
`JointSRTileCropDataset` walks exactly this non-overlapping grid. So a chip is
one un-stitched forward — no cell assembly, no convolution-border question, and
the arms with an unpinned generator (bicubic, SR4RS) see the same footprint as
the pinned ones. The bench's 2560 m cell is a different unit for a different
job (throughput); these probes are not the bench and must not be read as
reproducing its numbers.

STRATIFICATION AND THE EMPTY CHIPS
----------------------------------
Chips are allocated across `urbanisation_classification` proportionally to the
strata's tile counts, then drawn UNIFORMLY inside each stratum. Uniform is the
point: road-free chips then appear at their natural rate rather than being
filtered out, which is what the plan asks for and what keeps the occlusion
readout honest — an arm that suppresses clutter is only visible if clutter-only
chips are in the sample.

THE PIXEL SAMPLE IS POOLED, NOT PER-CHIP
----------------------------------------
Road pixels are drawn from the POOLED road population of the fixture (each chip
contributing in proportion to its road area), background likewise, at
`--bg-per-road` : 1. A fixed per-chip quota would over-weight the sparse rural
chips, which is the opposite of what LDA needs — it wants an unbiased sample of
the road-pixel population the model actually sees. Chips with no road simply
contribute background, at their natural weight.

The indices are written out rather than regenerated on demand: a stored sample
cannot drift when a mask, a seed convention or a numpy version moves under it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# Defaults confirmed against the R-series arm scripts (`_stages_tv.sh` in
# LABELS=new mode): the arms train and bench on ROSA_New with the pre-rasterised
# 2.5 m mask COGs, so the probes must read the same labels.
DATASET_DIR = "/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset"
SPLIT = "test"
MASK_DIRNAME = "mask_new_2pt5"
CROP = 128          # SEN2SR's FFT mask pins the LR patch; also the training crop
UPSCALE = 4
STRATA_COL = "urbanisation_classification"

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
CHIPS_JSON = "probe_chips.json"
PIXELS_NPZ = "probe_pixels.npz"


def _git_sha() -> str | None:
    """HEAD of the repo this file lives in — provenance for the frozen fixture."""
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                              "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _hr_mask_path(dataset_dir: Path, image_path: str, mask_dirname: str) -> Path:
    """`<split>/imagery/x.tif` -> `<split>/<mask_dirname>/x.tif` (loader's rule)."""
    rel = Path(image_path)
    return dataset_dir / rel.parent.parent / mask_dirname / rel.name


def enumerate_chips(dataset_dir: Path, split: str, crop: int) -> pd.DataFrame:
    """Every non-overlapping `crop` px window of every tile in the split.

    Mirrors `JointSRTileCropDataset._tile_grid`: the grid is derived from the
    FIRST tile's dimensions and applied to all of them, which is exactly what
    the eval loader does (the ROSA tiles are a uniform 512 px).
    """
    csv = dataset_dir / "splits" / f"{split}.csv"
    if not csv.exists():
        raise SystemExit(f"split CSV not found: {csv}")
    df = pd.read_csv(csv).reset_index(drop=True)
    with rasterio.open(dataset_dir / df.iloc[0]["image_path"]) as src:
        h, w = src.height, src.width
    gw, gh = (w + crop - 1) // crop, (h + crop - 1) // crop

    rows = []
    for _, r in df.iterrows():
        for cell in range(gw * gh):
            rows.append({
                "tile": Path(r["image_path"]).stem,
                "image_path": r["image_path"],
                "mask_graph_path": r["mask_graph_path"],
                "zone_name": r["zone_name"],
                "biome": r["biome"],
                "stratum": r[STRATA_COL],
                "cell": cell,
                "top": (cell // gw) * crop,
                "left": (cell % gw) * crop,
            })
    return pd.DataFrame(rows)


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding (sums to `total`)."""
    n = sum(counts.values())
    exact = {k: total * v / n for k, v in counts.items()}
    base = {k: int(np.floor(v)) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: exact[k] - base[k], reverse=True):
        if sum(base.values()) >= total:
            break
        base[k] += 1
    return base


def sample_chips(all_chips: pd.DataFrame, n_chips: int, seed: int) -> pd.DataFrame:
    """Stratified-by-count, uniform-within-stratum draw of `n_chips` windows."""
    rng = np.random.default_rng(seed)
    per_stratum = allocate(all_chips["stratum"].value_counts().to_dict(), n_chips)
    picks = []
    for stratum in sorted(per_stratum):
        pool = all_chips[all_chips["stratum"] == stratum]
        k = min(per_stratum[stratum], len(pool))
        picks.append(pool.iloc[rng.choice(len(pool), size=k, replace=False)])
    out = pd.concat(picks).sort_values(["tile", "cell"]).reset_index(drop=True)
    return out


def read_chip_stats(dataset_dir: Path, chip: dict, mask_dirname: str,
                    crop: int, upscale: int):
    """(hr_mask bool (crop*up)^2, road_px, valid_frac) for one chip.

    `valid_frac` is the share of the LR window that is not nodata — recorded so
    an analysis can, if it wants, separate "the model saw nothing here" from
    "the model saw ground with no road". It is NOT used to filter the fixture:
    the eval loader scores these windows, so the probes do too.
    """
    img = dataset_dir / chip["image_path"]
    with rasterio.open(img) as src:
        H, W = src.height, src.width
        win = Window(chip["left"], chip["top"],
                     min(crop, W - chip["left"]), min(crop, H - chip["top"]))
        arr = src.read([1, 2, 3, 4], window=win).astype("float32")
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr == -32768] = 0.0
    valid = float((np.abs(arr).sum(axis=0) > 0).mean())

    out = crop * upscale
    hr = _hr_mask_path(dataset_dir, chip["image_path"], mask_dirname)
    with rasterio.open(hr) as src:
        m = src.read(1, window=Window(int(win.col_off) * upscale,
                                      int(win.row_off) * upscale,
                                      int(win.width) * upscale,
                                      int(win.height) * upscale)) > 0
    if m.shape != (out, out):                       # edge tiles: zero-pad, as the loader does
        pad = np.zeros((out, out), dtype=bool)
        pad[:m.shape[0], :m.shape[1]] = m
        m = pad
    return m, int(m.sum()), valid


def sample_pixels(masks: list[np.ndarray], road_px: np.ndarray,
                  n_total: int, bg_per_road: float, seed: int):
    """Pooled road/background pixel draw -> (chip_idx, flat_index, is_road).

    Allocation is proportional to each chip's road (resp. background) count, so
    the draw is an unbiased sample of the fixture's pixel population; the
    within-chip choice is then uniform without replacement.
    """
    rng = np.random.default_rng(seed)
    n_road = int(round(n_total / (1.0 + bg_per_road)))
    n_bg = n_total - n_road

    npx = masks[0].size
    bg_px = np.array([npx - r for r in road_px], dtype="int64")

    def draw(counts: np.ndarray, want: int, road: bool):
        want = int(min(want, counts.sum()))
        if want == 0:
            return np.empty(0, "int32"), np.empty(0, "int32")
        # Multivariate-hypergeometric-in-spirit: proportional quotas, capped by
        # what each chip actually holds, remainder redistributed. Simpler than a
        # true MVHG draw and indistinguishable at these counts.
        quota = np.floor(counts / counts.sum() * want).astype("int64")
        quota = np.minimum(quota, counts)
        while quota.sum() < want:
            room = counts - quota
            cand = np.flatnonzero(room > 0)
            if cand.size == 0:
                break
            take = rng.choice(cand, size=min(int(want - quota.sum()), cand.size),
                              replace=False)
            quota[take] += 1
        chips, idxs = [], []
        for i, k in enumerate(quota):
            if k <= 0:
                continue
            flat = np.flatnonzero(masks[i].ravel() if road else ~masks[i].ravel())
            sel = rng.choice(flat, size=int(k), replace=False)
            chips.append(np.full(int(k), i, dtype="int32"))
            idxs.append(sel.astype("int32"))
        return np.concatenate(chips), np.concatenate(idxs)

    rc, ri = draw(np.asarray(road_px, dtype="int64"), n_road, road=True)
    bc, bi = draw(bg_px, n_bg, road=False)
    chip_idx = np.concatenate([rc, bc])
    pix_idx = np.concatenate([ri, bi])
    is_road = np.concatenate([np.ones(rc.size, bool), np.zeros(bc.size, bool)])
    # Sort by (chip, pixel) so extraction reads each chip's SR output once, in
    # order, and every arm's cache rows line up by construction.
    order = np.lexsort((pix_idx, chip_idx))
    return chip_idx[order], pix_idx[order], is_road[order]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=DATASET_DIR)
    ap.add_argument("--split", default=SPLIT,
                    help="test by design — see the plan's §2 on why test is "
                         "legitimate for reported (not selecting) analyses")
    ap.add_argument("--mask-dirname", default=MASK_DIRNAME)
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--upscale", type=int, default=UPSCALE)
    ap.add_argument("--n-chips", type=int, default=400)
    ap.add_argument("--n-pixels", type=int, default=50_000)
    ap.add_argument("--bg-per-road", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing fixture (it is meant to be frozen)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chips_json, pixels_npz = out_dir / CHIPS_JSON, out_dir / PIXELS_NPZ
    if chips_json.exists() and not args.force:
        raise SystemExit(
            f"{chips_json} already exists. The fixture is frozen on purpose — "
            "re-running it would silently move every arm's ground. Pass --force "
            "only if you intend to invalidate every extraction cache.")

    ds = Path(args.dataset_dir)
    all_chips = enumerate_chips(ds, args.split, args.crop)
    print(f"{len(all_chips)} candidate chips over "
          f"{all_chips['tile'].nunique()} {args.split} tiles")
    picked = sample_chips(all_chips, args.n_chips, args.seed)

    masks, road_px, valid = [], [], []
    for i, chip in enumerate(picked.to_dict("records")):
        m, r, v = read_chip_stats(ds, chip, args.mask_dirname, args.crop, args.upscale)
        masks.append(m)
        road_px.append(r)
        valid.append(v)
        if (i + 1) % 50 == 0:
            print(f"  read {i + 1}/{len(picked)} masks")
    picked["road_px"] = road_px
    picked["valid_frac"] = np.round(valid, 5)

    chip_idx, pix_idx, is_road = sample_pixels(
        masks, np.asarray(road_px), args.n_pixels, args.bg_per_road, args.seed + 1)

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "dataset_dir": str(ds),
        "split": args.split,
        "mask_source": "raster",
        "mask_dirname": args.mask_dirname,
        "crop": args.crop,
        "upscale": args.upscale,
        "hr_size": args.crop * args.upscale,
        "strata_col": STRATA_COL,
        "seed": args.seed,
        "n_chips": int(len(picked)),
        "n_pixels": int(chip_idx.size),
        "bg_per_road": args.bg_per_road,
    }
    chips_json.write_text(json.dumps(
        {"meta": meta, "chips": picked.to_dict("records")}, indent=1))
    np.savez_compressed(pixels_npz, chip_idx=chip_idx, pix_idx=pix_idx,
                        is_road=is_road)

    empties = int((picked["road_px"] == 0).sum())
    print(f"\nwrote {chips_json}")
    print(f"      {len(picked)} chips  "
          f"({', '.join(f'{k}:{v}' for k, v in picked['stratum'].value_counts().items())})")
    print(f"      {empties} road-free ({empties / len(picked):.1%}), "
          f"median road_px {int(picked['road_px'].median())}, "
          f"road fraction {picked['road_px'].sum() / (len(picked) * meta['hr_size'] ** 2):.4f}")
    print(f"wrote {pixels_npz}")
    print(f"      {int(is_road.sum())} road + {int((~is_road).sum())} background "
          f"= {chip_idx.size} pixels over {np.unique(chip_idx).size} chips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
