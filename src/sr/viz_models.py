"""ONE grid per SCENE, ONE panel per model: the loss-arm contact sheet.

Sibling of `viz_tile` (one ckpt, one tile, separate full-res PNGs) and
`viz_grid` (one 128 px crop, the R-series SR columns). This one answers the
question those two cannot: *given a run directory holding N trained arms, how
do their predictions on the SAME ground differ?*

A **scene** is either a single tile (`--tiles`) or a whole ZONE mosaicked back
together from every tile of it in the split (`--zones`). The ROSA zones tile an
exact grid — 512 px at 10 m, 5120 m pitch, no overlap — so `<zone>_r{R}_c{C}`
reassembles losslessly by index, and a zone panel shows the model on the entire
scene rather than on whichever 5 km square happened to contain the interesting
part. Grid cells with no tile in the split are drawn as NO DATA and excluded
from every count; they are not "correctly empty" and must not be scored as it.

    <scene>_models_error.png

Reference panels lead every sheet, in this order: the ORIGINAL 10 m imagery at
its native sampling, the 2.5 m ground truth immediately after it, then the
bicubic x4 view — the same `BicubicUpsampler` an r0 arm actually consumes, so
the sheet shows what the network sees, not a prettier stand-in.

WHY THE PANELS ARE ERROR MAPS, NOT MASKS
----------------------------------------
At 20-odd panels a binary mask mostly encodes road DENSITY: the eye reads
"more white" as "more roads found" and cannot separate a recovered road from a
hallucinated one. Each panel is therefore the prediction scored against the
2.5 m ground truth: white TP, red FP, blue FN, black correctly empty, grey no
data. `--style mask` restores the plain binary rendering. A colour key is drawn
into the grid itself.

NOTHING IS EVER MATERIALISED AT MOSAIC SIZE
-------------------------------------------
A 5x5 zone is 10240 px square at 2.5 m; its bicubic RGB alone would be 1.7 GB.
Every panel is therefore composed BLOCK BY BLOCK — each member tile is scored
and reduced to its `2048/k` px block and pasted into the panel — so peak memory
is one tile's worth regardless of zone size. That is also why `k` is snapped to
a power of two dividing 2048: it must reduce a tile exactly, or blocks would
not abut.

WHICH THREE NUMBERS EACH PANEL CARRIES
--------------------------------------
`APLS`, `F1` and buffered F1 at rho = 3 px, all computed at FULL 2.5 m
resolution over the scene's valid area.

  * APLS is the connectivity measure: it skeletonises both masks into graphs
    and compares shortest paths, so it punishes a break in the middle of a
    road far more than the pixel metrics do. Two arms can share an F1 and
    differ sharply here, which is the whole reason for showing it. On a mosaic
    it is the MEAN OVER MEMBER TILES — the bench's own tile-level statistic —
    not one graph over the whole zone: the absent grid cells sever every road
    crossing them, so a zone-wide graph would score the holes, not the model.
  * Buffered F1 at rho = 3 px (7.5 m, about one lane either side) relaxes
    position: strict F1 scores a centreline one pixel off the label as an FP
    AND an FN, which at this GSD mostly measures label registration. The gap
    between F1 and bF1 reads as "how much of this arm's loss is misregistration
    rather than a missed road". F1 and bF1 pool exactly over member tiles.

EVERY MODEL IS SHOWN AT ITS OWN θ*
----------------------------------
θ* is loss-dependent by construction — a high-λ arm emits systematically
higher road probabilities — so rendering every arm at a common 0.5 would
confound calibration with segmentation quality. Each panel's θ comes from that
run's `sweep.json`, re-argmaxed from the recorded curve for `--select-on`,
because `best_threshold` records whichever criterion wrote the file last. A run
with no sweep.json is rendered at `--default-theta` and marked `*`.

APLS COMES FROM THE BENCH STORE WHEN THE STORE HAS IT
-----------------------------------------------------
`--store-dir` points at a benchmarking store. Any (run dir, θ, tile) whose
APLS was already scored there is READ rather than recomputed — it is the same
number, it is the one the thesis tables quote, and it is by far the most
expensive thing on this sheet (seconds per tile against milliseconds for the
pixel metrics). Matching is exact on the run directory named by the store's
`checkpoint_path` AND on θ, so a row scored at a different operating point can
never be silently substituted. Anything unmatched is computed locally and the
log says which panels those were.

Verified equal before being relied on: gap_ce seed1 on CapeTown r2_c2 scores
0.580307 in the store and 0.5803 recomputed here, off an identical GT graph
(4581 nodes / 7317 edges).

PROBABILITIES AND SCORES ARE CACHED PER TILE
--------------------------------------------
Inference writes `<cache>/<tile>__<run>.npy` (uint8) and scores land in
`<cache>/metrics.json`. Both are keyed by TILE, so a zone mosaic reuses
whatever single-tile work was already done, and re-rendering at a different θ,
style, layout or panel size costs no GPU at all.

    python -m sr.viz_models \
        --runs-dir <runs>/forViz \
        --dataset-dir <ROSA> --split test \
        --zones CapeTown_Fynbos_-33p96_18p61_Urban \
        --zones AzonalVegetation_-30p63_20p09_Rural \
        --zones AlexanderBay_Desert_-28p57_16p55_PeriUrban \
        --device mps --cols 6 --out-dir figures/loss_arms
"""
from __future__ import annotations

# MPS guard rails, set BEFORE torch initialises its Metal allocator. Same
# values as viz_sr: the Mac shares its GPU with the UI and an unbounded
# allocation swaps the whole machine instead of raising something catchable.
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import re
from pathlib import Path

import numpy as np

CKPT_NAME = "r2a_s2rosa_jointsr_final.ckpt"

# TP / FP / FN / background / no-data.
C_TP = (1.00, 1.00, 1.00)
C_FN = (0.25, 0.55, 1.00)
C_FP = (1.00, 0.27, 0.27)
C_BG = (0.00, 0.00, 0.00)
C_NA = (0.16, 0.16, 0.18)       # grid cell with no tile in this split

TILE_RC = re.compile(r"^(?P<zone>.+)_r(?P<r>\d+)_c(?P<c>\d+)$")


# ------------------------------------------------------------------ discovery
def run_key(run: str) -> str:
    """Run dir name as the bench store spells it (Finder's " copy" removed)."""
    return re.sub(r"\s+copy(\s+\d+)?$", "", run)


def short_name(run: str) -> str:
    """`sr_r2a_new_gap_ce_anorm_recalpost_seed66` -> `r2a_gap_ce_anorm_recalpost`.

    Drops the constant `sr_`/`_new` scaffolding and the seed suffix, which would
    otherwise spend a third of every panel title on characters identical across
    the figure — but KEEPS the R-series tag. Two arms from different series can
    carry the same loss name, and in a mixed --runs-dir dropping `r2a`/`r2b`
    would collide them into one label."""
    s = re.sub(r"^sr_(r\d+[a-z]?)_new_", r"\1_", run_key(run))
    s = re.sub(r"_holdout(_seed\d+)?$", "", s)
    return re.sub(r"_seed\d+$", "", s)


def find_ckpt(run: Path, pattern: str) -> Path | None:
    """The one checkpoint in `run` matching `pattern`, or None.

    A PATTERN rather than `sorted(glob("*.ckpt"))[0]`, which is what
    `_find_checkpoint` does and which picks `last.ckpt` over the final
    checkpoint in these directories — silently rendering the wrong epoch. It
    must still resolve to exactly ONE file: arms whose final checkpoint has
    been renamed per-arm (`r2a_s2rosa_jointsr_final.ckpt`) are matched by
    `*_s2rosa_jointsr_final.ckpt`, while a run holding several finals is an
    ambiguity the caller has to resolve rather than have guessed at."""
    hits = sorted((run / "checkpoints").glob(pattern)) if (run / "checkpoints").is_dir() else []
    if len(hits) > 1:
        raise SystemExit(
            f"{run.name}: {len(hits)} checkpoints match {pattern!r} "
            f"({', '.join(h.name for h in hits)}) — narrow --ckpt-name")
    return hits[0] if hits else None


def discover(runs_dir: Path, ckpt_name: str = CKPT_NAME) -> list[Path]:
    """Run dirs with exactly one `checkpoints/` file matching `ckpt_name`."""
    runs = [d for d in sorted(runs_dir.iterdir())
            if d.is_dir() and find_ckpt(d, ckpt_name)]
    return sorted(runs, key=lambda d: short_name(d.name))


def resolve_theta(run: Path, select_on: str, default: float):
    """(θ, label_suffix) for one run, from its sweep.json.

    Re-argmaxes the recorded curve rather than trusting `best_threshold`: that
    field records whichever criterion wrote the file LAST, so a figure built on
    it can sit at a different operating point from the table it illustrates.
    Both sweep schemas are accepted — the current one keys per-θ entries
    `iou`/`f1`/`buffered_f1_rN`, theta_sweep_bench's used `iou_mean`/`f1_mean`.
    """
    p = run / "sweep.json"
    if not p.is_file():
        return float(default), "no sweep"
    rec = json.loads(p.read_text())
    curve = rec.get("sweep", {})
    for key in (select_on, f"{select_on}_mean"):
        have = {t: v[key] for t, v in curve.items()
                if key in v and v[key] == v[key]}
        if have:
            best = max(have, key=lambda t: have[t])
            return float(best), f"{select_on}={have[best]:.3f}"
    bt = rec.get("best_threshold")
    if bt is not None:
        return float(bt), "recorded θ*"
    return float(default), "no sweep"


def parse_cells(spec: str | None):
    """'r2_c2,r3_c3' -> {(2, 2), (3, 3)}; None -> no restriction."""
    if not spec:
        return None
    out = set()
    for tok in spec.replace(",", " ").split():
        m = re.fullmatch(r"r(\d+)_c(\d+)", tok.strip())
        if not m:
            raise SystemExit(f"--cells: {tok!r} is not of the form rN_cN")
        out.add((int(m.group(1)), int(m.group(2))))
    return out


def zone_members(ds: Path, split: str, zone: str, cells=None):
    """{(row, col): tile_stem} for every tile of `zone` present in `split`.

    The ROSA zones tile an exact grid — verified 5120 m pitch against a 512 px
    / 10 m tile — so the `_r{R}_c{C}` index IS the mosaic position and no
    geotransform arithmetic is needed to place a block. Absent (r, c) are holes
    and are rendered as no-data rather than as empty ground."""
    d = ds / split / "imagery"
    out = {}
    for f in sorted(d.glob(f"{zone}_r*_c*.tif")):
        m = TILE_RC.match(f.stem)
        if m and m.group("zone") == zone:
            rc = (int(m.group("r")), int(m.group("c")))
            if cells is None or rc in cells:
                out[rc] = f.stem
    if not out:
        return out
    # Re-base to the selected block's own origin, so a --cells sub-mosaic is a
    # tight 2x2 rather than a 5x5 canvas that is mostly no-data.
    r0 = min(r for r, _ in out)
    c0 = min(c for _, c in out)
    return {(r - r0, c - c0): t for (r, c), t in out.items()}


def load_store_apls(store_dir: Path, split: str):
    """{(run_dir, θ, tile_id): apls} from a benchmarking store.

    Keyed on the run DIRECTORY the store's `checkpoint_path` points into and on
    θ, both exact. Matching on model_name would be wrong twice over: the store
    renames arms (`gapce_pstar_dice` is filed as `gap_ce_dice`), and several
    seeds of one arm share a model_name while being different trained models.
    Matching on θ as well means a row benched at another operating point can
    never be substituted for the one this figure renders."""
    import pandas as pd

    root = store_dir if (store_dir / "runs").is_dir() else store_dir / split
    if not (root / "runs").is_dir():
        raise SystemExit(f"no runs/ under {store_dir} or {store_dir / split}")
    runs = pd.concat([pd.read_parquet(f) for f in sorted((root / "runs").glob("*.parquet"))])
    tiles = pd.concat([pd.read_parquet(f) for f in sorted((root / "tiles").glob("*.parquet"))])
    if "apls" not in tiles.columns:
        raise SystemExit(f"{root}/tiles has no apls column (bench without tile_metrics=apls?)")

    meta = {r.run_id: (Path(r.checkpoint_path).parent.parent.name,
                       round(float(r.threshold), 4)) for r in runs.itertuples()}
    out = {}
    for t in tiles.itertuples():
        m = meta.get(t.run_id)
        if m is not None and t.apls == t.apls:
            out[(m[0], m[1], t.tile_id)] = float(t.apls)
    return out


# ------------------------------------------------------------------ inference
def read_tile(img_path, bands):
    """(C, H, W) RAW values — the training loader's own read, whole tile."""
    import rasterio
    from rasterio.windows import Window

    from sentinel2data.dataset.joint_sr_dataset import _read_native

    with rasterio.open(img_path) as src:
        w, h = src.width, src.height
        return _read_native(src, bands, Window(0, 0, w, h), max(w, h))


def predict_tile(model, img_chw, device, cell_px: int, want_sr: bool = False):
    """Whole-tile probability map at `upscale`x, scored in BENCH-SIZED windows.

    With `want_sr`, also returns the SR reflectance the UNet actually consumed
    — this ckpt's own SR net plus its `sr_pad` pad/crop, BEFORE the z-score
    adapter, which is the same quantity viz_single and viz_tile show as "the SR
    image". It is a second forward through `model.sr` rather than a byproduct
    of `model(x)`, because the end-to-end forward does not expose it.

    The window unit is the one `benchmarking.runner._score_tile_sr` used —
    `model._required_lr` for a pinned SR front-end, else the footprint cell —
    because convolution borders differ between a 256 px cell and a 512 px tile,
    so a single whole-tile pass would NOT reproduce the benchmarked
    prediction. Cells are stitched non-overlapping, as in the runner.
    """
    import torch

    up = int(model.hparams.upscale)
    req = model._required_lr                    # None = fully convolutional
    step = int(req) if req else int(cell_px)
    _, H, W = img_chw.shape
    if H % step or W % step:
        raise SystemExit(
            f"tile {H}x{W} is not a multiple of the model's window ({step} px)")

    prob = np.empty((H * up, W * up), dtype=np.float32)
    C = img_chw.shape[0]
    sr_out = np.empty((C, H * up, W * up), np.float32) if want_sr else None
    rs = float(getattr(model.hparams, "reflectance_scale", 1.0) or 1.0)
    pad = int(model.hparams.sr_pad)

    with torch.no_grad():
        for r in range(0, H, step):
            for c in range(0, W, step):
                x = torch.from_numpy(
                    np.ascontiguousarray(img_chw[:, r:r + step, c:c + step])
                )[None].float().to(device)
                p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                prob[r * up:(r + step) * up, c * up:(c + step) * up] = p
                if want_sr:
                    t_ref = x / rs
                    if pad:
                        t_ref = torch.nn.functional.pad(t_ref, (pad,) * 4,
                                                        mode="reflect")
                    hr = model.sr(t_ref)
                    if pad:
                        q = pad * up
                        hr = hr[..., q:-q, q:-q]
                    sr_out[:, r * up:(r + step) * up,
                           c * up:(c + step) * up] = hr[0].cpu().numpy()
                if device == "mps":
                    torch.mps.empty_cache()
    return (prob, sr_out) if want_sr else prob


def sr_tile(model, img_chw, device, cell_px: int) -> np.ndarray:
    """(C, H*up, W*up) SR reflectance only — the UNet forward is skipped.

    Same windows as `predict_tile`, because the SR net sees the same crops in
    the real pipeline and its border behaviour must match what the segmentation
    was actually fed."""
    import torch

    up = int(model.hparams.upscale)
    step = int(model._required_lr) if model._required_lr else int(cell_px)
    rs = float(getattr(model.hparams, "reflectance_scale", 1.0) or 1.0)
    pad = int(model.hparams.sr_pad)
    C, H, W = img_chw.shape
    out = np.empty((C, H * up, W * up), np.float32)
    with torch.no_grad():
        for r in range(0, H, step):
            for c in range(0, W, step):
                x = torch.from_numpy(
                    np.ascontiguousarray(img_chw[:, r:r + step, c:c + step])
                )[None].float().to(device) / rs
                if pad:
                    x = torch.nn.functional.pad(x, (pad,) * 4, mode="reflect")
                hr = model.sr(x)
                if pad:
                    q = pad * up
                    hr = hr[..., q:-q, q:-q]
                out[:, r * up:(r + step) * up,
                    c * up:(c + step) * up] = hr[0].cpu().numpy()
                if device == "mps":
                    torch.mps.empty_cache()
    return out


def load_sr_snapshot(model, run: Path, name: str) -> int | None:
    """Overwrite `model.sr` with a training snapshot's weights; returns its epoch.

    The snapshots in `<run>/sr_snapshots/` record the SR generator alone, every
    few epochs. Using one is how you see the FINISHED generator when the run's
    checkpoint is not from the end of training — but note that it then no longer
    pairs with the UNet in that checkpoint, so it is applied to the SR PANEL
    only and never to a prediction."""
    import torch

    p = run / "sr_snapshots" / name
    if not p.is_file():
        # A parameter-free upsampler (bicubic r0) has nothing to snapshot, and a
        # run may simply not have been configured to write them. Neither is an
        # error: fall back to the checkpoint's own SR, which for bicubic is the
        # same deterministic function anyway.
        print(f"    {short_name(run.name)}: no sr_snapshots/{name} "
              f"— SR panel comes from the checkpoint")
        return None
    snap = torch.load(p, map_location="cpu", weights_only=False)
    missing, unexpected = model.sr.load_state_dict(snap["sr_state_dict"],
                                                   strict=False)
    if missing or unexpected:
        print(f"    WARN snapshot load: {len(missing)} missing, "
              f"{len(unexpected)} unexpected tensors")
    return int(snap.get("epoch", -1))


def load_model(ckpt, sr_dir, device):
    import torch

    from sr.model import JointSRUNetLightning

    kwargs = {"map_location": "cpu"}
    if sr_dir is not None:
        # hparams bake the TRAINING node's weights dir; override for eval.
        kwargs["sen2sr_dir"] = str(sr_dir)
    m = JointSRUNetLightning.load_from_checkpoint(str(ckpt), **kwargs)
    # Some s2rosa ckpts saved reflectance_scale as None; forward divides by it.
    if getattr(m.hparams, "reflectance_scale", 10000.0) is None:
        m.hparams.reflectance_scale = 1.0
    del torch
    return m.eval().float().to(device)


def ckpt_read_meta(ckpt):
    """(bands, reflectance_scale, upscale) from hparams, no model built.

    All three matter to the REFERENCE panels, not just the model: the bands fix
    which arrays become RGB, and the divisor is what puts the panel in the
    reflectance units the stretch percentiles assume (10000 for the DN-era
    ckpts, 1.0 for the 0-1 COGs)."""
    import torch

    hp = torch.load(str(ckpt), map_location="cpu", weights_only=False,
                    mmap=True).get("hyper_parameters", {})
    return (list(hp.get("bands", (1, 2, 3, 4))),
            float(hp.get("reflectance_scale") or 1.0),
            int(hp.get("upscale", 4)))


# -------------------------------------------------------------------- display
def _blocks(a: np.ndarray, k: int) -> np.ndarray:
    """(h//k, k, w//k, k) view, trimming any partial trailing block."""
    h, w = a.shape
    return a[:h - h % k, :w - w % k].reshape(h // k, k, w // k, k)


def block_max(a: np.ndarray, k: int) -> np.ndarray:
    """Downsample a boolean array by MAX over kxk blocks.

    Used for the single-class panels (GT, `--style mask`), where max is the
    right reduction: area-averaging a road mask down to a thumbnail dissolves
    the 1-3 px centrelines, while max keeps a thin road visible at any panel
    size at the cost of thickening it."""
    return a if k <= 1 else _blocks(a, k).max(axis=(1, 3))


def block_count(a: np.ndarray, k: int) -> np.ndarray:
    """Per-block count of True, as float32."""
    if k <= 1:
        return a.astype(np.float32)
    return _blocks(a, k).sum(axis=(1, 3), dtype=np.int32).astype(np.float32)


def colourise(pred: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    """(h, w, 3) TP/FP/FN error map, reduced by `k` to the MAJORITY class.

    Per-class MAX would be wrong here even though it is right for a single
    mask: a road predicted one pixel off its GT centreline puts an FP and an
    FN beside every correct pixel, so `any FP -> red` repaints practically
    every true positive and the whole sheet reads as uniformly wrong. Counting
    the three classes per block and taking the argmax keeps a well-hit road
    white, a missed one blue and a hallucinated one red. Blocks with no
    prediction and no label stay background."""
    tp, fn, fp = pred & gt, (~pred) & gt, pred & (~gt)
    counts3 = np.stack([block_count(m, k) for m in (tp, fn, fp)])
    out = np.zeros(counts3.shape[1:] + (3,), dtype=np.float32)
    win = counts3.argmax(axis=0)
    any_road = counts3.sum(axis=0) > 0
    for i, colour in enumerate((C_TP, C_FN, C_FP)):
        out[any_road & (win == i)] = colour
    return out


def to_rgb(x, lo, hi, k: int = 1):
    """[B4,B3,B2,...] -> RGB in 0-1, area-downsampled by `k`.

    Imagery averages correctly (it is not a thin-structure mask), so this one
    uses a mean rather than `block_max`."""
    rgb = np.transpose(np.asarray(x)[:3], (1, 2, 0)).astype(np.float32)
    if k > 1:
        h, w, _ = rgb.shape
        rgb = rgb[:h - h % k, :w - w % k]
        rgb = rgb.reshape(h // k, k, w // k, k, 3).mean(axis=(1, 3))
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def bicubic_up(x: np.ndarray, up: int) -> np.ndarray:
    """(C, H*up, W*up) — the SAME upsample an r0 arm consumes.

    `sen2sr_loader.BicubicUpsampler`, antialias and all, rather than a
    hand-rolled resize: the reference panel is meant to show the network's
    actual input, and an antialias flag is exactly the kind of difference that
    would make it silently not that."""
    import torch

    from sr.sen2sr_loader import BicubicUpsampler

    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x))[None].float()
        return BicubicUpsampler(up)(t)[0].numpy()


def snap_k(hr_span: int, panel_px: int, tile_hr: int, upscale: int) -> int:
    """Reduction factor: a power of two that divides a tile exactly.

    Blocks are pasted per member tile, so `k` MUST divide `tile_hr` or adjacent
    blocks would not abut; and `k // upscale` must divide the LR tile for the
    10 m reference panel, hence k >= upscale. The smallest such k that brings
    the scene under `panel_px` wins."""
    k = upscale
    while k < tile_hr and hr_span // k > panel_px:
        k *= 2
    while tile_hr % k:                      # cannot happen for 2048, be safe
        k //= 2
    return max(k, upscale)


# --------------------------------------------------------------------- scoring
def tile_counts(pred: np.ndarray, gt: np.ndarray, buffer_px: float):
    """Poolable pixel evidence for one tile: strict counts + buffered halves.

    Buffered precision/recall are RATIOS, so they cannot be averaged over
    tiles — but their numerators and denominators pool exactly, which is what
    is returned here. That keeps a mosaic's bF1 identical to the number a
    single scoring pass over the whole zone would give."""
    from benchmarking.buffered_metrics import _within

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    # `_within` returns a MEAN over the mask's pixels; multiply back out so the
    # fractions pool. Both empty -> contributes nothing, as in buffered_scores.
    near_gt = (_within(pred, gt, buffer_px) * n_pred) if n_pred and n_gt else 0.0
    near_pred = (_within(gt, pred, buffer_px) * n_gt) if n_pred and n_gt else 0.0
    return dict(tp=tp, fp=fp, fn=fn, n_pred=n_pred, n_gt=n_gt,
                near_gt=near_gt, near_pred=near_pred)


def tile_apls(pred: np.ndarray, gt: np.ndarray, transform) -> float:
    from benchmarking.graph_metrics import apls_tile

    return float(apls_tile(pred, gt, transform=transform)["apls"])


def pool(rows: list[dict], aplss: list[float]):
    """(apls, f1, bf1) for a scene from its member tiles' evidence.

    F1 and bF1 pool exactly (micro over the scene's valid area). APLS is the
    MEAN of the per-tile scores, matching the bench's tile-level statistic —
    and deliberately not one graph over the mosaic, whose missing grid cells
    would sever every road crossing them and score the holes, not the model."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")

    n_pred = sum(r["n_pred"] for r in rows)
    n_gt = sum(r["n_gt"] for r in rows)
    prec = sum(r["near_gt"] for r in rows) / n_pred if n_pred else 0.0
    rec = sum(r["near_pred"] for r in rows) / n_gt if n_gt else 0.0
    bf1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    good = [a for a in aplss if a == a]
    return (float(np.mean(good)) if good else float("nan")), f1, bf1


# ------------------------------------------------------------------ the key
def draw_key(ax, args, k: int, has_holes: bool) -> None:
    """Draw the colour key into one grid cell.

    A key inside the grid rather than a line of prose in the title: at this
    panel count the reader meets the colours long before they finish reading a
    caption, and a swatch is unambiguous where "FN blue" is not."""
    from matplotlib.patches import Rectangle

    ax.set_axis_off()
    if args.style == "mask":
        rows = [(C_TP, "road", "sigmoid(logits) > θ*"),
                (C_BG, "—", "background")]
    else:
        rows = [(C_TP, "TP", "predicted road that IS road"),
                (C_FN, "FN", "road the model MISSED"),
                (C_FP, "FP", "predicted road that is NOT"),
                (C_BG, "—", "correctly empty")]
    if has_holes:
        rows.append((C_NA, "no data", "no tile in this split; unscored"))

    note = (f"blocks reduced {k}x to their majority class;\n"
            f"scores computed at full 2.5 m")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.98, "colour key", fontsize=9, fontweight="bold",
            va="top", transform=ax.transAxes)
    y = 0.88
    for colour, name, gloss in rows:
        ax.add_patch(Rectangle((0.03, y - 0.068), 0.13, 0.068, facecolor=colour,
                               edgecolor="0.55", linewidth=0.6,
                               transform=ax.transAxes))
        ax.text(0.20, y - 0.034, name, fontsize=8.5, fontweight="bold",
                va="center", transform=ax.transAxes)
        ax.text(0.40, y - 0.034, gloss, fontsize=7.2, va="center",
                transform=ax.transAxes)
        y -= 0.105
    ax.text(0.03, y - 0.02, note, fontsize=7, va="top", color="0.35",
            linespacing=1.5, transform=ax.transAxes)
    ax.text(0.03, y - 0.17,
            f"APLS  connectivity (graph shortest paths),\n"
            f"          mean over the scene's tiles\n"
            f"F1     strict pixel F1, pooled\n"
            f"bF1$_{{{args.buffer_px:g}}}$  F1 with a {args.buffer_px:g} px "
            f"(={2.5 * args.buffer_px:g} m) position\n"
            f"          tolerance, pooled",
            fontsize=7, va="top", color="0.2", linespacing=1.6,
            transform=ax.transAxes)


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True,
                    help="dir of run dirs, each with checkpoints/<--ckpt-name>")
    ap.add_argument("--dataset-dir", required=True, help="ROSA root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tiles", action="append", default=None,
                    help="tile stem, no .tif; one grid per tile (repeatable)")
    ap.add_argument("--zones", action="append", default=None,
                    help="zone prefix (a tile stem minus its _rN_cN); one grid "
                         "per zone, mosaicked from every tile of it in --split")
    ap.add_argument("--cells", default=None,
                    help="restrict a --zones mosaic to these grid cells, e.g. "
                         "'r2_c2,r2_c3,r3_c2,r3_c3'; the block is re-based to its "
                         "own origin so the panel is a tight 2x2, not a 5x5 canvas")
    ap.add_argument("--show-sr", action="store_true",
                    help="add a panel per arm showing the SR image that arm's "
                         "generator produces (what its UNet is fed)")
    ap.add_argument("--sr-snapshot", default=None,
                    help="take the SR weights for the SR PANEL from "
                         "<run>/sr_snapshots/<name> (e.g. epoch_099.pt) instead of "
                         "from the checkpoint. Use when the checkpoint is not from "
                         "the end of training. Never applied to a prediction — the "
                         "snapshot does not pair with the checkpoint's UNet.")
    ap.add_argument("--export-native", action="store_true",
                    help="also write per-tile PNGs at FULL 2.5 m resolution into "
                         "<out-dir>/native — the SR image each arm fed its UNet and "
                         "that arm's mask at θ*, one file per (tile, arm), plus the "
                         "GT once per tile. Borderless, one pixel per array element.")
    ap.add_argument("--mask-dirname", default="mask_new_2pt5")
    ap.add_argument("--ckpt-name", default=CKPT_NAME,
                    help="checkpoint filename or glob, resolved to exactly ONE file "
                         "per run dir (e.g. '*_s2rosa_jointsr_final.ckpt' when finals "
                         "have been renamed per arm)")
    ap.add_argument("--runs", action="append", default=None,
                    help="restrict to these run dir names (repeatable); "
                         "default = every run under --runs-dir")
    ap.add_argument("--sr-dir", default=None,
                    help="SR weights dir. REQUIRED for sen2sr/sr4rs ckpts (their "
                         "hparams bake the training node's path). Omit for bicubic r0.")
    ap.add_argument("--select-on", default="iou",
                    help="sweep.json criterion each θ* is re-argmaxed on")
    ap.add_argument("--default-theta", type=float, default=0.5,
                    help="θ for runs with no sweep.json; such panels are labelled")
    ap.add_argument("--theta", action="append", default=None,
                    help="pin one run's θ: --theta <run-or-short-name>=0.35 "
                         "(repeatable, overrides sweep.json)")
    ap.add_argument("--style", default="error", choices=("error", "mask"),
                    help="error = TP/FP/FN map vs GT (default); mask = plain binary")
    ap.add_argument("--cell-px", type=int, default=256,
                    help="window for UNPINNED upsamplers; 256 = the bench's 2560 m "
                         "footprint cell at 10 m. Ignored when the model pins its input.")
    ap.add_argument("--panel-px", type=int, default=768,
                    help="target panel edge; the true factor is snapped to a power "
                         "of two so member tiles reduce exactly")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--cache-dir", default=None,
                    help="probability + score cache (default <out-dir>/_probs), "
                         "keyed by TILE so zones reuse single-tile work")
    ap.add_argument("--refresh", action="store_true", help="ignore the prob cache")
    ap.add_argument("--store-dir", default=None,
                    help="benchmarking store to READ per-tile APLS from instead of "
                         "recomputing it; matched exactly on (run dir, θ, tile)")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute cached APLS/F1/bF1 evidence")
    ap.add_argument("--buffer-px", type=float, default=3.0,
                    help="buffered-F1 tolerance in PIXELS of the 2.5 m grid "
                         "(default 3 = 7.5 m, about one lane either side)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--stretch", type=float, nargs=2, default=(2, 98),
                    help="percentile stretch, from the ORIGINAL 10 m reflectance")
    ap.add_argument("--dry-run", action="store_true",
                    help="list scenes, runs, θ* and checkpoint health, then stop")
    args = ap.parse_args(argv)
    if not args.tiles and not args.zones:
        raise SystemExit("pass --tiles and/or --zones")

    runs_dir, ds, out = Path(args.runs_dir), Path(args.dataset_dir), Path(args.out_dir)
    runs = discover(runs_dir, args.ckpt_name)
    if args.runs:
        keep = set(args.runs)
        runs = [r for r in runs if r.name in keep or short_name(r.name) in keep]
    if not runs:
        raise SystemExit(f"no run dir under {runs_dir} has checkpoints/{args.ckpt_name}")

    # ---- scenes: a single tile is just a 1x1 mosaic -------------------------
    scenes = []
    for t in args.tiles or []:
        if not (ds / args.split / "imagery" / f"{t}.tif").is_file():
            raise SystemExit(f"tile not found: {t}")
        scenes.append({"name": t, "members": {(0, 0): t}, "rows": 1, "cols": 1})
    cells = parse_cells(args.cells)
    for z in args.zones or []:
        mem = zone_members(ds, args.split, z, cells)
        if not mem:
            raise SystemExit(f"no {args.split} tiles for zone {z}")
        scenes.append({"name": z, "members": mem,
                       "rows": max(r for r, _ in mem) + 1,
                       "cols": max(c for _, c in mem) + 1})

    pinned = {}
    for spec in args.theta or []:
        k, _, v = spec.partition("=")
        pinned[k] = float(v)

    thetas, notes = {}, {}
    for r in runs:
        if r.name in pinned or short_name(r.name) in pinned:
            thetas[r.name] = pinned.get(r.name, pinned.get(short_name(r.name)))
            notes[r.name] = "pinned"
        else:
            thetas[r.name], notes[r.name] = resolve_theta(
                r, args.select_on, args.default_theta)

    for sc in scenes:
        holes = sc["rows"] * sc["cols"] - len(sc["members"])
        print(f"scene {sc['name']}: {len(sc['members'])} tiles "
              f"({sc['rows']}x{sc['cols']} grid, {holes} empty cell(s))")
    print(f"{len(runs)} runs under {runs_dir}")
    for r in runs:
        ck = find_ckpt(r, args.ckpt_name)
        size = ck.stat().st_size
        # A truncated rsync leaves a readable file that only fails deep inside
        # torch.load; flag it here so one bad transfer does not abort the sheet
        # after twenty minutes of inference.
        flag = "  <- TRUNCATED?" if size < 100_000_000 else ""
        print(f"  {short_name(r.name):<26} θ={thetas[r.name]:<6} "
              f"[{notes[r.name]:<12}] {size / 1e6:7.1f} MB{flag}  {ck.name}")
    all_tiles = sorted({t for sc in scenes for t in sc["members"].values()})
    print(f"{len(all_tiles)} distinct tiles x {len(runs)} runs")
    if args.dry_run:
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import torch

    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else out / "_probs"
    cache.mkdir(parents=True, exist_ok=True)

    bands, refl, upscale = ckpt_read_meta(find_ckpt(runs[0], args.ckpt_name))

    # ---- inference: models OUTER so each ckpt loads once --------------------
    ok_runs = []
    for r in runs:
        ck = find_ckpt(r, args.ckpt_name)
        want = [t for t in all_tiles
                if args.refresh or not (cache / f"{t}__{r.name}.npy").is_file()]
        if not want:
            print(f"{short_name(r.name):<26} all {len(all_tiles)} tiles cached")
            ok_runs.append(r)
            continue
        try:
            model = load_model(ck, args.sr_dir, args.device)
        except Exception as e:                      # truncated / incompatible
            print(f"{short_name(r.name):<26} SKIPPED — {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:110]}")
            continue
        b = list(model.hparams.get("bands", (1, 2, 3, 4)))
        for i, t in enumerate(want, 1):
            img = read_tile(ds / args.split / "imagery" / f"{t}.tif", b)
            p = predict_tile(model, img, args.device, args.cell_px)
            np.save(cache / f"{t}__{r.name}.npy",
                    np.clip(p * 255.0 + 0.5, 0, 255).astype(np.uint8))
            print(f"{short_name(r.name):<26} [{i:>3}/{len(want)}] {t[-24:]:<24} "
                  f"mean={p.mean():.4f}")
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
        ok_runs.append(r)

    missing = [short_name(r.name) for r in runs if r not in ok_runs]
    if not ok_runs:
        raise SystemExit("no run produced a prediction — nothing to render")

    # ---- score cache: APLS costs seconds a tile, so it is memoised on disk --
    mfile = cache / "metrics.json"
    mcache = json.loads(mfile.read_text()) if mfile.is_file() else {}
    dirty = False
    store = load_store_apls(Path(args.store_dir), args.split) if args.store_dir else {}
    if store:
        print(f"store: {len(store)} per-tile APLS rows from {args.store_dir}")
    prov = {"store": 0, "local": 0}

    def tile_scores(tile, run, theta, pred, gt, transform):
        """Cached per-TILE evidence. Keyed by (tile, run, θ, buffer) so changing
        θ invalidates only what it should — and so a zone mosaic inherits every
        score a single-tile sheet already paid for.

        The pixel counts are always computed here (milliseconds). APLS is taken
        from the bench store when that exact (run dir, θ, tile) was scored
        there — same number, orders of magnitude cheaper, and it ties the panel
        to the row the tables quote."""
        nonlocal dirty
        key = f"{tile}|{run}|{theta:g}|{args.buffer_px:g}"
        # Entries written before the pooling rewrite are a bare [apls, f1, bf1]
        # list; F1/bF1 ratios cannot be pooled across a mosaic, so those are
        # recomputed rather than trusted.
        if args.rescore or not isinstance(mcache.get(key), dict):
            row = tile_counts(pred, gt, args.buffer_px)
            hit = store.get((run_key(run), round(theta, 4), tile))
            row["apls"] = hit if hit is not None else tile_apls(pred, gt, transform)
            row["apls_src"] = "store" if hit is not None else "local"
            mcache[key] = row
            dirty = True
        prov[mcache[key].get("apls_src", "local")] += 1
        return mcache[key]

    # ---- one figure per scene ----------------------------------------------
    for sc in scenes:
        prov.update(store=0, local=0)          # provenance is per-scene
        members, R, C = sc["members"], sc["rows"], sc["cols"]
        any_tile = next(iter(members.values()))
        with rasterio.open(ds / args.split / args.mask_dirname / f"{any_tile}.tif") as g:
            hr, transform = g.height, g.transform
        lr = hr // upscale
        k = snap_k(max(R, C) * hr, args.panel_px, hr, upscale)
        blk, blk_lr = hr // k, lr // (k // upscale)
        print(f"\n{sc['name']}: {R}x{C} grid, reduce {k}x -> "
              f"{max(R, C) * blk} px panels")

        # Read every member once: GT, and the 10 m reflectance for the stretch.
        gts, refs = {}, {}
        for rc, t in members.items():
            with rasterio.open(ds / args.split / args.mask_dirname / f"{t}.tif") as g:
                gts[rc] = g.read(1) > 0
            refs[rc] = read_tile(ds / args.split / "imagery" / f"{t}.tif",
                                 bands) / refl
        # ONE stretch for the whole scene, from the original 10 m reflectance:
        # per-panel autoscaling would make a model look brighter purely because
        # its histogram moved, and per-TILE autoscaling would seam the mosaic.
        lo, hi = np.percentile(np.concatenate([v[:3].ravel() for v in refs.values()]),
                               args.stretch)

        def canvas(rgb=True):
            a = np.zeros((R * blk, C * blk) + ((3,) if rgb else ()), np.float32)
            if rgb:
                a[:] = C_NA          # holes read as no-data, not as black ground
            return a

        def paste(a, rc, block):
            r, c = rc
            a[r * blk:(r + 1) * blk, c * blk:(c + 1) * blk] = block

        # --- reference panels ------------------------------------------------
        orig = np.zeros((R * blk_lr, C * blk_lr, 3), np.float32) + C_NA
        bic = canvas()
        gt_panel = canvas()
        road_px = tot_px = 0
        for rc, t in members.items():
            r, c = rc
            orig[r * blk_lr:(r + 1) * blk_lr, c * blk_lr:(c + 1) * blk_lr] = \
                to_rgb(refs[rc], lo, hi, k // upscale)
            paste(bic, rc, to_rgb(bicubic_up(refs[rc], upscale), lo, hi, k))
            m = block_max(gts[rc], k).astype(np.float32)
            paste(gt_panel, rc, np.repeat(m[:, :, None], 3, axis=2))
            road_px += int(gts[rc].sum())
            tot_px += gts[rc].size

        n_hole = R * C - len(members)
        area_km2 = tot_px * (abs(transform.a) ** 2) / 1e6
        panels = [
            (f"{args.split} imagery — ORIGINAL 10 m\n"
             f"{len(members)} tiles, {area_km2:.0f} km²", orig, "nearest"),
            (f"ground truth 2.5 m\nroad frac = {road_px / tot_px:.4f}",
             gt_panel, "nearest"),
            (f"bicubic x{upscale} -> 2.5 m\nthe UNet's input", bic, "antialiased"),
        ]

        # --- optional: the SR image each arm's generator produces -------------
        # Composed here rather than reused from the prediction pass because the
        # SR tensor is 4x the prediction's footprint per band; holding one per
        # arm across a mosaic is what the block-by-block design exists to avoid.
        sr_panels, sr_epoch, bic_arm = {}, {}, {}
        if args.show_sr or args.export_native:
            for r_ in ok_runs:
                model = load_model(find_ckpt(r_, args.ckpt_name), args.sr_dir,
                                   args.device)
                bic_arm[r_.name] = str(model.hparams.get("upsampler")) == "bicubic"
                if args.sr_snapshot:
                    e = load_sr_snapshot(model, r_, args.sr_snapshot)
                    if e is not None:
                        sr_epoch[r_.name] = e
                b = list(model.hparams.get("bands", (1, 2, 3, 4)))
                panel = canvas()
                for rc, t in sorted(members.items()):
                    img = read_tile(ds / args.split / "imagery" / f"{t}.tif", b)
                    sr = sr_tile(model, img, args.device, args.cell_px)
                    paste(panel, rc, to_rgb(sr, lo, hi, k))
                    if args.export_native:
                        nat = out / "native"
                        nat.mkdir(parents=True, exist_ok=True)
                        e = sr_epoch.get(r_.name)
                        tag = f"_e{e}" if e is not None else ""
                        plt.imsave(nat / f"{t}__{short_name(r_.name)}{tag}_sr.png",
                                   to_rgb(sr, lo, hi))
                    print(f"    sr {t[-24:]:<24} {short_name(r_.name):<24} {sr.shape[1:]}")
                sr_panels[r_.name] = panel
                del model
                if args.device == "mps":
                    torch.mps.empty_cache()


        # --- one panel per arm, composed block by block ----------------------
        for r_ in ok_runs:
            th = thetas[r_.name]
            img = canvas()
            rows, aplss = [], []
            for rc, t in members.items():
                p = np.load(cache / f"{t}__{r_.name}.npy")
                pred = (p.astype(np.float32) / 255.0) > th
                row = tile_scores(t, r_.name, th, pred, gts[rc], transform)
                rows.append(row)
                aplss.append(row["apls"])
                block = (np.repeat(block_max(pred, k).astype(np.float32)[:, :, None],
                                   3, axis=2) if args.style == "mask"
                         else colourise(pred, gts[rc], k))
                paste(img, rc, block)
            apls, f1, bf1 = pool(rows, aplss)
            mark = "*" if notes[r_.name] == "no sweep" else ""
            if args.show_sr and r_.name in sr_panels:
                e = sr_epoch.get(r_.name)
                up = "bicubic x4" if bic_arm.get(r_.name) else "SR output"
                panels.append(
                    (f"{short_name(r_.name)}\n{up}"
                     + (f"  (snapshot e={e})" if e is not None
                        else "  (from ckpt)"),
                     sr_panels[r_.name], "antialiased"))
            panels.append(
                (f"{short_name(r_.name)}\n"
                 f"θ={th:g}{mark}    APLS={apls:.3f}\n"
                 f"F1={f1:.3f}    bF1$_{{{args.buffer_px:g}}}$={bf1:.3f}",
                 img, "nearest"))
            src = {rw.get("apls_src", "local") for rw in rows}
            print(f"  {short_name(r_.name):<26} APLS={apls:.3f} "
                  f"F1={f1:.3f} bF1={bf1:.3f}   apls<-{'+'.join(sorted(src))}")
            if dirty:
                mfile.write_text(json.dumps(mcache, indent=0, sort_keys=True))
                dirty = False

        ncol = args.cols
        # One spare cell holds the colour key; ask for a row only if the panels
        # do not already leave a gap.
        nrow = -(-(len(panels) + 1) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.95 * nrow),
                                 facecolor="white")
        axes = np.atleast_1d(axes).ravel()
        for ax, (title, arr, interp) in zip(axes, panels):
            ax.imshow(arr, interpolation=interp)
            ax.set_title(title, fontsize=7.2, linespacing=1.4)
            ax.set_axis_off()
        for ax in axes[len(panels):]:
            ax.set_axis_off()
        draw_key(axes[len(panels)], args, k, n_hole > 0)

        # --- optional: full-resolution per-tile PNGs --------------------------
        # The mosaic panel exists to compare arms; at 8x reduction over 600 km²
        # it cannot also serve as a look at what the model produced. These are
        # the un-reduced arrays, borderless (one pixel per element), so they
        # drop into a document at native 2.5 m.
        if args.export_native:
            nat = out / "native"
            nat.mkdir(parents=True, exist_ok=True)
            for rc, t in sorted(members.items()):
                plt.imsave(nat / f"{t}_gt.png", gts[rc], cmap="gray",
                           vmin=0, vmax=1)
                plt.imsave(nat / f"{t}_rgb10m.png", to_rgb(refs[rc], lo, hi))
            # SR PNGs were written by the SR pass above; masks come off the
            # cached probabilities, so this costs no further inference.
            for r_ in ok_runs:
                arm, th = short_name(r_.name), thetas[r_.name]
                for rc, t in sorted(members.items()):
                    pr = np.load(cache / f"{t}__{r_.name}.npy").astype(np.float32) / 255.0
                    plt.imsave(nat / f"{t}__{arm}_pred.png",
                               (pr > th).astype(np.float32), cmap="gray",
                               vmin=0, vmax=1)
            print(f"  -> {nat}/  ({len(members)} tiles x {len(ok_runs)} arms, "
                  f"+ gt & 10 m rgb per tile)")

        sub = (f"{sc['name']}   —   {len(ok_runs)} arms, each at its own θ* "
               f"(argmax {args.select_on} on that run's val sweep)")
        if R * C > 1:
            sub += (f"\nwhole zone mosaicked from {len(members)} {args.split} "
                    f"tiles on a {R}x{C} grid"
                    + (f"; {n_hole} cell(s) have no tile in this split "
                       f"(grey, unscored)" if n_hole else ""))
        if any(notes[r.name] == "no sweep" for r in ok_runs):
            sub += f"\n*no sweep.json — rendered at θ={args.default_theta:g}"
        if store:
            sub += (f"\nAPLS read from the bench store for {prov['store']} "
                    f"tile-scores; {prov['local']} computed here")
        if missing:
            sub += f"\nnot rendered: {', '.join(missing)}"
        fig.suptitle(sub, fontsize=10.5, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.6, w_pad=0.6)
        p = out / f"{sc['name']}_models_{args.style}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  -> {p}")

    if dirty:
        mfile.write_text(json.dumps(mcache, indent=0, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
