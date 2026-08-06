"""Benchmark Runner: score a trained checkpoint over footprint-aligned chips.

Model families (``--model``) are loaded through a small predictor registry:

  * ``unet``  UNetLightning. Reads image + mask at the same native (10 m)
              window; normalisation replays the checkpoint's frozen train
              stats from ``hparams``.
  * ``sr``    JointSRUNetLightning. Reads the native window as RAW DN (the
              forward normalises internally, after super-resolution) and is
              scored at the upscaled (2.5 m) resolution against an HR ground
              truth: ``mask_source="graph"`` rasterises the tile's
              ``masks_graph`` parquet, ``"raster"`` reads pre-generated HR
              mask COGs (``<split>/<mask_dirname>/``) — the exact helpers the
              training dataloader uses, so eval GT cannot drift from train GT.

**Footprint grid.** The evaluation chip is a ground-footprint cell of
``cell_m`` metres (default 2560 m = 256 px @ 10 m = 1024 px @ 2.5 m), derived
from each tile's transform. ``chip_id`` (``{tile_stem}_r{ri}_c{ci}``) therefore
names the same geography at every resolution, which is what lets the paired
stats line up chips across model families. Chips are non-overlapping — each is
an independent evaluation unit, the bootstrap/Wilcoxon pairing key.

**Edge chips.** Ragged cells at tile borders are padded up to a valid model
input (next /32 for unet, the full cell for sr — zero-pad, matching the SR
training loader) and the logits are cropped back to the true extent BEFORE
scoring, so counts only ever cover real pixels.

**Tile metric plugins.** Names passed via ``tile_metrics`` (see
``benchmarking.tile_metrics``) receive the stitched binary prediction + GT for
each tile and may return tile-level rows (-> ``tiles/`` shards) and per-chip
values (merged onto the chip rows). This is the seam the custom APLS
implementation drops into.

Writes the sharded store (``runs/ chips/ tiles/`` — see ``benchmarking.store``).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from rasterio.windows import Window

from benchmarking import strata
from benchmarking.confusion_matrix import confusion_counts, pixel_metrics_from_counts
from benchmarking.store import append_chips, append_run, append_tiles
from benchmarking.tile_metrics import resolve_tile_metrics

CELL_M_DEFAULT = 2560.0  # 256 px @ 10 m; 1024 px @ 2.5 m

MODEL_FAMILIES = ("unet", "sr")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _config_hash(config_yaml: str) -> str:
    """First 12 hex chars of the SHA-256 of the CANONICALISED config (parsed,
    keys sorted, re-serialised) so formatting/key-order changes don't alter the
    hash. Empty config -> empty hash (never a fake per-run value)."""
    if not config_yaml:
        return ""
    canon = json.dumps(yaml.safe_load(config_yaml), sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def _sync(device: str) -> None:
    """Make wall-clock timing honest on GPU (kernels launch async)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _grid(height: int, width: int, chip_px: int):
    """Non-overlapping footprint cells: (ri, ci, r0, c0, h, w), row-major."""
    cells = [
        (ri, ci, r0, c0, min(chip_px, height - r0), min(chip_px, width - c0))
        for ri, r0 in enumerate(range(0, height, chip_px))
        for ci, c0 in enumerate(range(0, width, chip_px))
    ]
    # Coverage invariant: cells must partition the tile exactly (cheap, always on).
    assert sum(h * w for *_, h, w in cells) == height * width, "grid does not partition tile"
    return cells


def _chip_px_from_transform(src, cell_m: float, chip_px: int | None, tile: str) -> int:
    """Ground-metre cell -> native pixels, from the raster transform."""
    if chip_px is not None:
        return int(chip_px)
    if src.crs is None or src.crs.is_geographic:
        raise ValueError(
            f"{tile}: CRS is missing or geographic — cannot derive the footprint "
            "grid from cell_m; pass an explicit chip_px."
        )
    px_size = abs(src.transform.a)
    n = round(cell_m / px_size)
    if n < 1:
        raise ValueError(f"{tile}: cell_m={cell_m} smaller than one {px_size} m pixel")
    return int(n)


def _pad_hw(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Pad (B, C, h, w) up to (target_h, target_w): reflect where the input is
    wide enough (better context than zeros), replicate for slivers."""
    ph, pw = target_h - x.shape[-2], target_w - x.shape[-1]
    if ph == 0 and pw == 0:
        return x
    mode = "reflect" if (ph < x.shape[-2] and pw < x.shape[-1]) else "replicate"
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode)


def _next_mult(n: int, k: int) -> int:
    return ((n + k - 1) // k) * k


def _accuracy(tp, fp, fn, tn):
    den = tp + fp + fn + tn
    return float("nan") if den == 0 else (tp + tn) / den


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


def _safe_run_id(model_name: str, seed, split: str, stratum: str | None) -> str:
    """Descriptive, filesystem-safe shard name (run_id IS the shard filename).

    model_name is free-form, so anything outside [A-Za-z0-9._-] is folded to
    '-'; without that a name containing '/' would silently write outside the
    store dir.
    """
    def slug(s) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in str(s))

    parts = [slug(model_name), f"seed{slug(seed)}", slug(split)]
    if stratum:
        parts.append(slug(stratum))
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


# --------------------------------------------------------------------------- #
# predictors — one per model family
# --------------------------------------------------------------------------- #
class UNetPredictor:
    """UNetLightning checkpoint; input and output at native resolution."""

    family = "unet"
    scale = 1

    def __init__(self, checkpoint, device: str):
        from unet.model import UNetLightning  # lazy: pulls torch/lightning/smp

        self.model = UNetLightning.load_from_checkpoint(checkpoint, map_location="cpu")
        self.model.eval().float().to(device)
        self.device = device
        hp = self.model.hparams
        self.bands = list(hp.get("bands", (1, 2, 3)))
        self.threshold = float(hp.get("threshold", 0.5))
        self.normalize = bool(hp.get("normalize", True))
        self.norm_mean = hp.get("norm_mean")
        self.norm_std = hp.get("norm_std")

    def read_chip(self, src, window) -> np.ndarray:
        """(C, h, w) float32, normalised exactly as at train time."""
        from sentinel2data.dataset.reading import apply_norm, read_window

        img = read_window(src, self.bands, window)
        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        return img

    def predict_probs(self, imgs: list[np.ndarray]) -> tuple[torch.Tensor, float]:
        """Same-sized chips -> (sigmoid probs (B, 1, h, w) on CPU, elapsed ms).

        Pads to the next /32 (smp-UNet's decoder constraint) and crops the
        logits back to the true extent, so scores never include pad pixels.
        """
        h, w = imgs[0].shape[-2:]
        x = torch.from_numpy(np.stack([np.ascontiguousarray(i) for i in imgs]))
        x = _pad_hw(x, max(_next_mult(h, 32), 32), max(_next_mult(w, 32), 32))
        x = x.to(self.device)
        _sync(self.device)
        t0 = time.perf_counter()
        logits = self.model(x)
        _sync(self.device)
        ms = (time.perf_counter() - t0) * 1000.0
        return torch.sigmoid(logits[..., :h, :w]).cpu(), ms


class SRPredictor:
    """JointSRUNetLightning checkpoint; raw-DN input, output at ``upscale`` x.

    SEN2SR variants have their LR input pinned by the shipped FFT mask
    (``_required_lr``, 128 px): a cell is run as a grid of pinned sub-windows
    whose HR outputs are stitched (non-overlapping — the training crop unit;
    ``sr_pad`` handles per-window border ringing inside forward). bicubic /
    sr4rs are fully convolutional and predict the whole cell in one pass.
    """

    family = "sr"

    def __init__(self, checkpoint, device: str, sen2sr_dir=None):
        from sr.model import JointSRUNetLightning  # lazy

        kwargs = {"map_location": "cpu"}
        if sen2sr_dir is not None:
            # hparams bake the TRAINING node's weights dir; override for eval.
            kwargs["sen2sr_dir"] = str(sen2sr_dir)
        self.model = JointSRUNetLightning.load_from_checkpoint(checkpoint, **kwargs)
        self.model.eval().float().to(device)
        self.device = device
        hp = self.model.hparams
        self.scale = int(hp.get("upscale", 4))
        self.bands = list(hp.get("bands", (1, 2, 3, 4)))
        self.threshold = float(hp.get("threshold", 0.5))
        self.required_lr = self.model._required_lr  # None = no pin

    def read_chip(self, src, window, cell_px: int) -> np.ndarray:
        """(C, cell_px, cell_px) RAW DN, zero-padded at tile edges — byte-for-
        byte the training loader's read (``_read_native``)."""
        from sentinel2data.dataset.joint_sr_dataset import _read_native

        return _read_native(src, self.bands, window, cell_px)

    def predict_probs(self, img: np.ndarray, cell_px: int) -> tuple[torch.Tensor, float]:
        """One padded cell -> (sigmoid probs (1, 1, s*cell, s*cell) CPU, ms)."""
        x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
        _sync(self.device)
        t0 = time.perf_counter()
        if self.required_lr is None:
            logits = self.model(x.unsqueeze(0))
        else:
            req = self.required_lr
            if cell_px % req != 0:
                raise ValueError(
                    f"cell of {cell_px} px is not a multiple of the SR model's "
                    f"pinned input ({req} px) — choose cell_m accordingly."
                )
            k = cell_px // req
            subs = torch.stack([
                x[:, r * req:(r + 1) * req, c * req:(c + 1) * req]
                for r in range(k) for c in range(k)
            ])  # (k*k, C, req, req)
            out = self.model(subs)  # (k*k, 1, s*req, s*req)
            sq = self.scale * req
            logits = out.new_empty((1, out.shape[1], self.scale * cell_px, self.scale * cell_px))
            for i in range(k * k):
                r, c = divmod(i, k)
                logits[0, :, r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = out[i]
        _sync(self.device)
        ms = (time.perf_counter() - t0) * 1000.0
        return torch.sigmoid(logits).cpu(), ms


def load_predictor(model: str, checkpoint, device: str, sen2sr_dir=None):
    if model == "unet":
        return UNetPredictor(checkpoint, device)
    if model == "sr":
        return SRPredictor(checkpoint, device, sen2sr_dir=sen2sr_dir)
    raise ValueError(f"unsupported model family {model!r} (choose from {MODEL_FAMILIES})")


# --------------------------------------------------------------------------- #
# SR ground truth (HR) — the training dataloader's own helpers
# --------------------------------------------------------------------------- #
class SRMaskReader:
    """HR (``scale`` x) ground truth for one tile, per window or whole-tile."""

    def __init__(self, dataset_dir, row, mask_source: str, mask_dirname: str, scale: int):
        if mask_source not in ("graph", "raster"):
            raise ValueError(f"sr mask_source must be graph|raster, got {mask_source!r}")
        self.dataset_dir = Path(dataset_dir)
        self.row = row
        self.mask_source = mask_source
        self.mask_dirname = mask_dirname
        self.scale = scale

    def window(self, src, win: Window, cell_px: int) -> np.ndarray:
        """(s*h, s*w) int64 for a (possibly edge-clipped) native window."""
        h, w = int(win.height), int(win.width)
        out_size = cell_px * self.scale
        if self.mask_source == "graph":
            from sentinel2data.dataset.upscale_dataset import _graph_mask

            m = _graph_mask(self.dataset_dir / self.row["mask_graph_path"], src,
                            win, out_size, self.scale)
        else:
            from sentinel2data.dataset.joint_sr_dataset import _read_raster_hr_mask

            m = _read_raster_hr_mask(self.dataset_dir, self.row, self.mask_dirname,
                                     win, out_size, self.scale)
        # The helpers zero-pad to the square cell; crop back to real pixels.
        return (m[: self.scale * h, : self.scale * w] > 0).astype("int64")

    def full_road_px(self, src) -> int:
        """Road-pixel count of the WHOLE tile's HR mask, read independently of
        the chip loop (the tp+fn invariant must not share the chips' code path)."""
        if self.mask_source == "raster":
            from sentinel2data.dataset.joint_sr_dataset import _hr_mask_path

            hr = _hr_mask_path(self.dataset_dir, self.row["image_path"], self.mask_dirname)
            with rasterio.open(hr) as m:
                return int((m.read(1) > 0).sum())
        return int(self._rasterize_full(src).sum())

    def _rasterize_full(self, src) -> np.ndarray:
        """Whole-tile HR rasterisation of the masks_graph parquet (non-square
        tiles supported, unlike the square-cell ``_graph_mask``)."""
        from affine import Affine
        from rasterio import features
        from sentinel2data.dataset.upscale_dataset import _BUFFER_COL, _load_graph

        out_shape = (src.height * self.scale, src.width * self.scale)
        roads = _load_graph(str(self.dataset_dir / self.row["mask_graph_path"]))
        if roads.empty:
            return np.zeros(out_shape, dtype="uint8")
        buffered = roads.geometry.buffer(roads[_BUFFER_COL].to_numpy(dtype="float64"))
        up_tf = src.transform * Affine.scale(1.0 / self.scale)
        return features.rasterize(
            ((g, 1) for g in buffered), out_shape=out_shape, transform=up_tf,
            fill=0, all_touched=True, dtype="uint8",
        )


# --------------------------------------------------------------------------- #
# per-tile scoring
# --------------------------------------------------------------------------- #
def _new_rows(sweep_thresholds):
    """Per-θ accumulator in sweep mode, flat list otherwise."""
    return {t: [] for t in sweep_thresholds} if sweep_thresholds is not None else []


def _accumulate(rows, out):
    """Extend the per-θ dict (sweep mode) or the flat list (normal mode)."""
    if isinstance(rows, dict):
        for t, rs in out.items():
            rows[t].extend(rs)
    else:
        rows.extend(out)


def _rows_at_threshold(metas, probs, target, masks, threshold, ms_per_chip,
                       canvases, scale):
    """Score one same-sized batch at ONE θ -> rows; optionally stitch the
    binary prediction + GT into the tile canvases for the tile-metric plugins."""
    counts = confusion_counts(probs, target, threshold=threshold, from_logits=False)
    metrics = pixel_metrics_from_counts(counts)
    pred_bin = (probs.squeeze(1).numpy() >= threshold)
    rows = []
    for i, (tile_id, ri, ci, r0, c0, h, w) in enumerate(metas):
        tp, fp = counts.tp[i].item(), counts.fp[i].item()
        fn, tn = counts.fn[i].item(), counts.tn[i].item()
        rows.append({
            "chip_id": f"{tile_id}_r{ri}_c{ci}", "tile_id": tile_id,
            "patch_row_id": ri, "patch_col_id": ci,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": metrics["iou"][i].item(), "f1": metrics["f1"][i].item(),
            "precision": metrics["precision"][i].item(),
            "recall": metrics["recall"][i].item(),
            "accuracy": _accuracy(tp, fp, fn, tn),
            "inference_ms": ms_per_chip,
        })
        if canvases is not None:
            pred_full, gt_full = canvases
            r, c = scale * r0, scale * c0
            sh, sw = scale * h, scale * w
            pred_full[r:r + sh, c:c + sw] = pred_bin[i]
            gt_full[r:r + sh, c:c + sw] = np.asarray(masks[i], dtype="uint8")
    return rows


def _batch_rows(metas, probs, masks, threshold, ms_per_chip, canvases, scale,
                sweep_thresholds=None):
    """Score one same-sized batch of chips -> rows.

    ``sweep_thresholds`` scores the SAME probs at every θ in the sequence and
    returns ``{θ: rows}`` — one forward pass, N binarisations, which is what
    lets a θ sweep cost one inference pass instead of one per θ. Canvases are
    never stitched in sweep mode: the tile-metric plugins are θ-dependent and
    would need one canvas per θ, so ``evaluate`` rejects the combination.
    """
    target = torch.from_numpy(np.stack(masks))
    if sweep_thresholds is not None:
        return {t: _rows_at_threshold(metas, probs, target, masks, t,
                                      ms_per_chip, None, scale)
                for t in sweep_thresholds}
    return _rows_at_threshold(metas, probs, target, masks, threshold,
                              ms_per_chip, canvases, scale)


def _score_tile_unet(pred: UNetPredictor, dataset_dir, row, cell_m, chip_px_opt,
                     batch_size, plugins, check_gt, sweep_thresholds=None):
    """Footprint cells over one tile, batching same-sized chips through the GPU."""
    tile_id = Path(row["image_path"]).stem
    img_path = Path(dataset_dir) / row["image_path"]
    mask_path = Path(dataset_dir) / row["mask_path"]
    rows, canvases, gt_road_px = _new_rows(sweep_thresholds), None, None
    with rasterio.open(img_path) as src, rasterio.open(mask_path) as msrc:
        chip_px = _chip_px_from_transform(src, cell_m, chip_px_opt, tile_id)
        gt_transform = src.transform
        H, W = src.height, src.width
        if plugins:
            canvases = (np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype="uint8"))
        buf = []  # (meta, img, mask) of identical (h, w)

        def flush():
            if not buf:
                return
            metas, imgs, masks = zip(*buf)
            probs, ms = pred.predict_probs(list(imgs))
            _accumulate(rows, _batch_rows(
                metas, probs, list(masks), pred.threshold,
                ms / len(buf), canvases, scale=1,
                sweep_thresholds=sweep_thresholds))
            buf.clear()

        for ri, ci, r0, c0, h, w in _grid(H, W, chip_px):
            win = Window(c0, r0, w, h)
            img = pred.read_chip(src, win)
            mask = (msrc.read(1, window=win) > 0).astype("int64")
            if buf and buf[-1][1].shape[-2:] != (h, w):
                flush()  # size change (edge row/col) -> new batch
            buf.append(((tile_id, ri, ci, r0, c0, h, w), img, mask))
            if len(buf) >= batch_size:
                flush()
        flush()

        if check_gt:
            gt_road_px = int((msrc.read(1) > 0).sum())
    return rows, canvases, chip_px, gt_road_px, gt_transform


def _score_tile_sr(pred: SRPredictor, dataset_dir, row, cell_m, chip_px_opt,
                   mask_source, mask_dirname, plugins, check_gt,
                   sweep_thresholds=None):
    """Footprint cells over one tile; predict at ``scale`` x, score against HR GT."""
    from affine import Affine

    tile_id = Path(row["image_path"]).stem
    img_path = Path(dataset_dir) / row["image_path"]
    s = pred.scale
    rows, canvases, gt_road_px = _new_rows(sweep_thresholds), None, None
    with rasterio.open(img_path) as src:
        chip_px = _chip_px_from_transform(src, cell_m, chip_px_opt, tile_id)
        gt_transform = src.transform * Affine.scale(1.0 / s)
        gt = SRMaskReader(dataset_dir, row, mask_source, mask_dirname, s)
        H, W = src.height, src.width
        if plugins:
            canvases = (np.zeros((H * s, W * s), dtype=bool),
                        np.zeros((H * s, W * s), dtype="uint8"))
        for ri, ci, r0, c0, h, w in _grid(H, W, chip_px):
            win = Window(c0, r0, w, h)
            img = pred.read_chip(src, win, chip_px)          # zero-padded cell
            probs, ms = pred.predict_probs(img, chip_px)     # (1,1,s*cell,s*cell)
            probs = probs[..., : s * h, : s * w]             # real pixels only
            mask = gt.window(src, win, chip_px)              # (s*h, s*w)
            _accumulate(rows, _batch_rows(
                [(tile_id, ri, ci, r0, c0, h, w)], probs, [mask],
                pred.threshold, ms, canvases, scale=s,
                sweep_thresholds=sweep_thresholds,
            ))
        if check_gt:
            gt_road_px = gt.full_road_px(src)
    return rows, canvases, chip_px, gt_road_px, gt_transform


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def evaluate(dataset_dir, checkpoint, model_name, seed, store_dir, split="test",
             model="unet", cell_m=CELL_M_DEFAULT, chip_px=None, batch_size=8,
             mask_source=None, mask_dirname=None, sen2sr_dir=None,
             config_yaml_path=None, exp_tag="", label_source="",
             tile_metrics=(), check="first", device=None, threshold=None,
             max_tiles=None, sweep_thresholds=None,
             stratum=None, stratum_col=None):
    """Score a checkpoint over the split's footprint chips -> the sharded store.

    ``check`` runs the tp+fn-vs-mask invariant on the ``first`` tile (default),
    ``all`` tiles, or ``off``. ``threshold`` overrides the checkpoint's
    binarisation threshold hparam (e.g. the θ* a loss-ablation run tuned on
    val — see ``unet.train_ablation``); the value used is recorded in the runs
    table either way. ``max_tiles`` scores only the first N tiles of the split
    (a quick local smoke; ``None`` = all tiles). Returns the ``run_id``.

    ``sweep_thresholds`` switches on SWEEP MODE: every θ in the sequence is
    scored off the same forward pass, so an N-point θ sweep costs one inference
    pass instead of N. It returns ``{θ: chips DataFrame}`` and writes NOTHING to
    the store — a sweep selects an operating point, it is not a benchmark run;
    the bench at the chosen θ* is a separate ``evaluate`` call. ``tile_metrics``
    are rejected in this mode because they are θ-dependent (one stitched canvas
    per θ), which would defeat the point. ``threshold`` is ignored when set.

    ``stratum`` restricts scoring to the tiles whose ``stratum_col`` (default
    ``urbanisation_classification``) equals it -- e.g. ``stratum="Urban"``.
    The value is matched case- and separator-insensitively and recorded on the
    run row, so a stratified run is self-describing. To slice a store that was
    already scored over the whole split, use ``benchmarking.cli report
    --stratum`` instead; it needs no re-inference. See ``benchmarking.strata``.
    """
    if model not in MODEL_FAMILIES:
        raise ValueError(f"unsupported model family {model!r} (choose from {MODEL_FAMILIES})")
    if check not in ("first", "all", "off"):
        raise ValueError(f"check must be first|all|off, got {check!r}")
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if sweep_thresholds is not None:
        sweep_thresholds = [float(t) for t in sweep_thresholds]
        if not sweep_thresholds:
            raise ValueError("sweep_thresholds must be a non-empty sequence")
        bad = [t for t in sweep_thresholds if not 0.0 < t < 1.0]
        if bad:
            raise ValueError(f"sweep thresholds must be in (0, 1), got {bad}")
        if tile_metrics:
            raise ValueError(
                "tile_metrics are θ-dependent and cannot be swept off one pass "
                f"(got {tuple(tile_metrics)}); sweep with tile_metrics=(), then "
                "bench once at θ* with them on")
    if model == "sr":
        mask_source = mask_source or "graph"
        mask_dirname = mask_dirname or "mask_osm_2pt5"
    else:
        mask_source = mask_source or "csv"
        if mask_source != "csv":
            raise ValueError(f"unet mask_source must be 'csv' (got {mask_source!r}); "
                             "use --mask-dirname to remap the CSV's mask dir")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = load_predictor(model, checkpoint, device, sen2sr_dir=sen2sr_dir)
    if threshold is not None:
        pred.threshold = float(threshold)
    plugins = resolve_tile_metrics(tile_metrics)

    df = _read_split_csv(dataset_dir, split)
    # Stratify BEFORE max_tiles, so --max-tiles N means "N tiles of this
    # stratum" rather than "whatever survives of the first N of the split".
    stratum_col = stratum_col or strata.DEFAULT_COL
    if stratum:
        df, stratum = strata.filter_split_df(df, split, stratum_col, stratum)
        print(f"stratum {stratum_col}={stratum}: {len(df)} tile(s)")
    if max_tiles is not None:
        df = df.head(int(max_tiles))
    if model == "unet" and mask_dirname:
        # Same remap the training datamodule applies (missing masks = error).
        from sentinel2data.dataset.datasets import _remap_mask_paths

        df = _remap_mask_paths(df, dataset_dir, mask_dirname)

    config_yaml = Path(config_yaml_path).read_text() if config_yaml_path else ""
    # run_id doubles as the shard FILENAME (store._write_shard), so it must stay
    # filesystem-safe; model_name is free-form. Two runs of the same
    # model/seed/split/stratum inside one second would collide, and the store
    # refuses to overwrite -- that is the intended guard, not a bug.
    run_id = _safe_run_id(model_name, seed, split, stratum)
    started = datetime.now(timezone.utc)
    print(f"run {run_id} family={model} cell_m={cell_m} tiles={len(df)} "
          f"device={device} mask_source={mask_source}"
          + (f" stratum={stratum} ({stratum_col})" if stratum else ""))

    chip_rows, tile_rows = _new_rows(sweep_thresholds), []
    chip_px_used = None
    with torch.inference_mode():
        for i, (_, row) in enumerate(df.iterrows()):
            # tile_id = the tile's unique stem (e.g. "Mtubatuba_r0_c2"), NOT
            # zone_name — many tiles share a zone and chip_id derives from
            # tile_id, so zone_name would collide chips across tiles.
            tile_id = Path(row["image_path"]).stem
            check_gt = check == "all" or (check == "first" and i == 0)
            if model == "unet":
                rows, canvases, chip_px_used, gt_road_px, gt_tf = _score_tile_unet(
                    pred, dataset_dir, row, cell_m, chip_px, batch_size, plugins,
                    check_gt, sweep_thresholds=sweep_thresholds)
            else:
                rows, canvases, chip_px_used, gt_road_px, gt_tf = _score_tile_sr(
                    pred, dataset_dir, row, cell_m, chip_px, mask_source,
                    mask_dirname, plugins, check_gt,
                    sweep_thresholds=sweep_thresholds)

            if gt_road_px is not None:
                # tp+fn is the GT road-pixel count, which is θ-independent, so
                # in sweep mode any one θ's rows prove the invariant for all.
                probe = next(iter(rows.values())) if isinstance(rows, dict) else rows
                got = sum(r["tp"] + r["fn"] for r in probe)
                if got != gt_road_px:
                    raise RuntimeError(
                        f"{tile_id}: invariant failed — chips' tp+fn = {got} but the "
                        f"tile mask has {gt_road_px} road px. Grid/window/GT bug."
                    )

            if plugins:
                pred_full, gt_full = canvases
                s = pred.scale
                grid_gt = [  # the footprint cells in GT pixels, for per-chip plugin values
                    (f"{tile_id}_r{ri}_c{ci}", ri, ci, s * r0, s * c0, s * h, s * w)
                    for ri, ci, r0, c0, h, w in _grid(
                        gt_full.shape[0] // s, gt_full.shape[1] // s, chip_px_used)
                ]
                tile_extra, per_chip = {}, {}
                for plugin in plugins:
                    res = plugin(pred_full, gt_full, transform=gt_tf,
                                 tile_id=tile_id, grid=grid_gt)
                    tile_extra.update(res.tile)
                    for cid, vals in (res.chips or {}).items():
                        per_chip.setdefault(cid, {}).update(vals)
                tile_rows.append({"tile_id": tile_id, **tile_extra})
                for r in rows:
                    r.update(per_chip.get(r["chip_id"], {}))

            _accumulate(chip_rows, rows)

    if sweep_thresholds is not None:
        # A sweep picks an operating point; it is not a benchmark run, so it
        # never touches the store (which is append-only and uuid-keyed — see
        # store._write_shard). Bench once at θ* with a separate evaluate call.
        out = {}
        for t, rs in chip_rows.items():
            c = pd.DataFrame(rs)
            c["model_name"] = model_name
            c["seed"] = int(seed)
            c["threshold"] = t
            out[t] = c
        n = len(next(iter(out.values())))
        print(f"swept {len(sweep_thresholds)} θ over {n} chips in ONE inference "
              f"pass | nothing written to the store")
        return out

    # GT resolution = native pixel size / family scale (NaN if non-metric CRS).
    with rasterio.open(Path(dataset_dir) / df.iloc[0]["image_path"]) as src:
        px = abs(src.transform.a) if src.crs and not src.crs.is_geographic else float("nan")
    gt_res_m = px / pred.scale

    chips = pd.DataFrame(chip_rows)
    chips["model_name"] = model_name
    chips["seed"] = int(seed)
    chips["run_id"] = run_id
    # Per-chip stratum, so a whole-split store can be sliced later without
    # re-reading the split CSV. Cheap, and makes the shard self-contained.
    try:
        chips["stratum"] = chips["tile_id"].astype(str).map(
            strata.tile_strata(dataset_dir, split, stratum_col))
    except (KeyError, FileNotFoundError):
        chips["stratum"] = ""      # split CSV has no such column — not fatal

    run_row = {
        "run_id": run_id,
        "run_started_at": started,
        "run_finished_at": datetime.now(timezone.utc),
        "model_name": model_name,
        "model_family": model,
        "exp_tag": exp_tag,
        "label_source": label_source,
        "mask_source": mask_source,
        "mask_dirname": mask_dirname or "",
        "config_hash": _config_hash(config_yaml),
        "config_yaml": config_yaml,
        "seed": int(seed),
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "dataset_split": split,
        # "" = the whole split. A stratified run is NOT pixel-comparable with a
        # whole-split run, so this is a comparability key in cli._GT_KEYS.
        "stratum": stratum or "",
        "stratum_col": (stratum_col if stratum else ""),
        "cell_m": float(cell_m),
        "chip_px": int(chip_px_used),
        "gt_res_m": float(gt_res_m),
        "threshold": pred.threshold,
        "batch_size": int(batch_size),
        "device": device,
        "tile_metrics": ",".join(tile_metrics),
        "n_tiles": int(len(df)),
        "n_chips": int(len(chips)),
    }
    append_run(run_row, store_dir)
    append_chips(chips, store_dir)
    if tile_rows:
        tiles = pd.DataFrame(tile_rows)
        tiles["model_name"] = model_name
        tiles["seed"] = int(seed)
        tiles["run_id"] = run_id
        append_tiles(tiles, store_dir)

    mean = chips[["iou", "f1", "precision", "recall"]].mean().round(4)
    print(f"scored {len(chips)} chips | mean iou={mean['iou']} f1={mean['f1']} "
          f"precision={mean['precision']} recall={mean['recall']}")
    print(f"wrote store -> {Path(store_dir)}/{{runs,chips"
          f"{',tiles' if tile_rows else ''}}}/{run_id}.parquet")
    return run_id
