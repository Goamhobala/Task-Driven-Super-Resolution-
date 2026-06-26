"""Dummy TerraTorch inference harness — for testing the benchmarking modules.

Runs OUT OF THE BOX with no real data and no model download (synthetic mode +
dummy model), so you can verify the plumbing of your modules immediately. Then
flip the flags in CONFIG to swap in a real TerraTorch model and your Indian
dataset — the rest of the pipeline doesn't change.

Pipeline exercised end to end:
    build_model -> Predictor(.predict -> probabilities) -> chip loop
    -> confusion_counts -> pixel_metrics_from_counts -> per-chip DataFrame

Terminology (yours): a *chip* is one subtile the model ingests; a *tile* is the
whole image. CHIP mode feeds pre-tiled chips; TILE mode feeds a whole tile and
lets TerraTorch's tiled_inference slide a window and stitch (see INFERENCE_MODE).

Requires: torch, numpy, pandas. The real-model paths additionally need terratorch.
Your module `pixel_metrics.py` must be importable (same folder).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# `stats` is the combined benchmarking surface — it re-exports the confusion
# matrix alongside the bootstrap / Wilcoxon / cross-seed analysis.
from stats import confusion_counts, pixel_metrics_from_counts

_HERE = Path(__file__).resolve().parent


# ============================ CONFIG — EDIT HERE ============================
class CONFIG:
    # ---- model ----
    # "dummy"                -> tiny untrained net, no download, runs anywhere
    # "terratorch_checkpoint"-> load a trained .ckpt (set CKPT_PATH + MODEL_ARGS)
    # "terratorch_scratch"   -> build EncoderDecoderFactory from a pretrained backbone
    MODEL_SOURCE = "dummy"
    # Identity of THIS evaluation run. `model_name` is the thing the stats module
    # pairs/groups on; `seed` distinguishes repeated runs of the same model. Both
    # are env-overridable so you can append several runs without editing the file:
    #     BENCH_MODEL=dummy_b BENCH_SEED=0 python dummy_pipeline.py
    MODEL_NAME = os.environ.get("BENCH_MODEL", "dummy_a")
    SEED = int(os.environ.get("BENCH_SEED", uuid.uuid4().int & 0x7FFFFFFF))
    CKPT_PATH = "/path/to/model.ckpt"            # for terratorch_checkpoint
    MODEL_ARGS = {                               # for both terratorch_* sources
        "backbone": "prithvi_eo_v2_300",
        "backbone_pretrained": True,
        "decoder": "UNetDecoder",
        "decoder_channels": [512, 256, 128, 64],
        "num_classes": 2,                        # 2 = (background, road); see NOTE below
        # ViTs need necks to become pyramidal for CNN decoders — adjust to your
        # backbone/decoder combo per the EncoderDecoderFactory guide:
        # "necks": [{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        #           {"name": "ReshapeTokensToImage"},
        #           {"name": "LearnedInterpolateToPyramidal"}],
    }

    # ---- data ----
    USE_SYNTHETIC_DATA = True          # True = random tensors, needs no files
    DATA_ROOT = "/path/to/indian/dataset"        # you hook this up
    SPLIT = "test"
    IN_CHANNELS = 14                   # bands the model expects (CapeTown.tif has 14)
    CHIP_SIZE = 224                    # H = W of a chip
    N_SYNTHETIC_CHIPS = 8              # synthetic mode only

    # ---- whole-tile (tile mode) ----
    # The real binary road mask. Sliding-window inference runs over a tile of its
    # size and the random prediction is scored against it. The image the model
    # sees is synthetic (the dummy model is random), so only the mask is loaded.
    # Defaults to the CapeTown sample in dummy_data/ so tile mode runs OOTB.
    TILE_MASK_PATH = str(_HERE / "dummy_data" / "CapeTown_mask.tif")

    # ---- inference ----
    INFERENCE_MODE = "tile"            # "chip" | "tile"
    THRESHOLD = 0.5
    FROM_LOGITS = True                 # model emits logits -> Predictor makes probs
    POSITIVE_CLASS_INDEX = 1           # which channel is "road" when num_classes > 1
    IGNORE_INDEX = None                # e.g. 255 if your masks use an ignore value
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- tile mode (sliding window) ----
    TILE_CROP = 224
    TILE_STRIDE = 192

    # ---- output ----
    SAVE_PARQUET = True
    APPEND_PARQUET = True              # read-concat-write so multiple runs accrue
    OUT_PARQUET = str(_HERE / "dummy_data" / "tile_metrics_dummy.parquet")
# ===========================================================================
#
# NOTE on num_classes: your pixel_metrics expects a binary road channel. The
# Predictor below handles BOTH conventions — 1 channel (sigmoid) or 2 channels
# (softmax, take POSITIVE_CLASS_INDEX) — so either trains fine. Just keep this
# consistent with how the checkpoint was trained.


@dataclass
class _ModelOutput:
    """Mimics terratorch.models.model.ModelOutput so the Predictor's `.output`
    unwrapping is exercised even in dummy mode."""
    output: torch.Tensor


class DummySegModel(nn.Module):
    """Tiny untrained segmentation net -> [B, num_classes, H, W] logits.
    Predictions are garbage by design; this tests module plumbing, not accuracy."""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1),
        )

    def forward(self, x, **kwargs):
        return _ModelOutput(output=self.net(x))


def build_model(cfg) -> nn.Module:
    """Return a model whose forward gives a ModelOutput-like object with `.output`."""
    if cfg.MODEL_SOURCE == "dummy":
        n_cls = cfg.MODEL_ARGS.get("num_classes", 1)
        return DummySegModel(cfg.IN_CHANNELS, num_classes=n_cls)

    if cfg.MODEL_SOURCE == "terratorch_checkpoint":
        from terratorch.tasks import SemanticSegmentationTask  # noqa: PLC0415
        return SemanticSegmentationTask.load_from_checkpoint(
            cfg.CKPT_PATH,
            model_factory="EncoderDecoderFactory",
            model_args=cfg.MODEL_ARGS,
        )

    if cfg.MODEL_SOURCE == "terratorch_scratch":
        # Pretrained backbone + (untrained) decoder head. No checkpoint needed —
        # this is the "whatever pretrained is available" path. Predictions won't
        # be good (decoder is random) but the format is real.
        from terratorch.tasks import SemanticSegmentationTask  # noqa: PLC0415
        return SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args=cfg.MODEL_ARGS,
        )

    raise ValueError(f"unknown MODEL_SOURCE: {cfg.MODEL_SOURCE}")


def logits_to_road_prob(logits: torch.Tensor, cfg) -> torch.Tensor:
    """Collapse model output to a road-probability map [B, H, W] in [0, 1].

    Handles both binary conventions and the `FROM_LOGITS` flag, so the chip
    Predictor and the whole-tile path apply identical post-processing:
        * `FROM_LOGITS=False`            -> already probabilities, pass through
        * 1 channel (or [B,H,W])         -> sigmoid
        * N channels                     -> softmax, take POSITIVE_CLASS_INDEX
    """
    if not cfg.FROM_LOGITS:
        probs = logits
    elif logits.dim() == 3 or logits.shape[1] == 1:            # single-channel binary
        probs = torch.sigmoid(logits)
    else:                                                      # multi-class -> road prob
        probs = torch.softmax(logits, dim=1)[:, cfg.POSITIVE_CLASS_INDEX]

    if probs.dim() == 4:                                       # [B,1,H,W] -> [B,H,W]
        probs = probs.squeeze(1)
    return probs


class Predictor:
    """Thin adapter over a TerraTorch (or dummy) model. Hides ModelOutput and
    the 1-vs-2-channel convention; exposes one thing: predict() -> probabilities.

    This is the seam your benchmarking depends on. Swap the model inside, the
    benchmarking code never changes.
    """

    def __init__(self, model: nn.Module, cfg):
        self.model = model.eval().to(cfg.DEVICE)
        self.cfg = cfg

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W] -> probabilities [B, H, W] in [0, 1], on CPU."""
        out = self.model(images.to(self.cfg.DEVICE))
        logits = out.output if hasattr(out, "output") else out
        return logits_to_road_prob(logits, self.cfg).cpu()


class ChipDataset(Dataset):
    """Yields (image[C,H,W], mask[H,W], chip_id). Synthetic by default; replace
    the marked section with your real loader (read the split CSV / metadata
    parquet, open the COGs, and normalise EXACTLY as in training)."""

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.USE_SYNTHETIC_DATA:
            self.chip_ids = [f"synthetic_{i:04d}" for i in range(cfg.N_SYNTHETIC_CHIPS)]
        else:
            # TODO: read DATA_ROOT/splits/{SPLIT}.csv -> chip ids + image/mask paths
            raise NotImplementedError("hook up your split index here")

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, i: int):
        cid = self.chip_ids[i]
        if self.cfg.USE_SYNTHETIC_DATA:
            img = torch.randn(self.cfg.IN_CHANNELS, self.cfg.CHIP_SIZE, self.cfg.CHIP_SIZE)
            mask = (torch.rand(self.cfg.CHIP_SIZE, self.cfg.CHIP_SIZE) > 0.9).long()  # ~10% road
            return img, mask, cid
        # TODO real path: load img (rasterio) -> normalise like training -> tensor;
        #                 load the 10m raster mask (or rasterise the graph at eval res).
        raise NotImplementedError("hook up real chip loading here")


def run_chip_inference(predictor: Predictor, loader: DataLoader, cfg) -> pd.DataFrame:
    """Predict per chip, score with the benchmarking modules, return per-chip rows."""
    rows = []
    for imgs, masks, chip_ids in loader:
        probs = predictor.predict(imgs)                         # [B,H,W] in [0,1], CPU
        counts = confusion_counts(
            probs, masks,
            threshold=cfg.THRESHOLD,
            from_logits=False,                                  # Predictor already applied sigmoid
            ignore_index=cfg.IGNORE_INDEX,
        )
        metrics = pixel_metrics_from_counts(counts)
        for j, cid in enumerate(chip_ids):
            rows.append({
                "chip_id": cid,                                 # unit of comparison for stats
                "tp": counts.tp[j].item(), "fp": counts.fp[j].item(),
                "fn": counts.fn[j].item(), "tn": counts.tn[j].item(),
                "iou": metrics["iou"][j].item(), "f1": metrics["f1"][j].item(),
                "precision": metrics["precision"][j].item(), "recall": metrics["recall"][j].item(),
            })
    return pd.DataFrame(rows)


def run_tile_inference(model: nn.Module, tile: torch.Tensor, cfg) -> torch.Tensor:
    """Whole-tile sliding-window inference via TerraTorch. `tile`: [B, C, H, W].
    Returns stitched logits [B, num_classes, H, W]. Needs terratorch installed."""
    from terratorch.tasks.tiled_inference import tiled_inference  # noqa: PLC0415

    def model_forward(x, **kw):
        out = model(x, **kw)
        return out.output if hasattr(out, "output") else out

    return tiled_inference(
        model_forward, tile,
        crop=cfg.TILE_CROP, stride=cfg.TILE_STRIDE,
        batch_size=cfg.BATCH_SIZE, verbose=True,
    )


def load_mask_and_dummy_image(cfg) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Read the real binary mask; synthesise a matching-size dummy image.

    Only the *mask* is real — it defines the ground truth and the tile dimensions
    the sliding window has to cover. The image the model "sees" is irrelevant to
    this demo (the model is random), so we hand it a random tensor of the right
    shape rather than wrangling the GeoTIFF's bands/normalisation. Swap this for a
    real loader when you plug in a real model.

    Returns (image[1,C,H,W] float32, mask[H,W] long, tile_id).
    """
    import rasterio  # noqa: PLC0415

    with rasterio.open(cfg.TILE_MASK_PATH) as ds:
        mask = ds.read(1)                                     # [H, W]
    h, w = mask.shape

    tile = torch.randn(1, cfg.IN_CHANNELS, h, w)             # content doesn't matter
    mask_t = torch.from_numpy(mask.astype(np.int64))         # [H, W]
    tile_id = Path(cfg.TILE_MASK_PATH).stem.replace("_mask", "")
    return tile, mask_t, tile_id


def score_in_chips(prob: torch.Tensor, mask: torch.Tensor, tile_stem: str, cfg) -> pd.DataFrame:
    """Dice a seamless whole-tile prediction + mask into a non-overlapping chip
    grid and score each chip independently. One row per chip.

    This is the seam between INFERENCE (whole-tile, overlapping, blended by
    tiled_inference) and EVALUATION UNITS (non-overlapping chips). The stats
    module resamples/pairs over these rows, so a chip is its bootstrap unit.

    Two keys per row:
      * `chip_id` = `{tile}_r{row}_c{col}` — the unique unit of evaluation, and a
        foreign key into the dataset's per-patch metadata catalogue.
      * `tile_id` = the parent image stem — denormalised so chips roll up to tiles
        (or resample over tiles) without joining the catalogue.
    Adjacent chips from one image are spatially autocorrelated; with a real
    multi-tile dataset you'd resample over `tile_id` instead of `chip_id`.
    """
    prob = prob.squeeze(0) if prob.dim() == 3 else prob       # [H, W]
    h, w = mask.shape
    chip = cfg.CHIP_SIZE

    rows = []
    for ri, r0 in enumerate(range(0, h, chip)):
        for ci, c0 in enumerate(range(0, w, chip)):
            r1, c1 = min(r0 + chip, h), min(c0 + chip, w)
            counts = confusion_counts(
                prob[r0:r1, c0:c1], mask[r0:r1, c0:c1],
                threshold=cfg.THRESHOLD,
                from_logits=False,                            # already probabilities
                ignore_index=cfg.IGNORE_INDEX,
            )
            m = pixel_metrics_from_counts(counts)
            rows.append({
                "chip_id": f"{tile_stem}_r{ri}_c{ci}",        # unit of comparison
                "tile_id": tile_stem,                          # parent image (FK)
                "patch_row_id": ri, "patch_col_id": ci,
                "tp": counts.tp[0].item(), "fp": counts.fp[0].item(),
                "fn": counts.fn[0].item(), "tn": counts.tn[0].item(),
                "iou": m["iou"][0].item(), "f1": m["f1"][0].item(),
                "precision": m["precision"][0].item(), "recall": m["recall"][0].item(),
            })
    return pd.DataFrame(rows)


def run_tile_eval(model: nn.Module, cfg) -> pd.DataFrame:
    """Whole-tile evaluation: slide a window over the tile, stitch the logits,
    then score the prediction against the real mask CHIP BY CHIP. One row per
    chip — the granularity the stats module needs to bootstrap/pair over."""
    tile, mask, tile_stem = load_mask_and_dummy_image(cfg)
    print(f"tile {tile_stem}: image {tuple(tile.shape)}, mask {tuple(mask.shape)}")

    logits = run_tile_inference(model, tile, cfg)              # [1, n_cls, H, W]
    probs = logits_to_road_prob(logits.cpu(), cfg)            # [1, H, W] in [0,1]

    df = score_in_chips(probs, mask, tile_stem, cfg)
    print(f"scored {len(df)} chips of {cfg.CHIP_SIZE}px")
    return df


def append_parquet(df: pd.DataFrame, path: str) -> None:
    """Parquet has no in-place append: read the existing table (if any), concat,
    rewrite. Fine at benchmarking scale; for high volume, write a directory of
    part-files and read them back as one dataset instead."""
    p = Path(path)
    if p.exists():
        df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
    df.to_parquet(p, index=False)


def _effective_seed(model_name: str, seed: int) -> int:
    """Fold model_name INTO the RNG seed so distinct models give distinct
    predictions even at the same `seed`, while the same model at different seeds
    still varies. Without this the dummy model is a pure function of the global
    seed, so `dummy_a` and `dummy_b` at seed 0 would be identical — degenerate
    (zero) paired differences that break the bootstrap/Wilcoxon demo."""
    h = int(hashlib.sha256(model_name.encode()).hexdigest(), 16) % 100_000
    return (seed * 100_003 + h) % (2**31)


def tag_run(df: pd.DataFrame, cfg, run_id: str) -> pd.DataFrame:
    """Stamp every per-chip row with its run identity (the columns the stats
    module filters/pairs on)."""
    df = df.copy()
    df["model_name"] = cfg.MODEL_NAME
    df["seed"] = cfg.SEED
    df["run_id"] = run_id
    return df


def main():
    cfg = CONFIG
    seed = _effective_seed(cfg.MODEL_NAME, cfg.SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_id = str(uuid.uuid4())
    print(f"run {run_id[:8]}  model={cfg.MODEL_NAME}  seed={cfg.SEED}")

    model = build_model(cfg)
    predictor = Predictor(model, cfg)

    if cfg.INFERENCE_MODE == "chip":
        loader = DataLoader(ChipDataset(cfg), batch_size=cfg.BATCH_SIZE, shuffle=False)
        df = run_chip_inference(predictor, loader, cfg)
    elif cfg.INFERENCE_MODE == "tile":
        df = run_tile_eval(model, cfg)
    else:
        raise ValueError(f"unknown INFERENCE_MODE: {cfg.INFERENCE_MODE}")

    df = tag_run(df, cfg, run_id)
    print("\nmean over chips:")
    print(df[["iou", "f1", "precision", "recall"]].mean().round(4).to_string())

    if cfg.SAVE_PARQUET:
        if cfg.APPEND_PARQUET:
            append_parquet(df, cfg.OUT_PARQUET)
        else:
            df.to_parquet(cfg.OUT_PARQUET, index=False)
        total = len(pd.read_parquet(cfg.OUT_PARQUET))
        print(f"\nwrote {len(df)} rows to {cfg.OUT_PARQUET} ({total} total)")


if __name__ == "__main__":
    main()