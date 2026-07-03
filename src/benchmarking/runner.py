"""Benchmark Runner: score a trained checkpoint over non-overlapping chips.

Loads a U-Net checkpoint, slides a NON-overlapping ``chip_size`` grid over each
split tile in its native pixels (no blending -- each chip is an independent
evaluation unit, the bootstrap/Wilcoxon pairing key), scores every chip with the
pure confusion-matrix + pixel-metric functions, and writes the two-table store
(``runs.parquet`` + ``chip_metrics.parquet``).

Normalisation reuses the checkpoint's own frozen train stats (``hparams``) so the
evaluation matches training exactly. Only the pixel metrics are filled; graph
columns (``apls`` etc.) are left out (added by a future graph runner).
"""
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from benchmarking.confusion_matrix import confusion_counts, pixel_metrics_from_counts
from benchmarking.store import append_chips, append_run
from sentinel2data.dataset.reading import apply_norm, read_window


def load_unet_predictor(checkpoint):
    """Load a ``UNetLightning`` checkpoint -> (eval-mode model, eval-config dict)."""
    from unet.model import UNetLightning  # lazy: pulls torch/lightning/smp

    model = UNetLightning.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    hp = model.hparams
    cfg = {
        "bands": list(hp.get("bands", (1, 2, 3))),
        "threshold": float(hp.get("threshold", 0.5)),
        "normalize": bool(hp.get("normalize", True)),
        "norm_mean": hp.get("norm_mean"),
        "norm_std": hp.get("norm_std"),
    }
    return model, cfg


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


def _accuracy(tp, fp, fn, tn):
    den = tp + fp + fn + tn
    return float("nan") if den == 0 else (tp + tn) / den


@torch.no_grad()
def _score_tile(model, device, img_path, mask_path, tile_id, chip_size, cfg):
    """Non-overlapping chip grid over one tile -> list of per-chip metric rows."""
    bands = cfg["bands"]
    rows = []
    with rasterio.open(img_path) as src, rasterio.open(mask_path) as msrc:
        height, width = src.height, src.width
        for ri, r0 in enumerate(range(0, height, chip_size)):
            for ci, c0 in enumerate(range(0, width, chip_size)):
                h = min(chip_size, height - r0)
                w = min(chip_size, width - c0)
                win = Window(c0, r0, w, h)
                img = read_window(src, bands, win)                        # (C, h, w)
                if cfg["normalize"]:
                    img = apply_norm(img, bands, cfg["norm_mean"], cfg["norm_std"])
                mask = (msrc.read(1, window=win) > 0).astype("int64")     # (h, w)

                x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                logits = model(x)                                          # (1, 1, h, w)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                counts = confusion_counts(
                    logits.cpu(), torch.from_numpy(mask),
                    threshold=cfg["threshold"], from_logits=True,
                )
                m = pixel_metrics_from_counts(counts)
                tp, fp = counts.tp[0].item(), counts.fp[0].item()
                fn, tn = counts.fn[0].item(), counts.tn[0].item()
                rows.append({
                    "chip_id": f"{tile_id}_r{ri}_c{ci}", "tile_id": tile_id,
                    "patch_row_id": ri, "patch_col_id": ci,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "iou": m["iou"][0].item(), "f1": m["f1"][0].item(),
                    "precision": m["precision"][0].item(), "recall": m["recall"][0].item(),
                    "accuracy": _accuracy(tp, fp, fn, tn),
                    "inference_ms": infer_ms,
                })
    return rows


def evaluate(dataset_dir, checkpoint, model_name, seed, store_dir, split="test",
             chip_size=256, config_yaml="", model="unet"):
    """Score a checkpoint over the split's non-overlapping chips -> the store.

    Returns the ``run_id``. Writes one ``runs.parquet`` row + ``n_chips`` rows to
    ``chip_metrics.parquet`` under ``store_dir``.
    """
    if model != "unet":
        raise ValueError(f"unsupported model {model!r} (only 'unet' for now)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = load_unet_predictor(checkpoint)
    net.to(device)

    df = _read_split_csv(dataset_dir, split)
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc)
    print(f"run {run_id[:8]} model={model_name} seed={seed} split={split} "
          f"chip={chip_size} tiles={len(df)} device={device}")

    chip_rows = []
    for _, r in df.iterrows():
        # tile_id = the tile's unique stem (e.g. "Mtubatuba_r0_c2"), NOT zone_name --
        # many tiles share a zone, and chip_id is derived from tile_id, so using
        # zone_name would collide chips across tiles of the same zone and break the
        # per-chip pairing in compare/report.
        chip_rows.extend(_score_tile(
            net, device,
            Path(dataset_dir) / r["image_path"], Path(dataset_dir) / r["mask_path"],
            Path(r["image_path"]).stem, chip_size, cfg,
        ))

    chips = pd.DataFrame(chip_rows)
    chips["model_name"] = model_name
    chips["seed"] = int(seed)
    chips["run_id"] = run_id

    run_row = {
        "run_id": run_id,
        "run_started_at": started,
        "run_finished_at": datetime.now(timezone.utc),
        "model_name": model_name,
        "config_hash": hashlib.sha256((config_yaml or run_id).encode()).hexdigest()[:12],
        "config_yaml": config_yaml,
        "seed": int(seed),
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "dataset_split": split,
        "chip_size": int(chip_size),
        "threshold": cfg["threshold"],
        "n_chips": int(len(chips)),
    }
    append_run(run_row, store_dir)
    append_chips(chips, store_dir)

    mean = chips[["iou", "f1", "precision", "recall"]].mean().round(4)
    print(f"scored {len(chips)} chips | mean iou={mean['iou']} f1={mean['f1']} "
          f"precision={mean['precision']} recall={mean['recall']}")
    print(f"wrote store -> {Path(store_dir) / 'runs.parquet'} + chip_metrics.parquet")
    return run_id
