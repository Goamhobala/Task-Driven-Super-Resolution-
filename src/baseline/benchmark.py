"""Score a trained baseline checkpoint on the held-out test sites.

Loads `best.pth`, runs non-overlapping patch ("chip") inference over the test
sites, and writes one row per chip to a parquet — exactly the long-form table
`benchmarking.stats` consumes (model_name, seed, chip_id, tile_id, tp/fp/fn/tn,
iou/f1/precision/recall). Appends so repeated seeds accrue into one store for the
cross-seed / paired analyses.

Run:
    python -m baseline.benchmark --ckpt runs/baseline_m3/best.pth \
        --data /scratch/.../InstaRoad --out runs/baseline_m3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baseline.data import BenchDataset, build_splits, resolve_mask_suffix
from baseline.model import build_model
from benchmarking.confusion_matrix import confusion_counts, pixel_metrics_from_counts


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="best.pth written by train.py")
    ap.add_argument("--data", required=True, help="scratch root with Imagery/, mask_10m/, Data.npz")
    ap.add_argument("--imagery", default=None)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--out", required=True, help="dir for the metrics parquet")
    ap.add_argument("--parquet-name", default="tile_metrics.parquet")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    return ap.parse_args()


def append_parquet(df: pd.DataFrame, path: Path) -> None:
    """Read-concat-write (parquet has no in-place append)."""
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


@torch.no_grad()
def run(model, loader, device, threshold):
    rows = []
    for x, y, chip_ids, tile_ids in loader:
        logits = model(x.to(device))
        counts = confusion_counts(logits.cpu(), y, threshold=threshold, from_logits=True)
        metrics = pixel_metrics_from_counts(counts)
        for j in range(len(chip_ids)):
            rows.append({
                "chip_id": chip_ids[j], "tile_id": tile_ids[j],
                "tp": counts.tp[j].item(), "fp": counts.fp[j].item(),
                "fn": counts.fn[j].item(), "tn": counts.tn[j].item(),
                "iou": metrics["iou"][j].item(), "f1": metrics["f1"][j].item(),
                "precision": metrics["precision"][j].item(), "recall": metrics["recall"][j].item(),
            })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> None:
    """Print macro (mean over chips) and micro (count-pooled) test metrics."""
    print("\nper-tile macro f1 (mean over chips):")
    print(df.groupby("tile_id")["f1"].mean().round(4).to_string())

    macro = df[["iou", "f1", "precision", "recall"]].mean(skipna=True)
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    eps = 1e-9
    micro = {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }
    print("\ndataset-level metrics:")
    print(f"  macro  " + "  ".join(f"{k}={macro[k]:.4f}" for k in ("iou", "f1", "precision", "recall")))
    print(f"  micro  " + "  ".join(f"{k}={micro[k]:.4f}" for k in ("iou", "f1", "precision", "recall")))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(in_channels=ckpt["in_channels"], encoder=ckpt["encoder"],
                        encoder_weights=None).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = Path(args.data)
    imagery = Path(args.imagery) if args.imagery else data / "Imagery"
    masks = Path(args.masks) if args.masks else data / "mask_10m"
    stats = Path(args.stats) if args.stats else data / "Data.npz"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patch = ckpt.get("patch_size", 256)

    sites = build_splits(imagery)[args.split]
    mask_suffix = resolve_mask_suffix(masks, sites)
    print(f"model={ckpt['model_name']} seed={ckpt['seed']} config={config} "
          f"split={args.split} sites={sites}")

    # stride == patch -> non-overlapping chips, each scored exactly once.
    ds = BenchDataset(imagery, masks, sites, stats, config=config,
                      patch_size=patch, stride=patch, mask_suffix=mask_suffix)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"scoring {len(ds)} chips...")

    df = run(model, loader, device, args.threshold)
    df["model_name"] = ckpt["model_name"]
    df["seed"] = ckpt["seed"]

    summarise(df)

    parquet = out / args.parquet_name
    append_parquet(df, parquet)
    total = len(pd.read_parquet(parquet))
    print(f"\nwrote {len(df)} rows to {parquet} ({total} total)")


if __name__ == "__main__":
    main()
