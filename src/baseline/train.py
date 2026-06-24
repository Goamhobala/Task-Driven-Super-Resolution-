"""Train the UNet++ baseline on the combined Sentinel-2 COGs.

Reads the HPC layout directly:
    <data>/Imagery/{site}.tif      combined 14-band COGs
    <data>/mask_10m/{site}*.tif    binary road masks
    <data>/Data.npz                frozen per-band mean/std

The site-level split (train/val/test) comes from
`sentinel2data.processor.dataset`, so it matches benchmarking exactly. Logs to
Weights & Biases, early-stops on val loss, and writes the best checkpoint (with
the channel config baked in) to `--out`, which `benchmark.py` then loads.

Run:
    python -m baseline.train --data /scratch/.../InstaRoad --config M3 --out runs/baseline_m3
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from baseline.data import (
    CHANNEL_GROUPS,
    build_splits,
    compute_pos_weight,
    make_dataset,
    resolve_mask_suffix,
)
from baseline.model import RoadSegLoss, build_model


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="scratch root containing Imagery/, mask_10m/, Data.npz")
    ap.add_argument("--imagery", default=None, help="override Imagery dir")
    ap.add_argument("--masks", default=None, help="override mask dir")
    ap.add_argument("--stats", default=None, help="override Data.npz path")
    ap.add_argument("--config", default="M3", choices=list(CHANNEL_GROUPS), help="channel group")
    ap.add_argument("--out", required=True, help="output dir for checkpoint + metadata")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.3, help="clDice weight in the loss")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--wandb-project", default="instaroad-baseline")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--model-name", default=None, help="run/model name (default: baseline_<config>)")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Average loss + micro pixel metrics over a loader."""
    model.eval()
    total = 0.0
    tp = fp = fn = tn = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item()
        pred = (torch.sigmoid(logits) > 0.5).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
        tn += ((1 - pred) * (1 - y)).sum().item()
    eps = 1e-6
    metrics = {
        "val_loss": total / max(len(loader), 1),
        "val_iou": tp / (tp + fp + fn + eps),
        "val_f1": 2 * tp / (2 * tp + fp + fn + eps),
        "val_precision": tp / (tp + fp + eps),
        "val_recall": tp / (tp + fn + eps),
    }
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data = Path(args.data)
    imagery = Path(args.imagery) if args.imagery else data / "Imagery"
    masks = Path(args.masks) if args.masks else data / "mask_10m"
    stats = Path(args.stats) if args.stats else data / "Data.npz"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or f"baseline_{args.config}"

    in_channels = len(CHANNEL_GROUPS[args.config])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = build_splits(imagery)
    mask_suffix = resolve_mask_suffix(masks, splits["train"] + splits["val"])
    print(f"sites  train={splits['train']}\n       val={splits['val']}\n       test={splits['test']}")
    print(f"config={args.config} ({in_channels} ch)  mask_suffix={mask_suffix!r}  device={device}")

    train_ds = make_dataset(imagery, masks, splits["train"], stats, args.config,
                            args.patch_size, args.stride, mask_suffix)
    val_ds = make_dataset(imagery, masks, splits["val"], stats, args.config,
                          args.patch_size, args.patch_size, mask_suffix)
    print(f"patches  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    pos_weight = compute_pos_weight(masks, splits["train"], mask_suffix).to(device)
    print(f"pos_weight={pos_weight.item():.2f}")

    model = build_model(in_channels=in_channels, encoder=args.encoder).to(device)
    if torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    criterion = RoadSegLoss(pos_weight=pos_weight, alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args) | {"in_channels": in_channels, "channels": CHANNEL_GROUPS[args.config]}
    wandb.init(project=args.wandb_project, name=model_name, config=config, mode=args.wandb_mode)

    ckpt_path = out / "best.pth"
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / max(len(train_loader), 1)

        val = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch:>3}/{args.epochs} | train {avg_train:.4f} | "
              f"val {val['val_loss']:.4f} | f1 {val['val_f1']:.4f} | iou {val['val_iou']:.4f}")
        wandb.log({"epoch": epoch, "train_loss": avg_train, **val})

        if val["val_loss"] < best_val - args.min_delta:
            best_val = val["val_loss"]
            epochs_no_improve = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                "state_dict": state,
                "config": args.config,
                "in_channels": in_channels,
                "channels": CHANNEL_GROUPS[args.config],
                "encoder": args.encoder,
                "model_name": model_name,
                "seed": args.seed,
                "patch_size": args.patch_size,
                "best_val_loss": best_val,
            }, ckpt_path)
            print(f"  ✓ val improved -> {best_val:.4f}; saved {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  no improvement {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print(f"  early stopping at epoch {epoch}")
                break

    # Record where training landed so benchmark.py can pick it up unambiguously.
    (out / "train_meta.json").write_text(json.dumps({
        "model_name": model_name,
        "config": args.config,
        "in_channels": in_channels,
        "seed": args.seed,
        "best_val_loss": best_val,
        "checkpoint": str(ckpt_path),
        "imagery": str(imagery),
        "masks": str(masks),
        "stats": str(stats),
        "mask_suffix": mask_suffix,
    }, indent=2))
    wandb.finish()
    print(f"done. best val loss {best_val:.4f} at {ckpt_path}")


if __name__ == "__main__":
    main()
