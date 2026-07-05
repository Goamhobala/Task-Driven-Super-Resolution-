"""Train the R0/R1/R2 resolution-enhancement experiments (Lightning).

Reads the HPC layout directly:
    <data>/imagery/{site}.tif        combined 14-band 10 m COGs (only the
                                     [B4, B3, B2, B8] slice is used)
    <data>/mask_2pt5m/{site}*.tif    binary road masks on the 2.5 m grid
                                     (exactly 4x the imagery dims per site)
    <data>/Data.npz                  frozen per-band mean/std (adapter stats)

The site-level split comes from `sentinel2data.processor.dataset`, so it
matches the baseline and benchmarking exactly. Logs to Weights & Biases,
early-stops on val loss, checkpoints the best model to `--out`.

Experiments (one module, only the upsampler treatment varies):
    R0  bicubic x4 input upsampling (deterministic baseline)
    R1  SEN2SR frozen preprocessing
    R2  SEN2SR fine-tuned jointly via the segmentation loss alone
        (task-driven SR; differential LRs implement the Figure-1 alpha)

Run:
    python -m sr.train --data /scratch/.../InstaRoad --experiment R2 --out runs/sr_r2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from sr.data import (
    SR_CHANNELS,
    SRRoadSegDataset,
    build_splits,
    compute_pos_weight,
    loader_kwargs,
    resolve_mask_suffix,
)
from sr.module import LOSS_REGISTRY, JointSRSegModule
from sr.sen2sr_loader import download_sen2sr

# experiment id -> (upsampler, freeze_sr)
EXPERIMENTS = {
    "R0": ("bicubic", False),
    "R1": ("sen2sr", True),
    "R2": ("sen2sr", False),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="scratch root containing imagery/, mask_2pt5m/, Data.npz")
    ap.add_argument("--imagery", default=None, help="override imagery dir")
    ap.add_argument("--masks-hr", default=None, help="override 2.5m mask dir (default <data>/mask_2pt5m)")
    ap.add_argument("--stats", default=None, help="override Data.npz path")
    ap.add_argument("--out", required=True, help="output dir for checkpoints + metadata")
    ap.add_argument("--experiment", default="R2", choices=list(EXPERIMENTS))
    ap.add_argument("--sen2sr-dir", default=None,
                    help="mlstac SEN2SR model dir (default <data>/models/SEN2SRLite_RGBN)")
    ap.add_argument("--download-sen2sr", action="store_true",
                    help="fetch SEN2SR weights if missing (needs internet — run on a login node)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="512x512 U-Net stage is memory-heavy; 4 fits comfortably in 48GB at bf16")
    ap.add_argument("--lr-seg", type=float, default=1e-3, help="U-Net LR (baseline's default)")
    ap.add_argument("--lr-sr", type=float, default=1e-5,
                    help="SEN2SR LR; alpha = lr_sr/lr_seg. Keep well below lr-seg")
    ap.add_argument("--freeze-sr-steps", type=int, default=0,
                    help="hold the SR LR at 0 for this many steps (warm-up)")
    ap.add_argument("--sr-lr-ramp-steps", type=int, default=0,
                    help="after the hold, ramp the SR LR linearly to lr-sr over this many steps")
    ap.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    ap.add_argument("--loss", default="roadseg", choices=list(LOSS_REGISTRY))
    ap.add_argument("--alpha", type=float, default=0.3, help="clDice weight in RoadSegLoss")
    ap.add_argument("--patch-size", type=int, default=128,
                    help="10m patch; SEN2SR's shipped FFT mask pins this to 128")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--encoder", default="resnet34",
                    help="smp encoder for the baseline U-Net; keep constant across R0/R1/R2")
    ap.add_argument("--aug-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="D4 flips + rotations on the train split (default: on)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--precision", default="bf16-mixed",
                    help="Lightning precision (bf16-mixed recommended; 32-true for CPU debugging)")
    ap.add_argument("--accelerator", default="auto",
                    help="Lightning accelerator; pass 'cpu' to keep laptop debug runs light")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true",
                    help="torch deterministic algorithms (warn-only; some SR ops lack kernels)")
    ap.add_argument("--log-grad-norms-every", type=int, default=100,
                    help="log per-group grad norms every N steps (0 = off)")
    ap.add_argument("--wandb-project", default="instaroad-baseline")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--model-name", default=None, help="run/model name (default: sr_<experiment>)")
    return ap.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    data = Path(args.data)
    imagery = Path(args.imagery) if args.imagery else data / "imagery"
    masks_hr = Path(args.masks_hr) if args.masks_hr else data / "mask_2pt5m"
    stats = Path(args.stats) if args.stats else data / "Data.npz"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or f"sr_{args.experiment.lower()}"
    upsampler, freeze_sr = EXPERIMENTS[args.experiment]

    sen2sr_dir = Path(args.sen2sr_dir) if args.sen2sr_dir else data / "models" / "SEN2SRLite_RGBN"
    if upsampler == "sen2sr" and not (sen2sr_dir / "model.safetensor").exists():
        if args.download_sen2sr:
            download_sen2sr(sen2sr_dir)
        else:
            raise SystemExit(
                f"SEN2SR weights not found at {sen2sr_dir}. Run once with "
                f"--download-sen2sr on a node with internet, or pass --sen2sr-dir."
            )

    splits = build_splits(imagery)
    mask_suffix = resolve_mask_suffix(masks_hr, splits["train"] + splits["val"])
    print(f"sites  train={splits['train']}\n       val={splits['val']}\n       test={splits['test']}")
    print(f"experiment={args.experiment} ({upsampler}, freeze_sr={freeze_sr})  "
          f"mask_suffix={mask_suffix!r}")

    train_ds = SRRoadSegDataset(imagery, masks_hr, splits["train"],
                                patch_size=args.patch_size, stride=args.stride,
                                mask_suffix=mask_suffix, d4=args.aug_flip)
    val_ds = SRRoadSegDataset(imagery, masks_hr, splits["val"],
                              patch_size=args.patch_size, stride=args.patch_size,
                              mask_suffix=mask_suffix)
    print(f"patches  train={len(train_ds)}  val={len(val_ds)}  "
          f"augment={'flip' if args.aug_flip else 'off'}")

    kwargs = loader_kwargs(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    # pos_weight from the actual 2.5m road frequency over the train masks.
    pos_weight = compute_pos_weight(masks_hr, splits["train"], mask_suffix)
    print(f"pos_weight={pos_weight.item():.2f}")

    band_stats = np.load(stats)
    band_mean = tuple(float(v) for v in band_stats["mean"][SR_CHANNELS])
    band_std = tuple(float(v) for v in band_stats["std"][SR_CHANNELS])

    module = JointSRSegModule(
        upsampler=upsampler, sen2sr_dir=str(sen2sr_dir) if upsampler == "sen2sr" else None,
        band_mean=band_mean, band_std=band_std,
        encoder=args.encoder, loss=args.loss,
        pos_weight=pos_weight.item(), alpha=args.alpha,
        lr_sr=args.lr_sr, lr_seg=args.lr_seg,
        freeze_sr=freeze_sr, freeze_sr_steps=args.freeze_sr_steps,
        sr_lr_ramp_steps=args.sr_lr_ramp_steps, scheduler=args.scheduler,
        log_grad_norms_every=args.log_grad_norms_every,
    )

    logger = WandbLogger(project=args.wandb_project, name=model_name,
                         mode=args.wandb_mode, save_dir=str(out))
    logger.log_hyperparams(vars(args))
    ckpt_cb = ModelCheckpoint(dirpath=out, filename="best", monitor="val_loss",
                              mode="min", save_top_k=1, save_last=True)
    early_cb = EarlyStopping(monitor="val_loss", mode="min",
                             patience=args.patience, min_delta=args.min_delta)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator, devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb, early_cb],
        deterministic="warn" if args.deterministic else False,
        default_root_dir=str(out),
        log_every_n_steps=10,
    )
    trainer.fit(module, train_loader, val_loader)

    best_val = ckpt_cb.best_model_score
    # Record where training landed so the eval stage can pick it up unambiguously.
    (out / "train_meta.json").write_text(json.dumps({
        "model_name": model_name,
        "experiment": args.experiment,
        "upsampler": upsampler,
        "freeze_sr": freeze_sr,
        "encoder": args.encoder,
        "seed": args.seed,
        "lr_sr": args.lr_sr,
        "lr_seg": args.lr_seg,
        "best_val_loss": float(best_val) if best_val is not None else None,
        "checkpoint": ckpt_cb.best_model_path,
        "imagery": str(imagery),
        "masks_hr": str(masks_hr),
        "stats": str(stats),
        "mask_suffix": mask_suffix,
        "sen2sr_dir": str(sen2sr_dir) if upsampler == "sen2sr" else None,
        "patch_size": args.patch_size,
    }, indent=2))
    print(f"done. best val loss "
          f"{float(best_val) if best_val is not None else float('nan'):.4f} "
          f"at {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
