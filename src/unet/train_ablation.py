"""Loss-ablation trainer (protocol Phases A-C) for the UNet baseline.

Reuses the exact building blocks the LightningCLI drives — ``UNetLightning``
and ``RoadDataModule`` — but hardened for *fair cross-arm comparison*:

  * --arm selects the loss (see ``unet.losses.build_loss``); the arm and its
    hyperparameters are ``UNetLightning`` hparams, so the checkpoint records
    them and ``benchmarking`` loads it like any other unet checkpoint.
  * FIXED epoch budget, NO early stopping — val loss is loss-dependent, so
    stopping/selecting on it would give each arm a different operating point.
  * Checkpoint selection on val F1 @ 0.5 (loss-independent, pre-registered).
    val_loss is still logged, never used for selection.
  * ``seed_everything`` so ``--seed`` is the only RNG knob; augmentation is the
    protocol-fixed D4 flip (``--no-augment`` to disable).
  * §4.5 skeleton warmup runs inside the model (``on_train_epoch_start``).
  * Ends with a val threshold sweep (θ = 0.05…0.95) written to ``sweep.json`` —
    the inference threshold is in scope for the protocol decision, and
    ``benchmarking.cli eval --threshold θ*`` scores the checkpoint at it.

Run (Phase A example):
    python -m unet.train_ablation \
        --base-config src/unet/configs/unet.yaml \
        --base-config src/unet/configs/norm_stats.yaml \
        --dataset-dir /scratch/$USER/InstaRoad/ROSA_Dense_CDNGI \
        --arm gap_ce --gap-r 5 --seed 0 --out runs/phase_a/gap_ce_r5_s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

# Same config plumbing + building blocks as unet.cli/unet.tune, so an ablation
# run is a drop-in for `python -m unet.cli fit` with the same base configs.
from sentinel2data.dataset.datasets import RoadDataModule
from unet.config_utils import (data_kwargs, load_base_config,
                               resolve_encoder_weights)
from unet.model import UNetLightning

THRESH_GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    # config / data — mirrors unet.tune
    ap.add_argument("--base-config", action="append", default=[], metavar="YAML",
                    help="Base Lightning config(s); repeat to layer "
                         "(unet.yaml then norm_stats.yaml).")
    ap.add_argument("--dataset-dir", default=None, help="Override data.dataset_dir.")
    ap.add_argument("--mask-dirname", default=None,
                    help="Override data.mask_dirname (label source); "
                         "''/'none' -> the CSVs' masks_raster.")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Override data.num_workers (0 avoids GDAL forks).")
    ap.add_argument("--out", required=True, help="Run dir (checkpoints + artifacts).")
    # protocol-fixed training budget
    ap.add_argument("--epochs", type=int, default=100, help="FIXED budget, no early stop.")
    ap.add_argument("--batch-size", type=int, default=None, help="Default: base config.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Protocol screening LR (Appendix B); same for every arm "
                         "thanks to the §4.4 scale normalization.")
    ap.add_argument("--encoder", default=None,
                    help="Override model.encoder_name (protocol: one fixed encoder).")
    ap.add_argument("--encoder-weights", default=None,
                    help="'imagenet' | 'none'/'random'. Default: base config.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="Protocol-fixed D4 flip augmentation on the train crops.")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--precision", default="bf16-mixed")
    # loss arm + hyperparameters (protocol §3.2/§4.3)
    ap.add_argument("--arm", required=True,
                    help="bce | gap_ce | tl_ce | bce_dice | pstar_dice | "
                         "pstar_tversky | focal_tversky | <base>+cldice | <base>+skelrec")
    ap.add_argument("--pstar", default="bce", help="pixel slot for pstar_* arms")
    ap.add_argument("--gap-r", type=int, default=4, help="GapLoss buffer radius (paper 9x9 => 4)")
    ap.add_argument("--gap-k", type=float, default=60.0, help="GapLoss K (paper: 60)")
    ap.add_argument("--tl-ell", type=int, default=5,
                    help="TL/T2/T4 filter length (paper: 5; also sizes the "
                         "curvature kernels for t2_ce/t4_ce)")
    ap.add_argument("--tl-theta", type=float, default=0.5,
                    help="TL/T2/T4 weight-map binarization threshold "
                         "(papers: 0.375; protocol grid {0.375, 0.5})")
    ap.add_argument("--tversky-alpha", type=float, default=0.7)
    ap.add_argument("--cl-alpha", type=float, default=0.3)
    ap.add_argument("--cl-iters", type=int, default=5)
    ap.add_argument("--sr-w", type=float, default=1.0)
    ap.add_argument("--sr-radius", type=int, default=1)
    ap.add_argument("--warmup-start", type=int, default=30)
    ap.add_argument("--warmup-ramp", type=int, default=10)
    # logging
    ap.add_argument("--wandb-project", default="instaroad-loss-ablation")
    ap.add_argument("--wandb-mode", default="online",
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--run-name", default=None)
    return ap.parse_args(argv)


def make_run_name(args) -> str:
    hp_bits = []
    tl_like = ("tl_ce", "t2_ce", "t4_ce")
    if args.arm.startswith("gap_ce") or args.pstar == "gap_ce":
        hp_bits.append(f"r{args.gap_r}")
    if args.arm.startswith(tl_like) or args.pstar in tl_like:
        hp_bits.append(f"l{args.tl_ell}")
        if args.tl_theta != 0.5:
            hp_bits.append(f"th{args.tl_theta}")
    return args.run_name or "_".join(
        [args.arm.replace("+", "-"), *hp_bits, f"s{args.seed}"])


@torch.no_grad()
def threshold_sweep(model, loader, device, thresholds=THRESH_GRID):
    """Pixel P/R/F1/IoU at each threshold over the val quadrants, one pass."""
    model.eval().float().to(device)
    acc = {t: [0.0, 0.0, 0.0] for t in thresholds}  # tp, fp, fn
    for images, masks, _ in loader:
        probs = torch.sigmoid(model(images.to(device)))
        y = (masks.to(device) > 0.5).float()
        for t in thresholds:
            pred = (probs > t).float()
            acc[t][0] += (pred * y).sum().item()
            acc[t][1] += (pred * (1 - y)).sum().item()
            acc[t][2] += ((1 - pred) * y).sum().item()
    eps = 1e-6
    return {t: {"f1": 2 * tp / (2 * tp + fp + fn + eps),
                "iou": tp / (tp + fp + fn + eps),
                "precision": tp / (tp + fp + eps),
                "recall": tp / (tp + fn + eps)}
            for t, (tp, fp, fn) in acc.items()}


def main(argv=None):
    args = parse_args(argv)
    if not args.base_config:
        raise SystemExit("Pass at least one --base-config (unet.yaml + norm_stats.yaml).")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_name = make_run_name(args)

    base_cfg = load_base_config(args.base_config)
    data_cfg = data_kwargs(base_cfg, args.dataset_dir, args.num_workers,
                           args.mask_dirname)
    model_cfg = dict(base_cfg.get("model", {}))
    bands = tuple(data_cfg["bands"])
    batch_size = args.batch_size or data_cfg.get("batch_size", 16)
    encoder = args.encoder or model_cfg.get("encoder_name", "resnet34")
    encoder_weights = resolve_encoder_weights(base_cfg, args.encoder_weights)

    pl.seed_everything(args.seed, workers=True)

    dm = RoadDataModule(
        dataset_dir=data_cfg["dataset_dir"],
        bands=bands,
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 0),
        image_size=data_cfg.get("image_size", 256),
        length=data_cfg.get("length"),
        normalize=data_cfg.get("normalize", True),
        norm_mean=data_cfg["norm_mean"],
        norm_std=data_cfg["norm_std"],
        mask_dirname=data_cfg.get("mask_dirname"),
        aug_flip=args.augment,      # protocol-fixed geometric augmentation
    )

    model = UNetLightning(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=len(bands),
        classes=model_cfg.get("classes", 1),
        lr=args.lr,
        bands=bands,
        image_size=data_cfg.get("image_size", 256),
        threshold=0.5,              # selection metric is val F1 @ 0.5 (fixed)
        normalize=data_cfg.get("normalize", True),
        norm_mean=data_cfg["norm_mean"],
        norm_std=data_cfg["norm_std"],
        loss_arm=args.arm,
        pstar=args.pstar,
        gap_r=args.gap_r,
        gap_k=args.gap_k,
        tl_ell=args.tl_ell,
        tl_theta=args.tl_theta,
        tversky_alpha=args.tversky_alpha,
        cl_alpha=args.cl_alpha,
        cl_iters=args.cl_iters,
        sr_w=args.sr_w,
        sr_radius=args.sr_radius,
        warmup_start=args.warmup_start,
        warmup_ramp=args.warmup_ramp,
    )

    resolved = {
        "run_name": run_name, "arm": args.arm, "seed": args.seed,
        "epochs": args.epochs, "batch_size": batch_size, "lr": args.lr,
        "encoder": encoder, "encoder_weights": encoder_weights,
        "bands": list(bands), "augment": args.augment,
        "dataset_dir": str(data_cfg["dataset_dir"]),
        "mask_dirname": data_cfg.get("mask_dirname"),
        "selection": "val_f1@0.5",
        "hp": {"pstar": args.pstar, "gap_r": args.gap_r, "gap_k": args.gap_k,
               "tl_ell": args.tl_ell, "tl_theta": args.tl_theta,
               "tversky_alpha": args.tversky_alpha,
               "cl_alpha": args.cl_alpha, "cl_iters": args.cl_iters,
               "sr_w": args.sr_w, "sr_radius": args.sr_radius,
               "warmup_start": args.warmup_start, "warmup_ramp": args.warmup_ramp},
    }
    # The resolved run config doubles as `benchmarking.cli eval --config-yaml`
    # input (-> config_hash in the runs table).
    (out / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    print(f"run={run_name}  arm={args.arm}  bands={bands}  "
          f"dataset={data_cfg['dataset_dir']}")

    logger = False
    if args.wandb_mode != "disabled":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=run_name,
                             save_dir=str(out), mode=args.wandb_mode)
        logger.log_hyperparams(resolved)

    ckpt_cb = ModelCheckpoint(
        dirpath=out / "checkpoints", filename="best_f1",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,      # fixed budget — deliberately NO EarlyStopping
        accelerator=args.accelerator,
        devices=1,                   # no DDP: keeps the sweep + seeding simple
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)

    if not ckpt_cb.best_model_path:
        raise SystemExit("no checkpoint was saved — did validation run?")
    best = torch.load(ckpt_cb.best_model_path, map_location="cpu",
                      weights_only=False)
    best_epoch = int(best["epoch"])
    best_f1 = float(ckpt_cb.best_model_score)
    model.load_state_dict(best["state_dict"])

    # ---- threshold sweep on the selected checkpoint (protocol: θ in scope) --
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sweep = threshold_sweep(model, dm.val_dataloader(), device)
    best_t = max(sweep, key=lambda t: sweep[t]["f1"])
    (out / "sweep.json").write_text(json.dumps(
        {"run": run_name, "best_epoch": best_epoch, "best_threshold": best_t,
         "sweep": {str(t): v for t, v in sweep.items()}}, indent=1))

    (out / "train_meta.json").write_text(json.dumps({
        "run_name": run_name, "arm": args.arm, "seed": args.seed,
        "bands": list(bands), "in_channels": len(bands),
        "best_val_f1": best_f1, "best_epoch": best_epoch,
        "best_threshold": best_t, "f1_at_best_threshold": sweep[best_t]["f1"],
        "checkpoint": str(Path(ckpt_cb.best_model_path).resolve()),
        "dataset_dir": str(data_cfg["dataset_dir"]),
        "mask_dirname": data_cfg.get("mask_dirname"),
        "hp": resolved["hp"],
    }, indent=2))

    if logger:
        logger.experiment.summary.update(
            {"best_val_f1": best_f1, "best_epoch": best_epoch,
             "best_threshold": best_t,
             "f1_at_best_threshold": sweep[best_t]["f1"],
             "iou_at_best_threshold": sweep[best_t]["iou"]})
        logger.experiment.finish()

    print(f"done. best val F1 {best_f1:.4f} @ epoch {best_epoch}; "
          f"θ*={best_t} (F1 {sweep[best_t]['f1']:.4f})")
    print(f"checkpoint: {ckpt_cb.best_model_path}")
    print("benchmark it:  python -m benchmarking.cli eval "
          f"--checkpoint {ckpt_cb.best_model_path} --model unet "
          f"--threshold {best_t} ...")


if __name__ == "__main__":
    main()
