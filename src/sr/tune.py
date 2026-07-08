"""Optuna hyperparameter search for the joint SR + UNet model.

Mirrors :mod:`unet.tune` (same study/pruning/overlay conventions, scored on
best ``val_iou``) but drives ``JointSRUNetLightning`` + ``JointSRDataModule``
and searches the JOINT learning-rate pair: the UNet ``lr`` and the SR-net
``lr_sr`` are sampled independently (both log-uniform), with ``lr_sr``'s range
defaulting well below ``lr``'s — the lr_sr/lr ratio is the task-driven-SR
gradient scale (alpha), and "how small should lr_sr be" is exactly what this
search answers.

    python -m sr.tune \
        --base-config src/sr/configs/joint_sr.yaml \
        --base-config src/unet/configs/norm_stats.yaml \
        --dataset-dir /scratch/$USER/InstaRoad/ROSA_Dense_CDNGI \
        --n-trials 30 --max-epochs 8 \
        --out runs/sr_optuna

Best trial -> ``best_params.yaml`` overlay; full refit + test are then:

    python -m sr.cli fit  --config <base>... --config runs/sr_optuna/best_params.yaml
    python -m sr.cli test --config <base>... --config runs/sr_optuna/best_params.yaml \
        --ckpt_path checkpoints/unet_s2rosa_jointsr_best.ckpt

The encoder is NOT searched by default (encoder constancy across the R-series
and vs the baseline is the ablation's control); pass ``--encoders`` explicitly
to override.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import yaml

try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ImportError:  # pragma: no cover - fallback for older optuna
    from optuna.integration import PyTorchLightningPruningCallback

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

# Same imports the LightningCLI uses -- keep the search and the real fit identical.
from sentinel2data.dataset.joint_sr_dataset import JointSRDataModule
from sr.model import JointSRUNetLightning
from unet.tune import (
    MONITOR,
    MONITOR_MODE,
    _resolve_devices,
    load_base_config,
    resolve_encoder_weights,
)


def _data_kwargs(cfg: dict, dataset_dir: str | None, num_workers: int | None,
                 mask_source: str | None = None) -> dict:
    data = dict(cfg.get("data", {}))
    if mask_source:
        data["mask_source"] = mask_source
    if data.get("norm_mean") is None or data.get("norm_std") is None:
        raise SystemExit(
            "Base config has no frozen norm stats (the post-SR adapter needs them). "
            "Pass the norm-stats overlay too, e.g.\n"
            "  --base-config src/sr/configs/joint_sr.yaml "
            "--base-config src/unet/configs/norm_stats.yaml"
        )
    if dataset_dir:
        data["dataset_dir"] = dataset_dir
    if num_workers is not None:
        data["num_workers"] = num_workers
    return data


def build_objective(args, base_cfg: dict):
    data_cfg = _data_kwargs(base_cfg, args.dataset_dir, args.num_workers,
                            args.mask_source)
    model_cfg = dict(base_cfg.get("model", {}))
    bands = tuple(data_cfg.get("bands", (1, 2, 3, 4)))
    upscale = data_cfg.get("upscale", 4)
    devices = _resolve_devices(args.devices)
    encoder_weights = resolve_encoder_weights(base_cfg, args.encoder_weights)
    sen2sr_dir = args.sen2sr_dir or model_cfg.get("sen2sr_dir")
    upsampler = args.upsampler or model_cfg.get("upsampler", "sen2sr")
    # Bicubic (R0) has no learnable SR params, so lr_sr is a dead search
    # dimension -- skip it entirely rather than let TPE waste trials on it.
    search_lr_sr = upsampler == "sen2sr"
    if not search_lr_sr:
        print(f"[sr.tune] upsampler={upsampler!r}: lr_sr is not searched (no SR params).")

    def objective(trial: optuna.Trial) -> float:
        # --- search space: the joint LR pair is the star -------------------
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        lr_sr = (trial.suggest_float("lr_sr", args.lr_sr_min, args.lr_sr_max, log=True)
                 if search_lr_sr else args.lr_sr_min)  # bicubic ignores lr_sr
        pos_weight = trial.suggest_float("pos_weight", args.pos_weight_min, args.pos_weight_max)
        encoder_name = trial.suggest_categorical("encoder_name", args.encoders)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)

        pl.seed_everything(args.seed, workers=True)

        dm = JointSRDataModule(
            dataset_dir=data_cfg["dataset_dir"],
            bands=bands,
            batch_size=batch_size,
            num_workers=data_cfg.get("num_workers", 1),
            crop_size=data_cfg.get("crop_size", 128),
            upscale=upscale,
            image_size=data_cfg.get("image_size", 256),
            length=data_cfg.get("length"),
            normalize=data_cfg.get("normalize", True),
            norm_mean=data_cfg["norm_mean"],
            norm_std=data_cfg["norm_std"],
            min_road_density=data_cfg.get("min_road_density", 0.0),
            mask_source=data_cfg.get("mask_source", "graph"),
            mask_dirname=data_cfg.get("mask_dirname", "masks_osm_2pt5m"),
        )

        model = JointSRUNetLightning(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=len(bands),
            classes=model_cfg.get("classes", 1),
            lr=lr,
            pos_weight=pos_weight,
            bands=bands,
            image_size=data_cfg.get("image_size", 256),
            threshold=model_cfg.get("threshold", 0.5),
            normalize=data_cfg.get("normalize", True),
            norm_mean=data_cfg["norm_mean"],
            norm_std=data_cfg["norm_std"],
            upsampler=upsampler,
            sen2sr_dir=sen2sr_dir,
            lr_sr=lr_sr,
            freeze_sr=model_cfg.get("freeze_sr", False),
            upscale=upscale,
        )

        pruning_cb = PyTorchLightningPruningCallback(trial, monitor=MONITOR)
        callbacks = [pruning_cb]
        if args.patience > 0:
            callbacks.append(EarlyStopping(monitor=MONITOR, mode=MONITOR_MODE, patience=args.patience))

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=devices,
            strategy="auto",  # one GPU per process; see unet.tune._resolve_devices
            precision=args.precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=10,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=dm)
        pruning_cb.check_pruned()

        value = trainer.callback_metrics.get(MONITOR)
        if value is None:
            raise RuntimeError(f"'{MONITOR}' was never logged; cannot score the trial.")
        return float(value)

    return objective


def write_best_overlay(study: optuna.Study, out_dir: Path, encoder_weights,
                       upsampler: str) -> Path:
    p = study.best_params
    # Record the resolved upsampler so the refit is unambiguous (an R0 overlay
    # layered over joint_sr.yaml flips it back from sen2sr to bicubic).
    model_overlay = {
        "encoder_name": p["encoder_name"],
        "encoder_weights": encoder_weights,
        "upsampler": upsampler,
        "lr": p["lr"],
        "pos_weight": p["pos_weight"],
    }
    has_lr_sr = "lr_sr" in p  # absent for R0 (bicubic) searches
    if has_lr_sr:
        model_overlay["lr_sr"] = p["lr_sr"]
    overlay = {"model": model_overlay, "data": {"batch_size": p["batch_size"]}}
    overlay_path = out_dir / "best_params.yaml"
    alpha = (f"  alpha=lr_sr/lr={p['lr_sr'] / p['lr']:.2e}"
             if has_lr_sr else "  (bicubic R0: no lr_sr)")
    header = (
        "# Best hyperparameters from sr.tune (Optuna). Deep-merges over the base config:\n"
        f"#   python -m sr.cli fit --config src/sr/configs/joint_sr.yaml --config {overlay_path.name}\n"
        f"# best {MONITOR}={study.best_value:.4f}  trial #{study.best_trial.number}{alpha}\n"
    )
    with open(overlay_path, "w") as fh:
        fh.write(header)
        yaml.safe_dump(overlay, fh, sort_keys=False)

    with open(out_dir / "study_summary.json", "w") as fh:
        json.dump(
            {
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "best_params": p,
                "upsampler": upsampler,
                "alpha_lr_sr_over_lr": (p["lr_sr"] / p["lr"]) if has_lr_sr else None,
                "n_trials": len(study.trials),
                "monitor": MONITOR,
            },
            fh,
            indent=2,
        )
    return overlay_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Optuna search for the joint SR + UNet model.")
    ap.add_argument("--base-config", action="append", default=[], metavar="YAML",
                    help="Base Lightning config(s); repeat to layer (joint_sr.yaml then norm_stats.yaml).")
    ap.add_argument("--dataset-dir", default=None, help="Override data.dataset_dir from the base config.")
    ap.add_argument("--sen2sr-dir", default=None, help="Override model.sen2sr_dir from the base config.")
    ap.add_argument("--upsampler", default=None, choices=["sen2sr", "bicubic"],
                    help="Override model.upsampler. 'bicubic' = R0 baseline: lr_sr "
                         "is NOT searched (no SR params) and needs no SEN2SR weights.")
    ap.add_argument("--mask-source", default=None, choices=["graph", "raster"],
                    help="Override data.mask_source (graph = CDNGI, raster = OSM HR masks).")
    ap.add_argument("--out", default="runs/sr_optuna", help="Where to write best_params.yaml + study.")
    ap.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers (0 avoids GDAL forks).")

    # study controls
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--timeout", type=float, default=None, help="Wall-clock budget in seconds (optional).")
    ap.add_argument("--study-name", default="sr_optuna")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///runs/sr_optuna/study.db (enables resume).")
    ap.add_argument("--seed", type=int, default=42)

    # per-trial training budget
    ap.add_argument("--max-epochs", type=int, default=8, help="Short budget per trial; refit longer after.")
    ap.add_argument("--patience", type=int, default=3, help="EarlyStopping patience per trial (0=off).")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="1",
                    help="GPUs PER tuner process. Must be 1 (no DDP during search); "
                         "parallelise with one process per GPU sharing --storage.")
    ap.add_argument("--precision", default="bf16-mixed")

    # search space — the joint LR pair
    ap.add_argument("--lr-min", type=float, default=1e-5, help="UNet LR range (log-uniform)")
    ap.add_argument("--lr-max", type=float, default=1e-2)
    ap.add_argument("--lr-sr-min", type=float, default=1e-7,
                    help="SR-net LR range (log-uniform); default spans 'nearly frozen' to lr-scale")
    ap.add_argument("--lr-sr-max", type=float, default=1e-3)
    ap.add_argument("--pos-weight-min", type=float, default=1.0)
    ap.add_argument("--pos-weight-max", type=float, default=15.0)
    ap.add_argument("--encoders", nargs="+", default=["resnet34"],
                    help="Default: NOT searched (encoder constancy is the ablation control).")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8],
                    help="512px UNet stage is memory-heavy; keep small.")
    ap.add_argument("--encoder-weights", default=None,
                    help="'imagenet' or 'none'/'random'; default: base config's value.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.base_config:
        raise SystemExit("Pass at least one --base-config (joint_sr.yaml + norm stats).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config(args.base_config)
    objective = build_objective(args, base_cfg)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=args.storage is not None,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    encoder_weights = resolve_encoder_weights(base_cfg, args.encoder_weights)
    upsampler = args.upsampler or base_cfg.get("model", {}).get("upsampler", "sen2sr")
    overlay_path = write_best_overlay(study, out_dir, encoder_weights, upsampler)
    print(f"\nBest {MONITOR}={study.best_value:.4f} (trial #{study.best_trial.number})")
    print(f"Best params: {study.best_params}  encoder_weights={encoder_weights}")
    print(f"Wrote Lightning overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
