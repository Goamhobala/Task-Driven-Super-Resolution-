"""Optuna hyperparameter finetuning for the UNet road model.

Reuses the exact same building blocks the LightningCLI drives -- ``UNetLightning``
and ``RoadDataModule`` -- so a tuned run is a drop-in for ``python -m unet.cli fit``.
Each Optuna trial trains a short run and is scored on the best ``val_iou``; trials
are pruned early on ``val_iou`` via the median pruner.

    # search, seeding the fixed args from the canonical config(s)
    python -m unet.tune \
        --base-config src/unet/configs/unet.yaml \
        --base-config src/unet/configs/norm_stats.yaml \
        --dataset-dir /scratch/$USER/InstaRoad/S2ROSA_V2 \
        --n-trials 30 --max-epochs 8 \
        --out runs/unet_optuna

The best trial is written as a Lightning config overlay (``best_params.yaml``) that
deep-merges over the base config, so the full-length refit + test are just:

    python -m unet.cli fit  --config <base>... --config runs/unet_optuna/best_params.yaml
    python -m unet.cli test --config <base>... --config runs/unet_optuna/best_params.yaml \
        --ckpt_path checkpoints/unet_s2rosa_best.ckpt

Only ``data:`` / ``model:`` keys that the search actually varies are written to the
overlay; everything else (bands, norm stats, image_size, ...) stays in the base
config as the single source of truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna
import yaml

# Substrings marking a concurrent-first-init race on a shared SQLite study
# (two workers running create_all / creating the study row at the same time).
_STUDY_RACE = ("already exists", "database is locked", "database is busy")


def create_study_shared(study_name, storage, seed):
    """optuna.create_study that tolerates N workers racing on a fresh SQLite DB.

    ``load_if_exists`` only guards the study NAME; the crash (``table studies
    already exists``) happens earlier, while a second worker runs the schema DDL
    the first is still creating. Retrying attaches once it exists. Also raises
    SQLite's busy timeout so concurrent trial writes wait instead of erroring."""
    delay = 0.5
    for attempt in range(12):
        try:
            store = storage
            if storage and str(storage).startswith("sqlite"):
                from optuna.storages import RDBStorage
                store = RDBStorage(url=str(storage),
                                   engine_kwargs={"connect_args": {"timeout": 60}})
            return optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=store,
                load_if_exists=storage is not None,
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
            )
        except Exception as e:  # noqa: BLE001 - only retry the known init race
            if attempt < 11 and any(s in str(e).lower() for s in _STUDY_RACE):
                time.sleep(delay)
                delay = min(delay * 1.6, 8.0)
                continue
            raise

# PyTorchLightningPruningCallback moved from `optuna.integration` to the separate
# `optuna-integration` package (optuna>=3.5). Support both so this works whichever
# is installed in the scratch venv.
try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ImportError:  # pragma: no cover - fallback for older optuna
    from optuna.integration import PyTorchLightningPruningCallback

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

# Same imports the LightningCLI uses -- keep the search and the real fit identical.
from sentinel2data.dataset.datasets import RoadDataModule
from unet.model import UNetLightning

MONITOR = "val_iou"          # maximise per-crop IoU on the val quadrants
MONITOR_MODE = "max"


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base`` (overlay wins), like Lightning."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_base_config(paths: list[str]) -> dict:
    """Deep-merge one or more YAML configs the same way ``--config a --config b`` does."""
    merged: dict = {}
    for p in paths:
        with open(p) as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    return merged


def _data_kwargs(cfg: dict, dataset_dir: str | None, num_workers: int | None,
                 mask_dirname: str | None = None) -> dict:
    """Pull the fixed RoadDataModule args from the base config (search overrides later)."""
    data = dict(cfg.get("data", {}))
    if data.get("norm_mean") is None or data.get("norm_std") is None:
        raise SystemExit(
            "Base config has no frozen norm stats. Pass the norm-stats overlay too, e.g.\n"
            "  --base-config src/unet/configs/unet.yaml "
            "--base-config src/unet/configs/norm_stats.yaml"
        )
    if dataset_dir:
        data["dataset_dir"] = dataset_dir
    if num_workers is not None:
        data["num_workers"] = num_workers
    if mask_dirname is not None:
        # Label-source override: ""/"none"/"null" -> CDNGI (masks_raster);
        # a dir name (e.g. mask_osm_10) -> the alternative masks beside it.
        data["mask_dirname"] = (None if str(mask_dirname).lower() in {"", "none", "null"}
                                else mask_dirname)
    return data


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def _resolve_devices(v):
    """Coerce '1'/'2' -> int; keep 'auto'. Refuse multi-GPU: DDP re-launches the
    whole entry script per rank, which breaks the Optuna loop and segfaults on
    rasterio. Use one process per GPU (shared --storage) to parallelise instead."""
    dev = int(v) if isinstance(v, str) and v.lstrip("-").isdigit() else v
    if isinstance(dev, int) and dev > 1:
        raise SystemExit(
            f"--devices {dev}: the search must run one GPU per process (no DDP). "
            "Pass --devices 1. To use N GPUs, launch N tuner processes sharing the "
            "same --storage, each pinned with CUDA_VISIBLE_DEVICES."
        )
    return dev


def resolve_encoder_weights(base_cfg: dict, cli_value: str | None):
    """CLI overrides base config; map random-init aliases -> None (random init)."""
    val = base_cfg.get("model", {}).get("encoder_weights", "imagenet") if cli_value is None else cli_value
    if isinstance(val, str) and val.lower() in {"none", "null", "random", ""}:
        return None
    return val


def resolve_mask_dirname(base_cfg: dict, cli_value: str | None):
    """CLI overrides base config; ''/'none'/'null' -> None (CDNGI masks_raster)."""
    val = base_cfg.get("data", {}).get("mask_dirname") if cli_value is None else cli_value
    if isinstance(val, str) and val.lower() in {"", "none", "null"}:
        return None
    return val


def build_objective(args, base_cfg: dict):
    data_cfg = _data_kwargs(base_cfg, args.dataset_dir, args.num_workers, args.mask_dirname)
    model_cfg = dict(base_cfg.get("model", {}))
    bands = tuple(data_cfg["bands"])
    if data_cfg.get("mask_dirname"):
        print(f"[unet.tune] label source: <split>/{data_cfg['mask_dirname']}/ (not CDNGI masks_raster)")
    devices = _resolve_devices(args.devices)
    encoder_weights = resolve_encoder_weights(base_cfg, args.encoder_weights)

    def objective(trial: optuna.Trial) -> float:
        # --- search space (log-uniform where scale-free) ----------------------
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        pos_weight = trial.suggest_float("pos_weight", args.pos_weight_min, args.pos_weight_max)
        encoder_name = trial.suggest_categorical("encoder_name", args.encoders)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)

        # Training seed: the base --train-seed for EVERY trial (so a trial's
        # score doesn't depend on which parallel worker ran it); --seed only
        # decorrelates the per-worker TPE samplers.
        pl.seed_everything(args.train_seed if args.train_seed is not None
                           else args.seed, workers=True)

        dm = RoadDataModule(
            dataset_dir=data_cfg["dataset_dir"],
            bands=bands,
            batch_size=batch_size,
            num_workers=data_cfg.get("num_workers", 1),
            image_size=data_cfg.get("image_size", 256),
            length=data_cfg.get("length"),
            normalize=data_cfg.get("normalize", True),
            norm_mean=data_cfg["norm_mean"],
            norm_std=data_cfg["norm_std"],
            mask_dirname=data_cfg.get("mask_dirname"),
        )

        model = UNetLightning(
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
        )

        pruning_cb = PyTorchLightningPruningCallback(trial, monitor=MONITOR)
        callbacks = [pruning_cb]
        if args.patience > 0:
            callbacks.append(EarlyStopping(monitor=MONITOR, mode=MONITOR_MODE, patience=args.patience))

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=devices,
            # single_device: DDP would re-launch this whole script per rank, which
            # (a) re-drives the same Optuna study from every rank and (b) opens
            # rasterio/GDAL in a subprocess -> segfault. Each trial runs on one GPU.
            strategy="auto",
            precision=args.precision,
            logger=False,               # keep trials quiet; final refit does the W&B logging
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=10,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=dm)

        # Optuna>=3.5 pruning callbacks defer the raised TrialPruned to here.
        pruning_cb.check_pruned()

        value = trainer.callback_metrics.get(MONITOR)
        if value is None:
            raise RuntimeError(f"'{MONITOR}' was never logged; cannot score the trial.")
        return float(value)

    return objective


# --------------------------------------------------------------------------- #
# Output: best trial -> Lightning config overlay
# --------------------------------------------------------------------------- #
def write_best_overlay(study: optuna.Study, out_dir: Path, encoder_weights, mask_dirname) -> Path:
    p = study.best_params
    # Pin encoder_weights AND mask_dirname too, so the refit reproduces the SAME
    # init (imagenet vs random) and label source (CDNGI vs OSM) the search ran
    # under -- not whatever the base config defaults to. null = CDNGI masks_raster.
    overlay = {
        "model": {
            "encoder_name": p["encoder_name"],
            "encoder_weights": encoder_weights,
            "lr": p["lr"],
            "pos_weight": p["pos_weight"],
        },
        "data": {"batch_size": p["batch_size"], "mask_dirname": mask_dirname},
    }
    overlay_path = out_dir / "best_params.yaml"
    header = (
        "# Best hyperparameters from unet.tune (Optuna). Deep-merges over the base config:\n"
        f"#   python -m unet.cli fit --config src/unet/configs/unet.yaml --config {overlay_path.name}\n"
        f"# best {MONITOR}={study.best_value:.4f}  trial #{study.best_trial.number}\n"
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
                "n_trials": len(study.trials),
                "monitor": MONITOR,
            },
            fh,
            indent=2,
        )
    return overlay_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Optuna hyperparameter search for the UNet road model.")
    ap.add_argument("--base-config", action="append", default=[], metavar="YAML",
                    help="Base Lightning config(s); repeat to layer (e.g. unet.yaml then norm_stats.yaml).")
    ap.add_argument("--dataset-dir", default=None, help="Override data.dataset_dir from the base config.")
    ap.add_argument("--out", default="runs/unet_optuna", help="Where to write best_params.yaml + study.")
    ap.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers (0 avoids GDAL forks).")
    ap.add_argument("--mask-dirname", default=None,
                    help="Override data.mask_dirname (label source). Empty/'none' -> CDNGI "
                         "masks_raster; e.g. mask_osm_10 for OSM labels.")

    # study controls
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--timeout", type=float, default=None, help="Wall-clock budget in seconds (optional).")
    ap.add_argument("--study-name", default="unet_optuna")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///runs/unet_optuna/study.db (enables resume).")
    ap.add_argument("--seed", type=int, default=42,
                    help="TPE sampler seed — give each parallel worker a DIFFERENT "
                         "one so they don't propose duplicate points.")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="seed_everything() for every trial (default: --seed). Pin "
                         "to the base seed so trial scores are worker-independent.")

    # per-trial training budget
    ap.add_argument("--max-epochs", type=int, default=8, help="Short budget per trial; refit longer after.")
    ap.add_argument("--patience", type=int, default=3, help="EarlyStopping patience per trial (0=off).")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="1",
                    help="GPUs PER tuner process. Must be 1 (no DDP during search); "
                         "parallelise with one process per GPU sharing --storage.")
    ap.add_argument("--precision", default="bf16-mixed")

    # search space
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--lr-max", type=float, default=1e-2)
    ap.add_argument("--pos-weight-min", type=float, default=1.0)
    ap.add_argument("--pos-weight-max", type=float, default=15.0)
    ap.add_argument("--encoders", nargs="+", default=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--encoder-weights", default=None,
                    help="Override model.encoder_weights for the whole search: "
                         "'imagenet' = pretrained, 'none'/'random' = random init. "
                         "Default: use the base config's value.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.base_config:
        raise SystemExit("Pass at least one --base-config (the canonical unet.yaml + norm stats).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config(args.base_config)
    objective = build_objective(args, base_cfg)

    study = create_study_shared(args.study_name, args.storage, args.seed)
    # catch OOM: batch_size is searched, so exceeding VRAM is a per-trial FAIL,
    # not a reason to kill the worker and its remaining trial budget.
    import torch
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, catch=(torch.cuda.OutOfMemoryError,))

    encoder_weights = resolve_encoder_weights(base_cfg, args.encoder_weights)
    mask_dirname = resolve_mask_dirname(base_cfg, args.mask_dirname)
    overlay_path = write_best_overlay(study, out_dir, encoder_weights, mask_dirname)
    print(f"\nBest {MONITOR}={study.best_value:.4f} (trial #{study.best_trial.number})")
    print(f"Best params: {study.best_params}  encoder_weights={encoder_weights}  mask_dirname={mask_dirname}")
    print(f"Wrote Lightning overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
