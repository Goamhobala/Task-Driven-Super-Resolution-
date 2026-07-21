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
            if storage and str(storage).startswith("journal://"):
                # NFS-safe multi-NODE sharing (see sr.tune for rationale):
                # STORAGE=journal://<runs>/study.journal
                path = str(storage)[len("journal://"):]
                from optuna.storages import JournalStorage
                try:    # optuna >= 4
                    from optuna.storages.journal import (JournalFileBackend,
                                                         JournalFileOpenLock)
                    backend = JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
                except ImportError:  # optuna 3.x
                    from optuna.storages import (JournalFileOpenLock,
                                                 JournalFileStorage)
                    backend = JournalFileStorage(path, lock_obj=JournalFileOpenLock(path))
                store = JournalStorage(backend)
            elif storage and str(storage).startswith("sqlite"):
                from optuna.storages import RDBStorage
                store = RDBStorage(url=str(storage),
                                   engine_kwargs={"connect_args": {"timeout": 60}})
            return optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=store,
                load_if_exists=storage is not None,
                sampler=optuna.samplers.TPESampler(
                    seed=seed,
                    # Parallel workers share one storage: constant_liar makes
                    # RUNNING trials visible to TPE so concurrent workers stop
                    # proposing near-duplicates (recommended by the sampler
                    # docs for distributed optimisation).
                    constant_liar=True,
                    # Joint model over interacting dims (lr x batch_size x
                    # encoder; lr x lr_sr in sr.tune -- the alpha ratio).
                    # Experimental-flagged but widely used.
                    multivariate=True,
                    group=True,
                ),
                # Prune from the 3rd validation (n_warmup_steps=2): with ~10
                # epoch trials, epoch-1 IoU kills slow starters (low lr / high
                # pos_weight) that rank well later. n_min_trials=2: need two
                # finished trials at a rung before its median can prune.
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2,
                                                   n_min_trials=2),
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
import torch
from lightning.pytorch.callbacks import EarlyStopping

# Same imports the LightningCLI uses -- keep the search and the real fit identical.
from sentinel2data.dataset.datasets import RoadDataModule
# Config plumbing shared with unet.train_ablation (moved to unet.config_utils).
from unet.config_utils import (
    data_kwargs as _data_kwargs,
    load_base_config,
    resolve_encoder_weights,
    resolve_mask_dirname,
)
from unet.model import UNetLightning

MONITOR = "val_iou"          # maximise per-crop IoU on the val quadrants
MONITOR_MODE = "max"


class BestScoreCallback(pl.Callback):
    """Track the best MONITOR across validation epochs.

    ``trainer.callback_metrics`` alone holds only the LAST epoch's value --
    with EarlyStopping(patience=k) that is ~k epochs past the peak, i.e. peak
    minus noise, which both corrupts TPE's model and mis-ranks the best trial.
    Works with --patience 0 too (unlike reading EarlyStopping.best_score)."""

    def __init__(self, monitor: str = MONITOR, mode: str = MONITOR_MODE):
        self.monitor, self.mode, self.best = monitor, mode, None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(self.monitor)
        if v is None:
            return
        v = float(v)
        if (self.best is None
                or (v > self.best if self.mode == "max" else v < self.best)):
            self.best = v


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

        # Training seed: the SAME --train-seed for EVERY trial (so a trial's
        # score doesn't depend on which parallel worker ran it); --seed only
        # affects the per-worker TPE samplers.
        pl.seed_everything(args.train_seed, workers=True)

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
            lr_schedule=model_cfg.get("lr_schedule", "cosine"),
            pos_weight=pos_weight,
            bands=bands,
            image_size=data_cfg.get("image_size", 256),
            threshold=model_cfg.get("threshold", 0.5),
            normalize=data_cfg.get("normalize", True),
            norm_mean=data_cfg["norm_mean"],
            norm_std=data_cfg["norm_std"],
        )

        pruning_cb = PyTorchLightningPruningCallback(trial, monitor=MONITOR)
        best_cb = BestScoreCallback()
        callbacks = [pruning_cb, best_cb]
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
            gradient_clip_val=args.clip if args.clip > 0 else None,
            logger=False,               # keep trials quiet; final refit does the W&B logging
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=10,
            callbacks=callbacks,
        )
        try:
            trainer.fit(model, datamodule=dm)
        except torch.cuda.OutOfMemoryError:
            # Record OOM as PRUNED, not FAIL: TPE builds its model from
            # COMPLETE+PRUNED trials only, so a FAIL teaches the sampler
            # nothing and it keeps re-proposing the same too-big region.
            # Pruned-with-poor-intermediates ranks at the bottom instead.
            trial.set_user_attr("oom", True)
            raise optuna.TrialPruned(
                f"OOM: batch_size={batch_size}, encoder={encoder_name}")

        # Optuna>=3.5 pruning callbacks defer the raised TrialPruned to here.
        pruning_cb.check_pruned()

        # Score on the BEST val_iou across epochs, per the module docstring --
        # callback_metrics holds only the last epoch's value, which with
        # EarlyStopping is ~patience epochs past the peak.
        if best_cb.best is None:
            raise RuntimeError(f"'{MONITOR}' was never logged; cannot score the trial.")
        return float(best_cb.best)

    return objective


# --------------------------------------------------------------------------- #
# Output: best trial -> Lightning config overlay
# --------------------------------------------------------------------------- #
def write_best_overlay(study: optuna.Study, out_dir: Path, encoder_weights, mask_dirname,
                       precision=None, lr_schedule=None) -> Path:
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
    }
    if lr_schedule is not None:
        # Pin the LR schedule the search ran under (recipe v2: cosine).
        overlay["model"]["lr_schedule"] = lr_schedule
    overlay |= {
        "data": {"batch_size": p["batch_size"], "mask_dirname": mask_dirname},
    }
    if precision:
        # Pin the numerical regime the trials ran under (bf16-mixed by default)
        # so the refit doesn't silently fall back to the base config's fp32 --
        # fp32 doubles activation memory, so the searched batch_size that fit
        # during the search could OOM in the refit.
        overlay["trainer"] = {"precision": precision}
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
    ap.add_argument("--seed", type=int, default=None,
                    help="TPE sampler seed. Default None = each worker gets an "
                         "independently random sampler (a FIXED shared default "
                         "would make N workers burn their startup budget on "
                         "identical random points). Pass distinct explicit "
                         "seeds only for reproducible searches.")
    ap.add_argument("--train-seed", type=int, default=42,
                    help="seed_everything() for every trial. One shared value "
                         "so trial scores are worker-independent (common "
                         "random numbers across configs).")

    # per-trial training budget
    ap.add_argument("--max-epochs", type=int, default=8, help="Short budget per trial; refit longer after.")
    ap.add_argument("--patience", type=int, default=3, help="EarlyStopping patience per trial (0=off).")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="1",
                    help="GPUs PER tuner process. Must be 1 (no DDP during search); "
                         "parallelise with one process per GPU sharing --storage.")
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="gradient_clip_val (global L2 norm; 0 = off). Recipe "
                         "v2 floor -- harmless for the plain UNet, required "
                         "for the joint-SR arms; one recipe everywhere.")

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
    # The objective converts OOM to TrialPruned (so TPE learns the region);
    # catch= stays as a backstop for OOMs escaping outside trainer.fit.
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, catch=(torch.cuda.OutOfMemoryError,))

    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE
                     for t in study.trials)
    if n_complete == 0:
        raise SystemExit(
            f"Study '{study.study_name}' has 0 COMPLETE trials "
            f"({len(study.trials)} total: all failed/pruned/OOM) -- nothing "
            "to write. Check the trial logs before rerunning.")

    encoder_weights = resolve_encoder_weights(base_cfg, args.encoder_weights)
    mask_dirname = resolve_mask_dirname(base_cfg, args.mask_dirname)
    overlay_path = write_best_overlay(study, out_dir, encoder_weights, mask_dirname,
                                      precision=args.precision,
                                      lr_schedule=base_cfg.get("model", {})
                                                          .get("lr_schedule", "cosine"))
    print(f"\nBest {MONITOR}={study.best_value:.4f} (trial #{study.best_trial.number})")
    print(f"Best params: {study.best_params}  encoder_weights={encoder_weights}  mask_dirname={mask_dirname}")
    print(f"Wrote Lightning overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
