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
import gc
import json
from pathlib import Path

import optuna
import yaml

try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ImportError:  # pragma: no cover - fallback for older optuna
    from optuna.integration import PyTorchLightningPruningCallback

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping

# Same imports the LightningCLI uses -- keep the search and the real fit identical.
from sentinel2data.dataset.joint_sr_dataset import JointSRDataModule
from sr.model import JointSRUNetLightning
# Study plumbing shared with unet.tune (single source of truth: the retry-on-
# DDL-race create, the journal/RDB storage handling, the best-score tracker).
from unet.tune import (
    MONITOR,
    MONITOR_MODE,
    BestScoreCallback,
    _resolve_devices,
    create_study_shared,
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
    freeze_sr = (model_cfg.get("freeze_sr", False) if args.freeze_sr is None
                 else args.freeze_sr == "true")
    sr_pad = model_cfg.get("sr_pad", 0) if args.sr_pad is None else args.sr_pad
    warm_start_unet = (args.warm_start_unet
                       if args.warm_start_unet is not None
                       else model_cfg.get("warm_start_unet"))
    # Bicubic (R0) has no learnable SR params and frozen SEN2SR (R1) never
    # updates, so lr_sr is a dead search dimension in both -- skip it entirely
    # rather than let TPE waste trials on it.
    search_lr_sr = upsampler != "bicubic" and not freeze_sr
    if not search_lr_sr:
        print(f"[sr.tune] upsampler={upsampler!r} freeze_sr={freeze_sr}: "
              "lr_sr is not searched (no trainable SR params).")
    # Loss arm (unet.losses.build_loss). The protocol's arms use PLAIN CE —
    # pos_weight is itself a distribution-slot reweighting, so with an arm set
    # it is a dead search dimension too (UNetLightning ignores it): skip it.
    loss_arm = args.loss_arm if args.loss_arm is not None else model_cfg.get("loss_arm")
    loss_hp = dict(
        pstar=args.pstar, gap_r=args.gap_r, gap_k=args.gap_k,
        tl_ell=args.tl_ell, tl_theta=args.tl_theta,
        tversky_alpha=args.tversky_alpha, cl_alpha=args.cl_alpha,
        cl_iters=args.cl_iters, sr_w=args.skel_w, sr_radius=args.skel_radius,
        warmup_start=args.warmup_start, warmup_ramp=args.warmup_ramp,
    )
    # Recipe v2 constants (not searched; pinned into the overlay so the refit
    # reproduces them): schedule, SR warmup, dormant L2-SP.
    lr_schedule = args.lr_schedule or model_cfg.get("lr_schedule", "cosine")
    sr_warmup_epochs = (args.sr_warmup_epochs if args.sr_warmup_epochs is not None
                        else float(model_cfg.get("sr_warmup_epochs", 1.0)))
    l2sp_lambda = (args.l2sp_lambda if args.l2sp_lambda is not None
                   else float(model_cfg.get("l2sp_lambda", 0.0)))
    search_pos_weight = not loss_arm
    if loss_arm:
        print(f"[sr.tune] loss_arm={loss_arm!r}: pos_weight is not searched "
              "(protocol arms use plain CE).")
        if "+" in loss_arm and args.warmup_start >= args.max_epochs:
            print(f"[sr.tune] WARNING: warmup_start={args.warmup_start} >= "
                  f"max_epochs={args.max_epochs}: the skeleton slot never "
                  "activates inside the short tuning trials — trials score the "
                  "base compound only. Consider --warmup-start/--warmup-ramp "
                  "scaled to the trial budget.")

    def objective(trial: optuna.Trial) -> float:
        # --- search space: the joint LR pair is the star -------------------
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        lr_sr = (trial.suggest_float("lr_sr", args.lr_sr_min, args.lr_sr_max, log=True)
                 if search_lr_sr else args.lr_sr_min)  # bicubic ignores lr_sr
        pos_weight = (trial.suggest_float("pos_weight", args.pos_weight_min,
                                          args.pos_weight_max)
                      if search_pos_weight
                      else model_cfg.get("pos_weight", 5.0))  # unused by arms
        encoder_name = trial.suggest_categorical("encoder_name", args.encoders)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)

        # Training seed: the SAME --train-seed for EVERY trial (so a trial's
        # score doesn't depend on which parallel worker ran it); --seed only
        # affects the per-worker TPE samplers.
        pl.seed_everything(args.train_seed, workers=True)

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
            mask_dirname=data_cfg.get("mask_dirname", "mask_osm_2pt5"),
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
            freeze_sr=freeze_sr,
            upscale=upscale,
            sr_pad=sr_pad,
            reflectance_scale=model_cfg.get("reflectance_scale", 10000.0),
            warm_start_unet=warm_start_unet,
            lr_schedule=lr_schedule,
            sr_warmup_epochs=sr_warmup_epochs,
            l2sp_lambda=l2sp_lambda,
            loss_arm=loss_arm,
            **loss_hp,
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
            strategy="auto",  # one GPU per process; see unet.tune._resolve_devices
            precision=args.precision,
            gradient_clip_val=args.clip if args.clip > 0 else None,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=10,
            callbacks=callbacks,
        )
        try:
            trainer.fit(model, datamodule=dm)
        except torch.cuda.OutOfMemoryError:
            # PRUNED, not FAIL: TPE models COMPLETE+PRUNED trials only, so a
            # FAIL teaches the sampler nothing and it keeps re-proposing the
            # same too-big region for the rest of the study.
            trial.set_user_attr("oom", True)
            raise optuna.TrialPruned(
                f"OOM: batch_size={batch_size}, upsampler={upsampler}")
        finally:
            # OOM (an expected outcome for the larger searched batch sizes with
            # heavy SR nets) leaves the allocator full — release before the
            # next trial runs in this same process.
            del model, dm
            gc.collect()
            torch.cuda.empty_cache()
        pruning_cb.check_pruned()

        # Best val_iou across epochs (callback_metrics alone = last epoch's).
        if best_cb.best is None:
            raise RuntimeError(f"'{MONITOR}' was never logged; cannot score the trial.")
        return float(best_cb.best)

    return objective


def write_best_overlay(study: optuna.Study, out_dir: Path, encoder_weights,
                       upsampler: str, freeze_sr: bool = False,
                       sr_pad: int = 0, loss_arm: str | None = None,
                       loss_hp: dict | None = None,
                       warm_start_unet: str | None = None,
                       precision: str | None = None,
                       mask_source: str | None = None,
                       lr_schedule: str | None = None,
                       sr_warmup_epochs: float | None = None,
                       l2sp_lambda: float | None = None) -> Path:
    p = study.best_params
    # Record the resolved SR treatment AND loss so the refit is unambiguous
    # from the overlay alone (an R0/R1/padded/arm overlay layered over
    # joint_sr.yaml fully reproduces the searched configuration).
    model_overlay = {
        "encoder_name": p["encoder_name"],
        "encoder_weights": encoder_weights,
        "upsampler": upsampler,
        "freeze_sr": freeze_sr,
        "sr_pad": sr_pad,
        "lr": p["lr"],
    }
    if warm_start_unet:
        model_overlay["warm_start_unet"] = str(warm_start_unet)
    if loss_arm:
        model_overlay["loss_arm"] = loss_arm
        model_overlay.update(loss_hp or {})
    else:
        model_overlay["pos_weight"] = p["pos_weight"]  # legacy loss only
    has_lr_sr = "lr_sr" in p  # absent for R0 (bicubic) searches
    if has_lr_sr:
        model_overlay["lr_sr"] = p["lr_sr"]
    # Recipe v2 constants: pinned so the refit reproduces the search's recipe
    # even if the base config later drifts.
    if lr_schedule is not None:
        model_overlay["lr_schedule"] = lr_schedule
    if sr_warmup_epochs is not None:
        model_overlay["sr_warmup_epochs"] = sr_warmup_epochs
    if l2sp_lambda is not None:
        model_overlay["l2sp_lambda"] = l2sp_lambda
    data_overlay = {"batch_size": p["batch_size"]}
    if mask_source:
        # Label-source leak fix: a raster-mask search must not silently refit
        # on graph masks (or vice versa) -- same reason unet.tune pins
        # mask_dirname.
        data_overlay["mask_source"] = mask_source
    overlay = {"model": model_overlay, "data": data_overlay}
    if precision:
        # Pin the numerical regime the trials ran under (bf16-mixed by default)
        # so the refit doesn't silently fall back to the base config's fp32.
        # (The SR stage itself always runs fp32 -- see JointSRUNetLightning
        # .forward's autocast-disabled island -- this pins the UNet stage.)
        overlay["trainer"] = {"precision": precision}
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
                "loss_arm": loss_arm,
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
    ap.add_argument("--upsampler", default=None,
                    choices=["sen2sr", "sen2sr_full", "sr4rs", "bicubic"],
                    help="Override model.upsampler. 'bicubic' = R0 baseline: lr_sr "
                         "is NOT searched (no SR params) and needs no SEN2SR weights. "
                         "'sen2sr_full' = Mamba variant (needs mamba_ssm + its own dir).")
    ap.add_argument("--freeze-sr", default=None, choices=["true", "false"],
                    help="Override model.freeze_sr (true -> R1 frozen SR preprocessing).")
    ap.add_argument("--sr-pad", type=int, default=None,
                    help="Override model.sr_pad (reflect-pad in native px; 8 = border-artifact fix).")
    ap.add_argument("--warm-start-unet", default=None, metavar="CKPT",
                    help="Stage-1 (frozen-SR) JointSR ckpt whose UNet weights "
                         "initialise every trial's UNet (staged R6/R7 protocol; "
                         "pin pos_weight/batch/encoder to the stage-1 best and "
                         "search lr over a fine-tuning band anchored to it, "
                         "plus lr_sr). Overrides model.warm_start_unet.")
    ap.add_argument("--mask-source", default=None, choices=["graph", "raster"],
                    help="Override data.mask_source (graph = CDNGI, raster = OSM HR masks).")

    # loss arm + hyperparameters (unet.losses.build_loss; mirrors
    # unet.train_ablation). Default None = legacy Dice + pos-weighted BCE.
    ap.add_argument("--loss-arm", default=None,
                    help="bce | gap_ce | tl_ce | gap_tl_ce | t2_ce | t4_ce | "
                         "bce_dice | pstar_dice | pstar_tversky | focal_tversky "
                         "| <base>+cldice | <base>+skelrec. When set, pos_weight "
                         "is NOT searched (arms use plain CE).")
    ap.add_argument("--pstar", default="bce", help="pixel slot for pstar_* arms")
    ap.add_argument("--gap-r", type=int, default=4, help="GapLoss buffer radius")
    ap.add_argument("--gap-k", type=float, default=60.0, help="GapLoss K")
    ap.add_argument("--tl-ell", type=int, default=5, help="TL/T2/T4 filter length")
    ap.add_argument("--tl-theta", type=float, default=0.375,
                    help="TL/T2/T4/gap_tl weight-map binarization threshold")
    ap.add_argument("--tversky-alpha", type=float, default=0.7)
    ap.add_argument("--cl-alpha", type=float, default=0.3)
    ap.add_argument("--cl-iters", type=int, default=5)
    ap.add_argument("--skel-w", type=float, default=1.0,
                    help="Skeleton-Recall weight (build_loss's sr_w; renamed here "
                         "to avoid clashing with the SR-net flags)")
    ap.add_argument("--skel-radius", type=int, default=1,
                    help="Skeleton-Recall tube radius (build_loss's sr_radius)")
    ap.add_argument("--warmup-start", type=int, default=30)
    ap.add_argument("--warmup-ramp", type=int, default=10)
    ap.add_argument("--out", default="runs/sr_optuna", help="Where to write best_params.yaml + study.")
    ap.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers (0 avoids GDAL forks).")

    # study controls
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--timeout", type=float, default=None, help="Wall-clock budget in seconds (optional).")
    ap.add_argument("--study-name", default=None,
                    help="Default: derived from the treatment (upsampler/frozen/"
                         "pad/mask/warm/loss-arm) so different R-configs can "
                         "NEVER silently mix in one study -- their treatment "
                         "flags live outside the searched params, so mixed "
                         "trials would be incomparable and TPE would model "
                         "the union.")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///runs/sr_optuna/study.db (enables resume).")
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
                         "v2 floor across every arm.")
    ap.add_argument("--sr-warmup-epochs", type=float, default=None,
                    help="Linear LR ramp on the SR group, in epochs (fractional "
                         "ok; absolute, NOT scaled to the trial budget). "
                         "Default: model.sr_warmup_epochs (1.0). Auto-off in "
                         "the model for frozen/bicubic SR and staged warm "
                         "starts.")
    ap.add_argument("--lr-schedule", default=None, choices=["cosine", "none"],
                    help="Override model.lr_schedule (default: base config's, "
                         "cosine). Applies to the trials AND is pinned into "
                         "the overlay for the refit.")
    ap.add_argument("--l2sp-lambda", type=float, default=None,
                    help="Override model.l2sp_lambda (default: base config's, "
                         "0.0 = dormant). Escalation knob -- raise above 0 "
                         "only on sr_drift_rel evidence.")

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

    # Resolve the treatment ONCE, up front: it names the study (so different
    # R-configs can never share one), and stamps the overlay afterwards.
    model_cfg = base_cfg.get("model", {})
    upsampler = args.upsampler or model_cfg.get("upsampler", "sen2sr")
    freeze_sr = (model_cfg.get("freeze_sr", False) if args.freeze_sr is None
                 else args.freeze_sr == "true")
    sr_pad = model_cfg.get("sr_pad", 0) if args.sr_pad is None else args.sr_pad
    loss_arm = args.loss_arm if args.loss_arm is not None else model_cfg.get("loss_arm")
    warm_start_unet = (args.warm_start_unet if args.warm_start_unet is not None
                       else model_cfg.get("warm_start_unet"))
    mask_source = args.mask_source or base_cfg.get("data", {}).get("mask_source", "graph")
    lr_schedule = args.lr_schedule or model_cfg.get("lr_schedule", "cosine")
    sr_warmup_epochs = (args.sr_warmup_epochs if args.sr_warmup_epochs is not None
                        else float(model_cfg.get("sr_warmup_epochs", 1.0)))
    l2sp_lambda = (args.l2sp_lambda if args.l2sp_lambda is not None
                   else float(model_cfg.get("l2sp_lambda", 0.0)))

    study_name = args.study_name
    if study_name is None:
        study_name = ("sr_optuna_" + upsampler
                      + ("_frozen" if freeze_sr else "")
                      + f"_pad{sr_pad}_{mask_source}"
                      + ("_warm" if warm_start_unet else "")
                      + (f"_{loss_arm}" if loss_arm else ""))
        print(f"[sr.tune] study name (treatment-derived): {study_name}")

    objective = build_objective(args, base_cfg)
    study = create_study_shared(study_name, args.storage, args.seed)
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
    loss_hp = dict(
        pstar=args.pstar, gap_r=args.gap_r, gap_k=args.gap_k,
        tl_ell=args.tl_ell, tl_theta=args.tl_theta,
        tversky_alpha=args.tversky_alpha, cl_alpha=args.cl_alpha,
        cl_iters=args.cl_iters, sr_w=args.skel_w, sr_radius=args.skel_radius,
        warmup_start=args.warmup_start, warmup_ramp=args.warmup_ramp,
    )
    overlay_path = write_best_overlay(study, out_dir, encoder_weights, upsampler,
                                      freeze_sr, sr_pad, loss_arm, loss_hp,
                                      warm_start_unet, precision=args.precision,
                                      mask_source=mask_source,
                                      lr_schedule=lr_schedule,
                                      sr_warmup_epochs=sr_warmup_epochs,
                                      l2sp_lambda=l2sp_lambda)
    print(f"\nBest {MONITOR}={study.best_value:.4f} (trial #{study.best_trial.number})")
    print(f"Best params: {study.best_params}  encoder_weights={encoder_weights}")
    print(f"Wrote Lightning overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
