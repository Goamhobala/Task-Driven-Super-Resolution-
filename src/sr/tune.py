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

# Free throughput on tensor-core GPUs, no change to bf16-AMP math:
#  - TF32 for the matmuls that stay fp32 (silences Lightning's tensor-core hint)
#  - cudnn conv-algorithm autotuning; safe because every shape is fixed
#    (128px native crops -> 512px SR grid, constant batch size per trial).
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Same imports the LightningCLI uses -- keep the search and the real fit identical.
from sentinel2data.dataset.joint_sr_dataset import JointSRDataModule
from sr.model import AdaptiveNormBandExit, JointSRUNetLightning
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
                 mask_source: str | None = None,
                 mask_dirname: str | None = None) -> dict:
    data = dict(cfg.get("data", {}))
    if mask_source:
        data["mask_source"] = mask_source
    if mask_dirname:
        data["mask_dirname"] = mask_dirname
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
                            args.mask_source, args.mask_dirname)
    model_cfg = dict(base_cfg.get("model", {}))
    bands = tuple(data_cfg.get("bands", (1, 2, 3, 4)))
    upscale = data_cfg.get("upscale", 4)
    devices = _resolve_devices(args.devices)
    encoder_weights = resolve_encoder_weights(base_cfg, args.encoder_weights)
    # Selection criterion for the objective, the pruner AND EarlyStopping
    # (--monitor; module-constant default keeps legacy paths byte-identical).
    monitor = getattr(args, "monitor", MONITOR)
    monitor_mode = MONITOR_MODE  # both supported criteria maximise
    sen2sr_dir = args.sen2sr_dir or model_cfg.get("sen2sr_dir")
    upsampler = args.upsampler or model_cfg.get("upsampler", "sen2sr")
    freeze_sr = (model_cfg.get("freeze_sr", False) if args.freeze_sr is None
                 else args.freeze_sr == "true")
    sr_pad = model_cfg.get("sr_pad", 0) if args.sr_pad is None else args.sr_pad
    # FFT hard constraint x generator (docs/hc_2x2_plan.md). Never searched --
    # it IS the treatment -- and pinned into the overlay so the refit cannot
    # run under a different constraint than the search scored.
    sr_hc = args.sr_hc or model_cfg.get("sr_hc", "native")
    hc_mask_path = (args.hc_mask_path if args.hc_mask_path is not None
                    else model_cfg.get("hc_mask_path"))
    warm_start_unet = (args.warm_start_unet
                       if args.warm_start_unet is not None
                       else model_cfg.get("warm_start_unet"))
    # --- read-out head (docs/sr_linear_probe.md) -----------------------------
    head = args.head or model_cfg.get("head", "unet")
    warm_start_head = (args.warm_start_head if args.warm_start_head is not None
                       else model_cfg.get("warm_start_head"))
    clip_sr = (args.clip_sr if args.clip_sr is not None
               else float(model_cfg.get("clip_sr", 0.0)))
    # A linear probe has no encoder. Leaving the categorical in the space would
    # make TPE model a dimension that cannot affect the objective, and would put
    # an `encoder_name` in the overlay that no run ever used.
    search_encoder = head != "linear"
    if head == "linear":
        print("[sr.tune] head='linear': encoder_name is NOT searched "
              f"(no encoder exists); clip_sr={clip_sr} clips the SR group only.")
    # Bicubic (R0) has no learnable SR params and frozen SEN2SR (R1) never
    # updates, so lr_sr is a dead search dimension in both -- skip it entirely
    # rather than let TPE waste trials on it.
    search_lr_sr = upsampler != "bicubic" and not freeze_sr
    if not search_lr_sr:
        print(f"[sr.tune] upsampler={upsampler!r} freeze_sr={freeze_sr}: "
              "lr_sr is not searched (no trainable SR params).")
    # Loss arm (unet.losses.build_loss). Amendment 2026-08-02 (per-arm tuning
    # protocol): search dimensions are gated on what the arm actually
    # CONSUMES, so TPE never models dead dimensions:
    #   * pos_weight λ — searched for the λ-composing pixel slots (wbce +
    #     the spatial maps, 2026-07-30 amendment). NOT searched for 'bce'
    #     (the λ=1 floor by definition), 'balance_ce' (sets its own adaptive
    #     per-batch λ; fixed-β ≡ wbce, see BalancedCELoss), or region-only
    #     arms (dice/sdice/lcdice/focal_tversky — no CE term at all).
    #   * tl_theta / gap_theta — binarization thresholds of the weight maps;
    #     searched for the arms that build those maps (the official sources
    #     disagree on the value: TL official code 0.5 vs papers' 0.375).
    loss_arm = args.loss_arm if args.loss_arm is not None else model_cfg.get("loss_arm")
    loss_hp = dict(
        pstar=args.pstar, gap_r=args.gap_r, gap_k=args.gap_k,
        tl_ell=args.tl_ell, tl_theta=args.tl_theta, gap_theta=args.gap_theta,
        tversky_alpha=args.tversky_alpha, cl_alpha=args.cl_alpha,
        cl_iters=args.cl_iters, sr_w=args.skel_w, sr_radius=args.skel_radius,
        warmup_start=args.warmup_start, warmup_ramp=args.warmup_ramp,
        mix_w=args.mix_w,
    )
    if args.length:
        data_cfg["length"] = args.length   # tune-time patches/epoch budget
    # Recipe v2 constants (not searched; pinned into the overlay so the refit
    # reproduces them): schedule, SR warmup, dormant L2-SP.
    lr_schedule = args.lr_schedule or model_cfg.get("lr_schedule", "cosine")
    sr_warmup_epochs = (args.sr_warmup_epochs if args.sr_warmup_epochs is not None
                        else float(model_cfg.get("sr_warmup_epochs", 1.0)))
    l2sp_lambda = (args.l2sp_lambda if args.l2sp_lambda is not None
                   else float(model_cfg.get("l2sp_lambda", 0.0)))
    # Adaptive post-SR normalisation (docs/adaptive_norm_plan.md). PINNED, not
    # searched: the momentum is the EMA's lag, which is itself the only
    # restoring force against SR output drift -- letting TPE pick it would tune
    # a stability knob on 15-epoch trial scores. Same discipline as the pinned
    # batch size.
    adaptive_norm = (args.adaptive_norm == "true" if args.adaptive_norm is not None
                     else bool(model_cfg.get("adaptive_norm", False)))
    adaptive_norm_momentum = (args.adaptive_norm_momentum
                              if args.adaptive_norm_momentum is not None
                              else float(model_cfg.get("adaptive_norm_momentum", 0.01)))
    norm_recalibrate = (args.norm_recalibrate
                        if args.norm_recalibrate is not None
                        else str(model_cfg.get("norm_recalibrate", "off")))
    # The remaining adapter settings have no CLI flag (they are not treatment
    # variables), but they MUST still be read from the base config: otherwise a
    # config that changes one is honoured in the refit and silently ignored in
    # every trial, so the search would score a different adapter than it ships.
    adapt_rest = {
        k: model_cfg[k] for k in (
            "adaptive_norm_warmup_steps", "adaptive_norm_check_every",
            "norm_recalibrate_batches", "sr_functional_monitor",
            "sr_monitor_samples",
        ) if k in model_cfg
    }
    search_pos_weight = True
    search_tl_theta = search_gap_theta = search_mix_w = False
    if loss_arm:
        base = loss_arm.partition("+")[0]
        pixel = args.pstar if base.startswith("pstar_") else base
        combo = {"gap_t2_ce", "gap_t4_ce", "gap_t2t4_ce"}
        lambda_arms = {"wbce", "gap_ce", "tl_ce", "gap_tl_ce",
                       "t2_ce", "t4_ce"} | combo
        search_pos_weight = pixel in lambda_arms
        thetas_on = args.search_thetas != "false"
        search_tl_theta = thetas_on and (
            pixel in ("tl_ce", "gap_tl_ce", "t2_ce", "t4_ce") or pixel in combo)
        search_gap_theta = thetas_on and (
            pixel in ("gap_ce", "gap_tl_ce") or pixel in combo)
        #   * mix_w — the P*<->region ratio of the pstar_* compounds. Searched
        #     for those only: bce_dice is the frozen 0.5/0.5 anchor by design,
        #     and every other base has no region slot to trade against.
        search_mix_w = (args.search_mix_w != "false"
                        and base.startswith("pstar_"))
        print(f"[sr.tune] loss_arm={loss_arm!r} (pixel slot {pixel!r}): "
              f"search pos_weight={search_pos_weight} tl_theta={search_tl_theta} "
              f"gap_theta={search_gap_theta} mix_w={search_mix_w}")
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
                                          args.pos_weight_max, log=True)
                      if search_pos_weight
                      else model_cfg.get("pos_weight", 5.0))  # ignored by the arm
        encoder_name = (trial.suggest_categorical("encoder_name", args.encoders)
                        if search_encoder else None)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)
        trial_hp = dict(loss_hp)
        if search_tl_theta:
            trial_hp["tl_theta"] = trial.suggest_float(
                "tl_theta", args.theta_min, args.theta_max)
        if search_gap_theta:
            trial_hp["gap_theta"] = trial.suggest_float(
                "gap_theta", args.theta_min, args.theta_max)
        if search_mix_w:
            trial_hp["mix_w"] = trial.suggest_float(
                "mix_w", args.mix_w_min, args.mix_w_max)

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
            sr_hc=sr_hc,
            hc_mask_path=hc_mask_path,
            reflectance_scale=model_cfg.get("reflectance_scale", 10000.0),
            warm_start_unet=warm_start_unet,
            lr_schedule=lr_schedule,
            sr_warmup_epochs=sr_warmup_epochs,
            l2sp_lambda=l2sp_lambda,
            adaptive_norm=adaptive_norm,
            adaptive_norm_momentum=adaptive_norm_momentum,
            norm_recalibrate=norm_recalibrate,
            # Tuning is the ONE place aborting is the right trade: a short
            # trial that leaves the band is a verdict on that corner of the
            # search space, and pruning it buys budget for corners that might
            # win. Fits default to "warn" and must stay that way -- a band exit
            # there is the experiment's result, not a reason to bin a
            # multi-day run.
            std_band_action="raise",
            head=head,
            warm_start_head=warm_start_head,
            clip_sr=clip_sr,
            **adapt_rest,
            loss_arm=loss_arm,
            **trial_hp,
        )

        pruning_cb = PyTorchLightningPruningCallback(trial, monitor=monitor)
        # Track BOTH criteria every trial: the monitored one scores the trial,
        # the other lands in user attrs for the cross-criterion re-ranking
        # audit (docs/ap_threshold_protocol_plan.md §2.2) at zero extra cost.
        best_iou_cb = BestScoreCallback("val_iou", "max")
        best_ap_cb = BestScoreCallback("val_ap", "max")
        best_cb = best_ap_cb if monitor == "val_ap" else best_iou_cb
        callbacks = [pruning_cb, best_iou_cb, best_ap_cb]
        if args.patience > 0:
            callbacks.append(EarlyStopping(monitor=monitor, mode=monitor_mode, patience=args.patience))

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=devices,
            strategy="auto",  # one GPU per process; see unet.tune._resolve_devices
            precision=args.precision,
            # Per-group clipping is done by the module's
            # configure_gradient_clipping override, which REFUSES to run
            # alongside a Trainer-level value (both would apply, and the global
            # one is the treatment-dependent norm we are avoiding).
            gradient_clip_val=(None if clip_sr > 0
                               else (args.clip if args.clip > 0 else None)),
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
        except AdaptiveNormBandExit as exc:
            # A band exit means THIS hyperparameter corner destroys the SR
            # front-end -- which is a finding about the corner, not a bug. It
            # must fail the TRIAL, not the study: on 2026-08-13 an unhandled
            # one took down a whole tune after Optuna sampled lr_sr=3.1e-4
            # (30x the design default). Pruned rather than failed, for the same
            # reason as OOM: the sampler only models COMPLETE and PRUNED
            # trials, so a FAIL here would let TPE keep proposing the same
            # region for the rest of the study.
            trial.set_user_attr("adaptive_norm_band_exit", True)
            trial.set_user_attr("band_std_ratios", getattr(exc, "ratios", []))
            print(f"[sr.tune] trial {trial.number} pruned: post-SR std left "
                  f"the band {getattr(exc, 'band', None)} at lr_sr={lr_sr:.3g} "
                  f"(lr={lr:.3g}). Ratios: "
                  f"{['%.3f' % v for v in getattr(exc, 'ratios', [])]}")
            raise optuna.TrialPruned(f"adaptive_norm band exit: lr_sr={lr_sr:.3g}")
        finally:
            # OOM (an expected outcome for the larger searched batch sizes with
            # heavy SR nets) leaves the allocator full — release before the
            # next trial runs in this same process.
            del model, dm
            gc.collect()
            torch.cuda.empty_cache()
        pruning_cb.check_pruned()

        # Cross-criterion audit attrs (best-across-epochs, one per criterion).
        if best_iou_cb.best is not None:
            trial.set_user_attr("best_val_iou", best_iou_cb.best)
        if best_ap_cb.best is not None:
            trial.set_user_attr("best_val_ap", best_ap_cb.best)

        # Best monitored value across epochs (callback_metrics = last epoch's).
        if best_cb.best is None:
            raise RuntimeError(f"'{monitor}' was never logged; cannot score the trial.")
        return float(best_cb.best)

    return objective


def write_best_overlay(study: optuna.Study, out_dir: Path, encoder_weights,
                       upsampler: str, freeze_sr: bool = False,
                       sr_pad: int = 0, loss_arm: str | None = None,
                       loss_hp: dict | None = None,
                       warm_start_unet: str | None = None,
                       precision: str | None = None,
                       mask_source: str | None = None,
                       mask_dirname: str | None = None,
                       lr_schedule: str | None = None,
                       sr_warmup_epochs: float | None = None,
                       l2sp_lambda: float | None = None,
                       adaptive_norm: bool | None = None,
                       adaptive_norm_momentum: float | None = None,
                       norm_recalibrate: str | None = None,
                       head: str = "unet",
                       warm_start_head: str | None = None,
                       clip_sr: float | None = None,
                       sr_hc: str = "native",
                       hc_mask_path: str | None = None,
                       monitor: str = MONITOR) -> Path:
    p = study.best_params
    # Record the resolved SR treatment AND loss so the refit is unambiguous
    # from the overlay alone (an R0/R1/padded/arm overlay layered over
    # joint_sr.yaml fully reproduces the searched configuration).
    model_overlay = {
        # None for a linear probe: encoder_name is not in the search space, and
        # writing a resnet34 into the overlay would let the refit be read back
        # as an encoder ablation of a network that was never built.
        "encoder_name": p.get("encoder_name"),
        "encoder_weights": encoder_weights if head != "linear" else None,
        "upsampler": upsampler,
        "freeze_sr": freeze_sr,
        "sr_pad": sr_pad,
        "lr": p["lr"],
    }
    # Written ONLY when forced, so every native-constraint arm's overlay stays
    # byte-identical to what it was before the flag existed (and an old overlay
    # replayed today still resolves to the behaviour it was searched under).
    if sr_hc != "native":
        model_overlay["sr_hc"] = sr_hc
        if hc_mask_path:
            model_overlay["hc_mask_path"] = str(hc_mask_path)
    if head != "unet":
        model_overlay["head"] = head
    if warm_start_head:
        model_overlay["warm_start_head"] = str(warm_start_head)
    if clip_sr:
        model_overlay["clip_sr"] = float(clip_sr)
    if warm_start_unet:
        model_overlay["warm_start_unet"] = str(warm_start_unet)
    if loss_arm:
        model_overlay["loss_arm"] = loss_arm
        model_overlay.update(loss_hp or {})
    if "pos_weight" in p:
        # legacy loss OR a λ-consuming arm (2026-08-02): pin the searched λ.
        model_overlay["pos_weight"] = p["pos_weight"]
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
    # Adaptive-norm constants: pinned into the overlay so the refit runs the
    # SAME adapter the search scored. An unpinned refit of an adaptive-norm
    # search would silently fall back to the base config's frozen stats.
    if adaptive_norm is not None:
        model_overlay["adaptive_norm"] = bool(adaptive_norm)
    if adaptive_norm_momentum is not None:
        model_overlay["adaptive_norm_momentum"] = adaptive_norm_momentum
    if norm_recalibrate is not None:
        model_overlay["norm_recalibrate"] = str(norm_recalibrate)
    data_overlay = {"batch_size": p["batch_size"]}
    if mask_source:
        # Label-source leak fix: a raster-mask search must not silently refit
        # on graph masks (or vice versa) -- same reason unet.tune pins
        # mask_dirname.
        data_overlay["mask_source"] = mask_source
    if mask_dirname:
        # Pin WHICH raster folder too (mask_new_2pt5 vs mask_osm_2pt5): the
        # datamodule default is mask_osm_2pt5, so an unpinned refit of a
        # mask_new_2pt5 search would silently swap label sources.
        data_overlay["mask_dirname"] = mask_dirname
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
        f"# best {monitor}={study.best_value:.4f}  trial #{study.best_trial.number}{alpha}\n"
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
                "monitor": monitor,
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
    ap.add_argument("--sr-hc", default=None, choices=["native", "on", "off"],
                    help="Override model.sr_hc — the FFT hard constraint as a "
                         "treatment, crossed with the generator (the HC 2x2, "
                         "docs/hc_2x2_plan.md). 'native' (default) = each "
                         "upsampler's shipped behaviour (sen2sr on, sr4rs and "
                         "bicubic off); 'on'/'off' force the whole bundle "
                         "(positivity clamp + frequency splice) on or off. "
                         "Never searched: it is the treatment.")
    ap.add_argument("--hc-mask-path", default=None, metavar="SAFETENSOR",
                    help="Override model.hc_mask_path: SEN2SR-Lite's shipped "
                         "hard_constraint.safetensor. REQUIRED for --sr-hc on "
                         "with --upsampler sr4rs (whose model dir ships no "
                         "mask); the SEN2SR arms read theirs from --sen2sr-dir.")
    ap.add_argument("--warm-start-unet", default=None, metavar="CKPT",
                    help="Stage-1 (frozen-SR) JointSR ckpt whose UNet weights "
                         "initialise every trial's UNet (staged R6/R7 protocol; "
                         "pin pos_weight/batch/encoder to the stage-1 best and "
                         "search lr over a fine-tuning band anchored to it, "
                         "plus lr_sr). Overrides model.warm_start_unet.")
    ap.add_argument("--head", default=None, choices=["unet", "linear"],
                    help="Read-out head. 'unet' (default) = the 24M-param "
                         "decoder, i.e. every R-arm. 'linear' = the RL-series "
                         "1x1-conv probe: encoder_name leaves the search space "
                         "and the head runs in fp32. Overrides model.head.")
    ap.add_argument("--warm-start-head", default=None, metavar="CKPT",
                    help="LP-FT: frozen-twin ckpt whose LINEAR PROBE weights "
                         "initialise every trial's head (rl2<-rl1, rl4<-rl3). "
                         "Strictly separate from --warm-start-unet, which "
                         "auto-disables the SR warmup ramp; this must not.")
    ap.add_argument("--clip-sr", type=float, default=None, metavar="NORM",
                    help="Per-GROUP gradient clipping: L2-clip the SR group at "
                         "NORM and leave the head unclipped, instead of "
                         "Lightning's single global norm over both (which means "
                         "different things in a frozen vs a joint arm). >0 also "
                         "disables the Trainer-level clip. 0 = global (default).")
    ap.add_argument("--mask-source", default=None, choices=["graph", "raster"],
                    help="Override data.mask_source (graph = CDNGI, raster = OSM HR masks).")
    ap.add_argument("--mask-dirname", default=None,
                    help="Mask folder under <split>/ when mask_source=raster "
                         "(mask_new_2pt5 = once-off pre-rasterised graph labels, "
                         "mask_osm_2pt5 = OSM). Default: base config's value.")

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
                    help="TL/T2/T4/gap_tl weight-map binarization threshold "
                         "(fixed value when --search-thetas false)")
    ap.add_argument("--gap-theta", type=float, default=0.5,
                    help="GapLoss weight-map binarization threshold (official "
                         "code: 0.5; fixed value when --search-thetas false)")
    ap.add_argument("--search-thetas", default="true", choices=["true", "false"],
                    help="Search tl_theta/gap_theta for arms that build those "
                         "maps (2026-08-02 amendment: the official sources "
                         "disagree on θ, so it is a per-arm hyperparameter).")
    ap.add_argument("--theta-min", type=float, default=0.3)
    ap.add_argument("--theta-max", type=float, default=0.7)
    ap.add_argument("--mix-w", type=float, default=0.5,
                    help="P*<->region mixing ratio for pstar_* compounds: "
                         "L = (1-mix_w)*P* + mix_w*region. Fixed value when "
                         "--search-mix-w false; always pinned into the overlay.")
    ap.add_argument("--search-mix-w", default="true", choices=["true", "false"],
                    help="Search mix_w for pstar_* compound arms (2026-08-05). "
                         "It is the compound's one genuinely free parameter — "
                         "no paper default exists — and adding a region term "
                         "changes where the optimum sits, so a compound "
                         "compared at a frozen 0.5 is not being compared at "
                         "its best. Consumption-gated like the θs: ignored by "
                         "non-compound arms and by the bce_dice anchor, which "
                         "build_loss deliberately freezes at 0.5/0.5.")
    ap.add_argument("--mix-w-min", type=float, default=0.25)
    ap.add_argument("--mix-w-max", type=float, default=0.75)
    ap.add_argument("--length", type=int, default=None,
                    help="Tune-time patches/epoch (data.length override); the "
                         "trials rank configs, they don't need full epochs — "
                         "this is the main tune-cost lever.")
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
    ap.add_argument("--monitor", default=MONITOR, choices=["val_iou", "val_ap"],
                    help="Selection metric for the objective, the pruner AND "
                         "EarlyStopping. val_ap = threshold-free (binned AP; "
                         "the _new-series protocol). Default val_iou keeps "
                         "legacy paths byte-identical. Studies must not "
                         "resume across a criterion change -- _stages_tv.sh "
                         "folds the monitor into STUDY_NAME for exactly that.")
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
    ap.add_argument("--adaptive-norm", default=None, choices=["true", "false"],
                    help="Override model.adaptive_norm: per-batch EMA of the "
                         "post-SR normalisation statistics (default: base "
                         "config's, i.e. false = frozen dataset stats).")
    ap.add_argument("--adaptive-norm-momentum", type=float, default=None,
                    help="Override model.adaptive_norm_momentum. PINNED into "
                         "the overlay, never searched: it is a stability knob "
                         "(the EMA lag IS the restoring force), not a "
                         "hyperparameter a 15-epoch trial can score.")
    ap.add_argument("--norm-recalibrate", default=None,
                    choices=["off", "pre", "post", "auto"],
                    help="Override model.norm_recalibrate: exact PreciseBN-"
                         "style recompute of the post-SR stats (pre = before "
                         "fitting, the complete fix for frozen-SR arms; post = "
                         "before the final ckpt is written).")
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
    sr_hc = args.sr_hc or model_cfg.get("sr_hc", "native")
    hc_mask_path = (args.hc_mask_path if args.hc_mask_path is not None
                    else model_cfg.get("hc_mask_path"))
    loss_arm = args.loss_arm if args.loss_arm is not None else model_cfg.get("loss_arm")
    warm_start_unet = (args.warm_start_unet if args.warm_start_unet is not None
                       else model_cfg.get("warm_start_unet"))
    head = args.head or model_cfg.get("head", "unet")
    warm_start_head = (args.warm_start_head if args.warm_start_head is not None
                       else model_cfg.get("warm_start_head"))
    clip_sr = (args.clip_sr if args.clip_sr is not None
               else float(model_cfg.get("clip_sr", 0.0)))
    mask_source = args.mask_source or base_cfg.get("data", {}).get("mask_source", "graph")
    mask_dirname = args.mask_dirname or base_cfg.get("data", {}).get("mask_dirname")
    lr_schedule = args.lr_schedule or model_cfg.get("lr_schedule", "cosine")
    sr_warmup_epochs = (args.sr_warmup_epochs if args.sr_warmup_epochs is not None
                        else float(model_cfg.get("sr_warmup_epochs", 1.0)))
    l2sp_lambda = (args.l2sp_lambda if args.l2sp_lambda is not None
                   else float(model_cfg.get("l2sp_lambda", 0.0)))
    adaptive_norm = (args.adaptive_norm == "true" if args.adaptive_norm is not None
                     else bool(model_cfg.get("adaptive_norm", False)))
    adaptive_norm_momentum = (args.adaptive_norm_momentum
                              if args.adaptive_norm_momentum is not None
                              else float(model_cfg.get("adaptive_norm_momentum", 0.01)))
    norm_recalibrate = (args.norm_recalibrate if args.norm_recalibrate is not None
                        else str(model_cfg.get("norm_recalibrate", "off")))

    study_name = args.study_name
    if study_name is None:
        study_name = ("sr_optuna_" + upsampler
                      + ("_frozen" if freeze_sr else "")
                      + f"_pad{sr_pad}_{mask_source}"
                      # The hard constraint is the treatment of the HC 2x2, so
                      # an HC-forced study must never share a storage row with
                      # the same generator's native-constraint study.
                      + ("" if sr_hc == "native" else f"_hc{sr_hc}")
                      + ("_warm" if warm_start_unet else "")
                      # The read-out is the treatment for the whole RL-series,
                      # so a linear-probe study must never share a storage row
                      # with the U-Net study of the same SR arm.
                      + ("" if head == "unet" else f"_{head}")
                      + ("_warmhead" if warm_start_head else "")
                      # The adapter is part of the treatment, not a nuisance
                      # setting: an adaptive-norm study must never share a
                      # storage row with the frozen-stats study of the same arm.
                      + ("_anorm" if adaptive_norm else "")
                      + (f"_recal{norm_recalibrate}" if norm_recalibrate != "off" else "")
                      + (f"_{loss_arm}" if loss_arm else ""))
        print(f"[sr.tune] study name (treatment-derived): {study_name}")

    objective = build_objective(args, base_cfg)
    study = create_study_shared(study_name, args.storage, args.seed)
    # The objective converts OOM and adaptive-norm band exits to TrialPruned
    # (so TPE learns the region); catch= stays as a backstop for either
    # escaping outside trainer.fit -- a bad corner of the search space must
    # never be able to end the study.
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True,
                   catch=(torch.cuda.OutOfMemoryError, AdaptiveNormBandExit))

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
        tl_ell=args.tl_ell, tl_theta=args.tl_theta, gap_theta=args.gap_theta,
        tversky_alpha=args.tversky_alpha, cl_alpha=args.cl_alpha,
        cl_iters=args.cl_iters, sr_w=args.skel_w, sr_radius=args.skel_radius,
        warmup_start=args.warmup_start, warmup_ramp=args.warmup_ramp,
        mix_w=args.mix_w,
    )
    # Searched θs and mix_w override the fixed defaults in the pinned overlay
    # (the best trial's values, like lr/pos_weight/batch).
    for _k in ("tl_theta", "gap_theta", "mix_w"):
        if _k in study.best_params:
            loss_hp[_k] = study.best_params[_k]
    overlay_path = write_best_overlay(study, out_dir, encoder_weights, upsampler,
                                      freeze_sr, sr_pad, loss_arm, loss_hp,
                                      warm_start_unet, precision=args.precision,
                                      mask_source=mask_source,
                                      mask_dirname=mask_dirname,
                                      lr_schedule=lr_schedule,
                                      sr_warmup_epochs=sr_warmup_epochs,
                                      l2sp_lambda=l2sp_lambda,
                                      adaptive_norm=adaptive_norm,
                                      adaptive_norm_momentum=adaptive_norm_momentum,
                                      norm_recalibrate=norm_recalibrate,
                                      head=head,
                                      warm_start_head=warm_start_head,
                                      clip_sr=clip_sr,
                                      sr_hc=sr_hc,
                                      hc_mask_path=hc_mask_path,
                                      monitor=args.monitor)
    print(f"\nBest {args.monitor}={study.best_value:.4f} (trial #{study.best_trial.number})")
    print(f"Best params: {study.best_params}  encoder_weights={encoder_weights}")
    print(f"Wrote Lightning overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
