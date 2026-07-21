"""Optuna search for the Phase B mixing ratio (protocol amendment 2026-07-21).

Searches ``mix_w`` — the P*↔region ratio in ``(1−mw)·P* + mw·region`` — for
one compound arm (``pstar_dice`` / ``pstar_tversky``). The ratio is the
compound class's ONE genuinely free parameter (no paper default exists), so
searching it makes Decision B faithful: a fixed-0.5 null would only mean "the
region slot doesn't help *at 0.5*". Optionally searches ``tversky_alpha``
jointly (2D is where TPE beats a grid).

Fairness rules, mirrored from unet.tune / the protocol:
  * LR is NOT searched — protocol Appendix B fixes the screening LR (1e-3),
    valid across arms via the §4.4 scale normalization. Encoder, batch size,
    bands: from the base config, fixed.
  * The SAME --train-seed seeds every trial, so the crop/augmentation stream
    is identical across trials and the objective differences are loss-driven
    (--seed only decorrelates parallel workers' TPE samplers).
  * Objective = best val_f1 within the trial (the protocol's selection
    metric), median-pruned.
  * Phase C arms ('+cldice'/'+skelrec') are REFUSED: the skeleton warmup
    starts at epoch 30, so a short trial never activates the skeleton term —
    tuning it here would tune a loss the trial never trains. Phase C uses the
    pre-registered grid (or trunk-branch screening + from-scratch refit).

The best trial lands in ``<out>/best_loss_params.yaml``; the loss engine's
STAGE=fit reads it automatically when MIX_W is not set explicitly, then the
standard fixed-budget fit + θ* sweep + bench produce the decision numbers —
the search never supplies them directly.

    python -m unet.tune_loss \
        --base-config src/unet/configs/unet.yaml \
        --base-config src/unet/configs/norm_stats.yaml \
        --dataset-dir /scratch/$USER/InstaRoad/ROSA_all \
        --arm pstar_dice --pstar bce \
        --n-trials 30 --max-epochs 8 --out runs/loss_l5_pstar_dice_seed0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-config", action="append", default=[], metavar="YAML")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--mask-dirname", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out", required=True, help="Run dir (best_loss_params.yaml + study).")
    # the arm under search (compound; its P* hyperparameters passed through)
    ap.add_argument("--arm", required=True, help="pstar_dice | pstar_tversky")
    ap.add_argument("--pstar", default="bce")
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--gap-r", type=int, default=4)
    ap.add_argument("--gap-k", type=float, default=60.0)
    ap.add_argument("--tl-ell", type=int, default=5)
    ap.add_argument("--tl-theta", type=float, default=0.375)
    ap.add_argument("--tversky-alpha", type=float, default=0.7,
                    help="fixed value when --search-tversky-alpha is off")
    # search space (pre-registered; change only by dated amendment)
    ap.add_argument("--mix-min", type=float, default=0.2)
    ap.add_argument("--mix-max", type=float, default=0.8)
    ap.add_argument("--search-tversky-alpha", action="store_true",
                    help="also search tversky_alpha (pstar_tversky only)")
    ap.add_argument("--tversky-alpha-min", type=float, default=0.5)
    ap.add_argument("--tversky-alpha-max", type=float, default=0.9)
    # study controls (mirroring unet.tune)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--study-name", default="loss_mix")
    ap.add_argument("--storage", default=None,
                    help="e.g. sqlite:///<run_dir>/study.db (enables parallel workers/resume)")
    ap.add_argument("--seed", type=int, default=42, help="TPE sampler seed (differs per worker)")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="seed_everything for EVERY trial: identical data stream "
                         "across trials -> objective differences are loss-driven")
    # per-trial budget (protocol-fixed LR; short proxy)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="FIXED screening LR (Appendix B)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: base config")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="1")
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args(argv)


MONITOR = "val_f1"  # the protocol's selection metric (not val_iou)


def build_objective(args, base_cfg):
    import lightning.pytorch as pl
    import optuna  # noqa: F401  (Trial type at runtime)

    from sentinel2data.dataset.datasets import RoadDataModule
    from unet.config_utils import data_kwargs, resolve_encoder_weights
    from unet.model import UNetLightning
    from unet.tune import PyTorchLightningPruningCallback, _resolve_devices

    data_cfg = data_kwargs(base_cfg, args.dataset_dir, args.num_workers, args.mask_dirname)
    model_cfg = dict(base_cfg.get("model", {}))
    bands = tuple(data_cfg["bands"])
    devices = _resolve_devices(args.devices)
    encoder_weights = resolve_encoder_weights(base_cfg, None)

    def objective(trial):
        mix_w = trial.suggest_float("mix_w", args.mix_min, args.mix_max)
        tv_alpha = (trial.suggest_float("tversky_alpha", args.tversky_alpha_min,
                                        args.tversky_alpha_max)
                    if args.search_tversky_alpha else args.tversky_alpha)

        pl.seed_everything(args.train_seed, workers=True)  # same stream every trial

        dm = RoadDataModule(
            dataset_dir=data_cfg["dataset_dir"], bands=bands,
            batch_size=args.batch_size or data_cfg.get("batch_size", 16),
            num_workers=data_cfg.get("num_workers", 0),
            image_size=data_cfg.get("image_size", 256),
            length=data_cfg.get("length"),
            normalize=data_cfg.get("normalize", True),
            norm_mean=data_cfg["norm_mean"], norm_std=data_cfg["norm_std"],
            mask_dirname=data_cfg.get("mask_dirname"),
            aug_flip=args.augment,
        )
        model = UNetLightning(
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=encoder_weights,
            in_channels=len(bands), classes=model_cfg.get("classes", 1),
            lr=args.lr,                       # FIXED — not part of the search
            bands=bands, image_size=data_cfg.get("image_size", 256),
            threshold=0.5, normalize=data_cfg.get("normalize", True),
            norm_mean=data_cfg["norm_mean"], norm_std=data_cfg["norm_std"],
            loss_arm=args.arm, pstar=args.pstar, pos_weight=args.pos_weight,
            gap_r=args.gap_r, gap_k=args.gap_k,
            tl_ell=args.tl_ell, tl_theta=args.tl_theta,
            tversky_alpha=tv_alpha, mix_w=mix_w,
        )
        pruning_cb = PyTorchLightningPruningCallback(trial, monitor=MONITOR)
        trainer = pl.Trainer(
            max_epochs=args.max_epochs, accelerator=args.accelerator,
            devices=devices, strategy="auto", precision=args.precision,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            log_every_n_steps=10, callbacks=[pruning_cb],
        )
        trainer.fit(model, datamodule=dm)
        pruning_cb.check_pruned()
        value = trainer.callback_metrics.get(MONITOR)
        if value is None:
            raise RuntimeError(f"'{MONITOR}' was never logged; cannot score the trial.")
        return float(value)

    return objective


def main(argv=None):
    args = parse_args(argv)
    if not args.base_config:
        raise SystemExit("Pass --base-config unet.yaml + norm_stats.yaml.")
    if "+" in args.arm:
        raise SystemExit(
            f"--arm {args.arm}: Phase C skeleton compounds cannot be tuned on short "
            "trials — the §4.5 warmup means an 8-epoch trial never activates the "
            "skeleton term. Use the pre-registered grid (or trunk-branch screening "
            "with a from-scratch refit of the winner).")
    if args.arm not in ("pstar_dice", "pstar_tversky"):
        raise SystemExit(f"--arm {args.arm}: mix_w search applies to pstar_dice | "
                         "pstar_tversky (the bce_dice/focal_tversky anchors stay frozen).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from unet.config_utils import load_base_config
    from unet.tune import create_study_shared

    base_cfg = load_base_config(args.base_config)
    study = create_study_shared(args.study_name, args.storage, args.seed)

    import torch

    study.optimize(build_objective(args, base_cfg), n_trials=args.n_trials,
                   timeout=args.timeout, gc_after_trial=True,
                   catch=(torch.cuda.OutOfMemoryError,))

    best = dict(study.best_params)
    payload = {
        "arm": args.arm, "pstar": args.pstar,
        "mix_w": best["mix_w"],
        "tversky_alpha": best.get("tversky_alpha", args.tversky_alpha),
        "best_val_f1_8ep": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "train_seed": args.train_seed, "tune_epochs": args.max_epochs,
    }
    out_path = out_dir / "best_loss_params.yaml"
    out_path.write_text(
        "# Phase B mixing-ratio search (unet.tune_loss). STAGE=fit reads mix_w\n"
        "# from here when MIX_W is unset; the decision numbers come from the\n"
        "# full-budget refit, never from these short trials.\n"
        + yaml.safe_dump(payload, sort_keys=False))

    # full λ-vs-F1 curve: the plateau (or not) is a reportable result either way
    rows = [{"number": t.number, "state": str(t.state), "value": t.value, **t.params}
            for t in study.trials]
    (out_dir / "mix_trials.json").write_text(json.dumps(rows, indent=1))

    print(f"best {MONITOR}={study.best_value:.4f} @ trial {study.best_trial.number}: {best}")
    print(f"wrote {out_path} and mix_trials.json ({len(rows)} trials)")


if __name__ == "__main__":
    main()
