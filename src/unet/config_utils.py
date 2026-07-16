"""Shared config plumbing for the non-LightningCLI unet entry points
(``unet.tune``, ``unet.train_ablation``): load/merge the same YAML base
configs ``unet.cli`` consumes and resolve the CLI-vs-config overrides.

Kept free of heavy imports (no optuna/lightning) so the ablation trainer can
use it without pulling in the tuning stack.
"""
from __future__ import annotations

import yaml


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base`` (overlay wins), like Lightning."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_base_config(paths: list[str]) -> dict:
    """Deep-merge one or more YAML configs the same way ``--config a --config b`` does."""
    merged: dict = {}
    for p in paths:
        with open(p) as fh:
            merged = deep_merge(merged, yaml.safe_load(fh) or {})
    return merged


def data_kwargs(cfg: dict, dataset_dir: str | None, num_workers: int | None,
                mask_dirname: str | None = None) -> dict:
    """Pull the fixed RoadDataModule args from the base config (CLI overrides win)."""
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
