"""Integration tests for the loss-ablation migration to the unet pipeline.

Covers the seams the unit tests in test_losses.py don't:
  * UNetLightning loss_arm wiring (criterion build, hparams round-trip through
    a Lightning checkpoint — what benchmarking's UNetPredictor relies on).
  * Augmentation inside RoadTileDataset / RoadDataModule (image and mask must
    transform together; val/test never augmented).
  * unet.train_ablation end-to-end on a tiny fixture dataset (1 epoch, CPU):
    fixed-budget fit -> val-F1 checkpoint -> threshold sweep artifacts.
  * benchmarking.runner.evaluate on the ablation checkpoint at the tuned θ*
    (the --threshold override), invariant check "all".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

from sentinel2data.dataset.datasets import RoadDataModule, RoadTileDataset

# Fixture geometry: 512x512 tiles at 10 m (so TileCropDataset's fixed 2x2
# quadrant crops and the benchmark's default 2560 m cell both fit exactly).
H = W = 512
CRS = "EPSG:32734"
BANDS = (1, 2)
NORM = {"norm_mean": [500.0, 200.0], "norm_std": [400.0, 150.0]}


def _write_tif(path, arr, transform, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
        count=count, dtype=arr.dtype, crs=CRS, transform=transform,
    ) as dst:
        dst.write(arr if arr.ndim == 3 else arr[None, ...])


def _road_mask(rng):
    """Sparse synthetic roads: a few 2-px horizontal/vertical lines."""
    m = np.zeros((H, W), dtype="uint8")
    for r in rng.integers(20, H - 20, size=4):
        m[r:r + 2, :] = 1
    for c in rng.integers(20, W - 20, size=4):
        m[:, c:c + 2] = 1
    return m


def _make_dataset(root: Path, n_tiles=2, seed=0) -> Path:
    """ROSA-shaped fixture with train + val splits. Band 1 leaks the mask
    (mask*1000 + noise) so even a 1-epoch model has signal; band 2 is noise."""
    rng = np.random.default_rng(seed)
    ds = root / "dataset"
    rows = {"train": [], "val": []}
    for split in ("train", "val"):
        for i in range(n_tiles):
            tf = from_origin(500_000 + i * W * 10.0, 7_000_000, 10.0, 10.0)
            mask = _road_mask(rng)
            img = np.stack([
                mask.astype("uint16") * 1000 + rng.integers(0, 200, (H, W)).astype("uint16"),
                rng.integers(0, 400, (H, W)).astype("uint16"),
            ])
            _write_tif(ds / split / "imagery" / f"{split}tile{i}.tif", img, tf, 2)
            _write_tif(ds / split / "masks_raster" / f"{split}tile{i}.tif", mask, tf, 1)
            rows[split].append({
                "zone_name": f"zone{i}",
                "image_path": f"{split}/imagery/{split}tile{i}.tif",
                "mask_path": f"{split}/masks_raster/{split}tile{i}.tif",
            })
    (ds / "splits").mkdir(parents=True)
    for split, r in rows.items():
        pd.DataFrame(r).to_csv(ds / "splits" / f"{split}.csv", index=False)
    return ds


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return _make_dataset(tmp_path_factory.mktemp("data"))


# ------------------------------------------------------ model loss wiring
def test_loss_arm_builds_criterion_and_legacy_default_unchanged():
    from unet.model import UNetLightning

    armed = UNetLightning(encoder_name="resnet18", encoder_weights=None,
                          in_channels=2, bands=BANDS, loss_arm="gap_ce",
                          gap_r=5, **NORM)
    assert armed.criterion is not None and armed.dice_loss is None
    legacy = UNetLightning(encoder_name="resnet18", encoder_weights=None,
                           in_channels=2, bands=BANDS, **NORM)
    assert legacy.criterion is None and legacy.dice_loss is not None

    # Armed loss runs and backprops on a real batch shape.
    x = torch.randn(2, 2, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.9).float()
    loss = armed._loss(armed(x), y)
    loss.backward()
    assert torch.isfinite(loss)


def test_armed_checkpoint_hparams_roundtrip(tmp_path):
    """load_from_checkpoint must rebuild the criterion from hparams — this is
    what lets benchmarking's UNetPredictor load ablation checkpoints."""
    import lightning.pytorch as pl
    from unet.model import UNetLightning

    model = UNetLightning(encoder_name="resnet18", encoder_weights=None,
                          in_channels=2, bands=BANDS,
                          loss_arm="bce_dice+cldice", cl_alpha=0.3,
                          warmup_start=3, warmup_ramp=2, **NORM)
    path = tmp_path / "armed.ckpt"
    torch.save({"state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__,
                "epoch": 0, "global_step": 0}, path)

    loaded = UNetLightning.load_from_checkpoint(path, map_location="cpu")
    assert loaded.hparams.loss_arm == "bce_dice+cldice"
    assert loaded.criterion is not None
    assert loaded.criterion.warmup_start == 3
    # §4.5 convex mix: (1-α)·anchor + α·clDice
    assert abs(loaded.criterion.w_skel - 0.3) < 1e-9


# ------------------------------------------------------------ augmentation
def test_train_dataset_transform_keeps_image_and_mask_aligned(dataset):
    """Band 1 == mask (un-normalised), so under a geometric transform the
    equality must survive — proof image and mask are transformed together."""
    from sentinel2data.dataset.augment import build_transform

    ds = RoadTileDataset(dataset, bands=BANDS, image_size=64, length=16,
                         normalize=False, transform=build_transform(flip=True),
                         **NORM)
    moved = 0
    for i in range(len(ds)):
        img, mask, _ = ds[i]
        band_road = (img[0] > 500).float()          # band1 = mask*1000 + noise
        assert torch.equal(band_road, mask[0]), "image/mask transformed apart"
        moved += int(mask.sum() > 0)
    assert moved > 0  # sanity: the crops actually contained roads


def test_datamodule_augments_train_only(dataset):
    dm_plain = RoadDataModule(dataset_dir=dataset, bands=BANDS, batch_size=2,
                              num_workers=0, **NORM)
    assert dm_plain._train_transform() is None
    dm_aug = RoadDataModule(dataset_dir=dataset, bands=BANDS, batch_size=2,
                            num_workers=0, aug_flip=True, aug_noise=True, **NORM)
    tf = dm_aug._train_transform()
    assert tf is not None and len(tf.transforms) == 2
    assert dm_aug.train_dataloader().dataset.transform is not None
    # val/test datasets have no transform seam at all — nothing to assert on
    # beyond the loader building fine.
    assert dm_aug.val_dataloader() is not None


# ------------------------------------------- train_ablation + benchmarking
@pytest.fixture(scope="module")
def ablation_run(dataset, tmp_path_factory):
    """One-epoch Phase A-style run on the fixture dataset (CPU)."""
    from unet import train_ablation

    out = tmp_path_factory.mktemp("run")
    base = out / "base.yaml"
    base.write_text(yaml.safe_dump({
        "model": {"encoder_name": "resnet18", "encoder_weights": None, "classes": 1},
        "data": {"dataset_dir": str(dataset), "bands": list(BANDS),
                 "batch_size": 2, "num_workers": 0, "image_size": 256,
                 "length": 4, "normalize": True, **NORM},
    }))
    train_ablation.main([
        "--base-config", str(base), "--out", str(out),
        "--arm", "bce_dice", "--seed", "0", "--epochs", "1",
        "--accelerator", "cpu", "--precision", "32-true",
        "--wandb-mode", "disabled",
    ])
    return out


def test_train_ablation_artifacts(ablation_run):
    meta = json.loads((ablation_run / "train_meta.json").read_text())
    sweep = json.loads((ablation_run / "sweep.json").read_text())
    assert meta["arm"] == "bce_dice" and meta["seed"] == 0
    assert Path(meta["checkpoint"]).exists()
    assert 0.0 < float(meta["best_threshold"]) < 1.0
    assert len(sweep["sweep"]) == 19          # θ = 0.05 … 0.95
    assert meta["f1_at_best_threshold"] == pytest.approx(
        max(v["f1"] for v in sweep["sweep"].values()))
    # resolved config doubles as the benchmark --config-yaml input
    cfg = yaml.safe_load((ablation_run / "config.yaml").read_text())
    assert cfg["selection"] == "val_f1@0.5"


def test_benchmark_scores_ablation_ckpt_at_tuned_threshold(ablation_run, dataset, tmp_path):
    """The phase_a.sh chain: eval the ablation checkpoint on val at θ*."""
    from benchmarking.runner import evaluate
    from benchmarking.store import load_chips, load_runs

    meta = json.loads((ablation_run / "train_meta.json").read_text())
    theta = float(meta["best_threshold"])
    store = tmp_path / "store"
    run_id = evaluate(
        dataset_dir=dataset, checkpoint=meta["checkpoint"],
        model_name="unet_bce_dice", seed=0, store_dir=store, split="val",
        model="unet", exp_tag="phase_a", label_source="cdngi",
        config_yaml_path=ablation_run / "config.yaml",
        threshold=theta, check="all", device="cpu",
    )
    runs = load_runs(store)
    assert runs.loc[runs.run_id == run_id, "threshold"].item() == pytest.approx(theta)
    assert runs.loc[runs.run_id == run_id, "config_hash"].item() != ""
    chips = load_chips(store)
    # 512 px tile / 256 px cell -> 4 chips per tile, 2 val tiles
    assert len(chips) == 8
    assert chips["f1"].between(0, 1).all()


def test_threshold_override_validates():
    from benchmarking.runner import evaluate

    with pytest.raises(ValueError, match="threshold"):
        evaluate(dataset_dir=".", checkpoint="x", model_name="m", seed=0,
                 store_dir=".", threshold=1.5)
