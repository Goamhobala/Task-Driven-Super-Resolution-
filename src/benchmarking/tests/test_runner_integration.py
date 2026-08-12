"""Integration tests for the benchmark runner + sharded store.

Real checkpoints (random-init, saved in Lightning's on-disk format), real
GeoTIFF fixtures, real ``evaluate()`` calls — this is the laptop-side proof
that the eval path works before it ever touches the cluster. The fixture tiles
are DELIBERATELY not multiples of the chip size (ragged edge chips) and not
/32 (smp-UNet's decoder constraint), because those were the paths that had
never seen a real model.

The tp+fn-vs-mask invariant runs with ``check="all"`` in every evaluate call,
so a windowing/GT bug fails the test inside the runner itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from benchmarking import store as bstore
from benchmarking.runner import (
    SRPredictor,
    _config_hash,
    _grid,
    evaluate,
)

# Fixture geometry: 10 m pixels, cell_m=320 -> chip_px=32. Tiles are 80 x 96:
# rows = [32, 32, 16] (ragged), cols = [32, 32, 32] -> 9 chips/tile, 3 ragged.
CELL_M = 320.0
CHIP = 32
H, W = 80, 96
UPSCALE = 4
CRS = "EPSG:32734"
NORM = {"norm_mean": [100.0, 110.0, 120.0, 130.0], "norm_std": [50.0, 55.0, 60.0, 65.0]}
BANDS = (1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_tif(path, arr, transform, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
        count=count, dtype=arr.dtype, crs=CRS, transform=transform,
    ) as dst:
        dst.write(arr if arr.ndim == 3 else arr[None, ...])


def _make_dataset(root: Path, n_tiles=2, seed=0) -> Path:
    """Minimal ROSA-shaped dataset: splits/test.csv + imagery + 10 m masks +
    exactly-4x HR masks (the SR raster GT)."""
    rng = np.random.default_rng(seed)
    ds = root / "dataset"
    rows = []
    for i in range(n_tiles):
        tf = from_origin(500_000 + i * W * 10.0, 7_000_000, 10.0, 10.0)
        img = rng.integers(0, 4000, size=(4, H, W)).astype("uint16")
        mask = (rng.random((H, W)) < 0.25).astype("uint8")
        hr_mask = (rng.random((H * UPSCALE, W * UPSCALE)) < 0.25).astype("uint8")
        hr_tf = from_origin(500_000 + i * W * 10.0, 7_000_000, 2.5, 2.5)
        _write_tif(ds / "test" / "imagery" / f"tile{i}.tif", img, tf, 4)
        _write_tif(ds / "test" / "masks_raster" / f"tile{i}.tif", mask, tf, 1)
        _write_tif(ds / "test" / "mask_osm_2pt5" / f"tile{i}.tif", hr_mask, hr_tf, 1)
        rows.append({
            "zone_name": "zone", "image_path": f"test/imagery/tile{i}.tif",
            "mask_path": f"test/masks_raster/tile{i}.tif",
            "mask_graph_path": f"test/masks_graph/tile{i}.parquet",
        })
    (ds / "splits").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(ds / "splits" / "test.csv", index=False)
    return ds


def _save_ckpt(model, path: Path) -> Path:
    """Write a loadable Lightning checkpoint without spinning up a Trainer."""
    import lightning.pytorch as pl

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": pl.__version__,
        "epoch": 0,
        "global_step": 0,
    }, path)
    return path


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return _make_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="module")
def unet_ckpt(tmp_path_factory):
    from unet.model import UNetLightning

    torch.manual_seed(0)
    model = UNetLightning(encoder_name="resnet18", encoder_weights=None,
                          in_channels=4, bands=BANDS, **NORM)
    return _save_ckpt(model, tmp_path_factory.mktemp("ckpt") / "unet.ckpt")


@pytest.fixture(scope="module")
def sr_ckpt(tmp_path_factory):
    from sr.model import JointSRUNetLightning

    torch.manual_seed(0)
    model = JointSRUNetLightning(encoder_name="resnet18", encoder_weights=None,
                                 in_channels=4, bands=BANDS, upsampler="bicubic",
                                 freeze_sr=False, upscale=UPSCALE, **NORM)
    return _save_ckpt(model, tmp_path_factory.mktemp("ckpt") / "sr.ckpt")


def _road_px(ds, rel):
    with rasterio.open(ds / rel) as src:
        return int((src.read(1) > 0).sum())


# --------------------------------------------------------------------------- #
# unet family
# --------------------------------------------------------------------------- #
def test_unet_end_to_end(dataset, unet_ckpt, tmp_path):
    store = tmp_path / "store"
    run_id = evaluate(
        dataset_dir=dataset, checkpoint=unet_ckpt, model_name="unet_test", seed=0,
        store_dir=store, model="unet", cell_m=CELL_M, batch_size=4,
        tile_metrics=("road_frac",), check="all", device="cpu",
        label_source="cdngi", exp_tag="cdngi",
    )

    chips = bstore.load_chips(store)
    runs = bstore.load_runs(store)
    tiles = bstore.load_tiles(store)

    assert len(chips) == 2 * 9  # 3x3 footprint cells per tile, ragged row included
    assert set(chips["run_id"]) == {run_id}

    # tp+fn == GT road pixels, per tile (check="all" already enforced this
    # inside the runner; re-assert from the stored rows).
    for i in range(2):
        got = chips[chips["tile_id"] == f"tile{i}"][["tp", "fn"]].to_numpy().sum()
        assert got == _road_px(dataset, f"test/masks_raster/tile{i}.tif")

    # every chip covers its true (unpadded) pixel count
    edge = chips[chips["patch_row_id"] == 2]
    assert ((edge[["tp", "fp", "fn", "tn"]].sum(axis=1)) == 16 * 32).all()

    run = runs.iloc[0]
    assert run["model_family"] == "unet"
    assert run["chip_px"] == CHIP
    assert run["gt_res_m"] == pytest.approx(10.0)
    assert run["n_chips"] == 18

    # plugin seam: tile rows + per-chip merge
    assert {"pred_road_frac", "gt_road_frac"} <= set(tiles.columns)
    assert len(tiles) == 2
    assert chips["gt_road_frac"].notna().all()
    tile0 = chips[chips["tile_id"] == "tile0"]
    frac = tile0["gt_road_frac"].mul(tile0[["tp", "fp", "fn", "tn"]].sum(axis=1)).sum() / (H * W)
    assert frac == pytest.approx(tiles[tiles["tile_id"] == "tile0"]["gt_road_frac"].iloc[0])


def test_unet_metrics_in_range(dataset, unet_ckpt, tmp_path):
    store = tmp_path / "store"
    evaluate(dataset_dir=dataset, checkpoint=unet_ckpt, model_name="m", seed=0,
             store_dir=store, model="unet", cell_m=CELL_M, check="first", device="cpu")
    chips = bstore.load_chips(store)
    for col in ("iou", "f1", "precision", "recall", "accuracy"):
        v = chips[col].dropna()
        assert ((v >= 0) & (v <= 1)).all()
    assert (chips["inference_ms"] > 0).all()


# --------------------------------------------------------------------------- #
# sr family
# --------------------------------------------------------------------------- #
def test_sr_end_to_end_raster(dataset, sr_ckpt, unet_ckpt, tmp_path):
    store = tmp_path / "store"
    evaluate(
        dataset_dir=dataset, checkpoint=unet_ckpt, model_name="unet_test", seed=0,
        store_dir=store, model="unet", cell_m=CELL_M, check="first", device="cpu",
    )
    evaluate(
        dataset_dir=dataset, checkpoint=sr_ckpt, model_name="sr_test", seed=0,
        store_dir=store, model="sr", cell_m=CELL_M, mask_source="raster",
        mask_dirname="mask_osm_2pt5", check="all", device="cpu",
    )
    chips = bstore.load_chips(store)
    a = set(chips[chips["model_name"] == "unet_test"]["chip_id"])
    b = set(chips[chips["model_name"] == "sr_test"]["chip_id"])
    # THE footprint property: same geography -> same chip_id at both resolutions.
    assert a == b and len(a) == 18

    sr_chips = chips[chips["model_name"] == "sr_test"]
    # scored at HR: pixel counts are (4h x 4w) of the REAL extent, no pad pixels
    for i in range(2):
        tot = sr_chips[sr_chips["tile_id"] == f"tile{i}"][["tp", "fp", "fn", "tn"]].to_numpy().sum()
        assert tot == (H * UPSCALE) * (W * UPSCALE)
        got = sr_chips[sr_chips["tile_id"] == f"tile{i}"][["tp", "fn"]].to_numpy().sum()
        assert got == _road_px(dataset, f"test/mask_osm_2pt5/tile{i}.tif")

    runs = bstore.load_runs(store)
    sr_run = runs[runs["model_name"] == "sr_test"].iloc[0]
    assert sr_run["model_family"] == "sr"
    assert sr_run["gt_res_m"] == pytest.approx(2.5)


def test_sr_pinned_subwindow_stitch(dataset, sr_ckpt):
    """Emulate SEN2SR's pinned LR input on the bicubic checkpoint: a 32 px cell
    run as 2x2 sub-windows of 16 px must stitch to the full HR cell shape."""
    pred = SRPredictor(sr_ckpt, device="cpu")
    pred.required_lr = 16  # pretend the FFT mask pins 16 px
    img = np.random.default_rng(0).uniform(0, 3000, size=(4, 32, 32)).astype("float32")
    probs, ms = pred.predict_probs(img, 32)
    assert probs.shape == (1, 1, 128, 128)
    assert torch.isfinite(probs).all() and 0.0 <= probs.min() and probs.max() <= 1.0

    with pytest.raises(ValueError, match="not a multiple"):
        pred.required_lr = 15
        pred.predict_probs(img, 32)


def test_sr_graph_mask_source(dataset, sr_ckpt, tmp_path):
    """mask_source='graph': rasterised parquet centrelines as HR GT, invariant on."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    for i in range(2):
        x0 = 500_000 + i * W * 10.0
        gdf = gpd.GeoDataFrame(
            {"buffer": [15.0, 10.0]},
            geometry=[
                LineString([(x0 + 50, 6_999_900), (x0 + W * 10 - 50, 6_999_450)]),
                LineString([(x0 + 100, 6_999_950), (x0 + 300, 6_999_250)]),
            ],
            crs=CRS,
        )
        out = dataset / "test" / "masks_graph"
        out.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(out / f"tile{i}.parquet")

    store = tmp_path / "store"
    evaluate(
        dataset_dir=dataset, checkpoint=sr_ckpt, model_name="sr_graph", seed=0,
        store_dir=store, model="sr", cell_m=CELL_M, mask_source="graph",
        check="all", device="cpu",  # invariant: per-window rasterise == full-tile
    )
    chips = bstore.load_chips(store)
    assert len(chips) == 18
    assert chips[["tp", "fn"]].to_numpy().sum() > 0  # the lines rasterised to roads


# --------------------------------------------------------------------------- #
# store + helpers
# --------------------------------------------------------------------------- #
def test_store_shards_and_append_only(tmp_path):
    store = tmp_path / "store"
    bstore.append_run({"run_id": "a", "model_name": "m"}, store)
    bstore.append_run({"run_id": "b", "model_name": "m"}, store)
    assert len(bstore.load_runs(store)) == 2
    with pytest.raises(FileExistsError):
        bstore.append_run({"run_id": "a", "model_name": "m"}, store)

    chips = pd.DataFrame({"run_id": ["a", "a"], "chip_id": ["c0", "c1"], "iou": [0.5, 0.6]})
    bstore.append_chips(chips, store)
    assert len(bstore.load_chips(store)) == 2
    with pytest.raises(ValueError, match="single run_id"):
        bstore.append_chips(pd.DataFrame({"run_id": ["x", "y"], "iou": [0.1, 0.2]}), store)


def test_store_reads_legacy_flat_files(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    pd.DataFrame({"run_id": ["legacy"], "chip_id": ["c0"], "iou": [0.4]}).to_parquet(
        store / "chip_metrics.parquet", index=False)
    bstore.append_chips(
        pd.DataFrame({"run_id": ["new"], "chip_id": ["c0"], "iou": [0.7]}), store)
    df = bstore.load_chips(store)
    assert set(df["run_id"]) == {"legacy", "new"}


def test_config_hash_is_canonical():
    a = "lr: 0.001\nencoder: resnet34\n"
    b = "encoder: resnet34\nlr: 0.001\n"  # same config, different key order
    assert _config_hash(a) == _config_hash(b) != ""
    assert _config_hash("lr: 0.002\nencoder: resnet34\n") != _config_hash(a)
    assert _config_hash("") == ""


def test_grid_partitions_ragged_tiles():
    cells = _grid(80, 96, 32)
    assert len(cells) == 9
    assert sum(h * w for *_, h, w in cells) == 80 * 96
    ids = {(ri, ci) for ri, ci, *_ in cells}
    assert ids == {(r, c) for r in range(3) for c in range(3)}


# --------------------------------------------------------------------------- #
# buffered F1 (benchmarking.buffered_metrics) through the real scoring path
# --------------------------------------------------------------------------- #
def test_buffer_px_adds_columns_and_is_sweepable(dataset, unet_ckpt, tmp_path):
    """The columns must appear in BOTH modes, and — the part that matters —
    sweep mode must produce them, because that is what lets theta_sweep_bench
    select θ* on buffered F1. The tile-metric plugins cannot do this: they score
    a stitched tile and evaluate() rejects them outright in sweep mode."""
    store = tmp_path / "store"
    evaluate(dataset_dir=dataset, checkpoint=unet_ckpt, model_name="buf", seed=0,
             store_dir=store, model="unet", cell_m=CELL_M, check="first",
             device="cpu", buffer_px=3)
    chips = bstore.load_chips(store)
    cols = {"buffered_f1", "buffered_precision", "buffered_recall"}
    assert cols <= set(chips.columns)
    for col in cols:
        v = chips[col].dropna()
        assert ((v >= 0) & (v <= 1)).all(), f"{col} out of range"

    swept = evaluate(
        dataset_dir=dataset, checkpoint=unet_ckpt, model_name="buf", seed=0,
        store_dir=None, model="unet", cell_m=CELL_M, check="off", device="cpu",
        sweep_thresholds=[0.3, 0.5, 0.7], buffer_px=3)
    assert set(swept) == {0.3, 0.5, 0.7}
    for df in swept.values():
        assert cols <= set(df.columns)


def test_buffered_f1_at_least_strict_f1(dataset, unet_ckpt, tmp_path):
    """A buffer can only ever forgive, never punish: relaxing the match at
    rho=3 must not score below the strict pixel F1 on the same chips. This is
    the cheapest guard against the buffer being applied to the wrong mask."""
    store = tmp_path / "store"
    evaluate(dataset_dir=dataset, checkpoint=unet_ckpt, model_name="buf2", seed=0,
             store_dir=store, model="unet", cell_m=CELL_M, check="off",
             device="cpu", buffer_px=3)
    chips = bstore.load_chips(store)
    both = chips[chips["buffered_f1"].notna() & chips["f1"].notna()]
    assert len(both) > 0
    assert (both["buffered_f1"] >= both["f1"] - 1e-9).all()


def test_sweep_gt_distance_cache_matches_uncached(dataset, unet_ckpt):
    """The sweep hoists the GT distance transform out of the θ loop. Scoring one
    θ via the sweep path and via the normal path must agree exactly."""
    swept = evaluate(
        dataset_dir=dataset, checkpoint=unet_ckpt, model_name="c", seed=0,
        store_dir=None, model="unet", cell_m=CELL_M, check="off", device="cpu",
        sweep_thresholds=[0.5], buffer_px=3)[0.5]
    direct = evaluate(
        dataset_dir=dataset, checkpoint=unet_ckpt, model_name="c", seed=0,
        store_dir=None, model="unet", cell_m=CELL_M, check="off", device="cpu",
        sweep_thresholds=[0.5], buffer_px=None)[0.5]
    # same chips, same order
    assert list(swept["chip_id"]) == list(direct["chip_id"])
    assert swept["buffered_f1"].notna().any()
    # and the pixel columns are untouched by the buffer flag
    for col in ("iou", "f1", "tp", "fp", "fn"):
        assert swept[col].to_numpy() == pytest.approx(direct[col].to_numpy())
