# Benchmarking Schema

Reference for the on-disk schema used by the benchmarking module to record model evaluation results. This document covers **what is stored** and **how to query it**. The runner, model loader, and CLI are described separately.

## Overview

Benchmarking results are stored in **two parquet tables** that join on `run_id`:

- `runs.parquet` — one row per `(model, config, seed)` evaluation. Holds metadata that is constant across all tiles in a run: model identity, training config, loss function, training losses, checkpoint path, dataset split, evaluation resolution.
- `tile_metrics.parquet` — one row per `(run_id, tile_id)`. Holds per-tile measurements: confusion-matrix counts, derived pixel metrics, the graph metric (APLS), and per-tile inference latency.

The split is deliberate. Run-level fields would otherwise be repeated for every tile (wasteful and easy to corrupt). Per-tile fields are kept long-form (one row per measurement) so the data is ready for `groupby`, `pivot`, bootstrap resampling, and paired statistical tests without further reshaping.

A typical analysis joins the two tables, filters on a run-level attribute (e.g. `loss_fn`, `seed`), then aggregates per-tile rows.

## System architecture

One benchmark run is a single `(model, seed)` evaluation that consumes a trained checkpoint plus a dataset split and writes rows to both parquet tables. The stats module operates on those rows after the fact and never participates in inference.

```
                       ┌─────────────────┐
                       │  Training (ext) │
                       └────────┬────────┘
                                │ checkpoint + sidecar (train/val losses)
                                v
       ┌──────────────┐   ┌──────────────┐
       │  CLI / YAML  │──>│ ModelLoader  │
       │   (config)   │   │  (registry)  │
       └──────────────┘   └──────┬───────┘
                                 │ Predictor.predict
                                 v
       ┌──────────────┐   ┌──────────────────────────────────┐
       │  sentinel2   │   │              Runner              │
       │  manifest +  │──>│  per tile:                       │
       │  masks (ext) │   │   1. DataLoader yields (img, gt) │
       └──────────────┘   │   2. pred = predictor.predict    │
                          │   3. counts = confusion_counts   │ ──> metrics.py
                          │   4. pixel  = derive_metrics     │     (pure)
                          │   5. apls   = apls_metric        │
                          │   6. append to tile_metrics      │
                          │  finalise runs row               │
                          └────────────────┬─────────────────┘
                                           │
                                           v
                                  ┌──────────────────┐
                                  │      store       │
                                  │  (parquet I/O)   │
                                  └────────┬─────────┘
                                           │
                                           v
                                  runs.parquet
                                  tile_metrics.parquet
                                           │
                                           v
                                  ┌──────────────────┐
                                  │      stats       │
                                  │  cross_seed_ci   │ ──> CIs, p-values
                                  │  bootstrap_pair  │     for the loss pilot
                                  │  wilcoxon_pair   │     and final reports
                                  └──────────────────┘
```

### External producers

Out of scope for this module; benchmarking consumes their outputs but does not own their schemas.

- **Training pipeline** — emits the checkpoint (`.pth`) and a sidecar JSON containing the train/val loss series and the best epoch. Benchmarking copies the relevant scalars into `runs.parquet`.
- **sentinel2 pipeline** — emits the dataset manifest, the mask variants (10m, 2.5m, skeletonised), and the ground-truth road graphs for APLS.

### Internal modules

| module | role |
|---|---|
| `model_loader.py` | Registry mapping `model_name -> (build_fn, predict_fn)`. Loads a checkpoint and returns a `Predictor` exposing a single `.predict(batch)` method. |
| `data_loader.py` | Iterates `(image, gt_mask, tile_id)` according to the manifest and the requested split and resolution. |
| `metrics.py` | Pure functions. `confusion_counts` and `derive_metrics` for the pixel metrics; `apls_metric` for the graph metric. No I/O, no state. |
| `runner.py` | Orchestrator. Holds no state of its own; threads the predictor, data loader, metrics, and store together for one `(model, seed)` evaluation. |
| `store.py` | Parquet I/O. Append rows, refuse duplicate `run_id`, read both tables back as pandas DataFrames. |
| `stats.py` | Post-hoc analysis on the joined table. `cross_seed_ci` for per-config training-instability CIs; `bootstrap_paired_diff` and `wilcoxon_paired` for between-model comparisons. |
| `cli.py` | Typer entrypoint matching the `sentinel2data/cli.py` style. Wires CLI flags into the runner. |

The split between `runner.py` (orchestration, has side effects) and `metrics.py` / `stats.py` (pure, no I/O) is the key invariant. It lets the statistical and metric code be tested in isolation against synthetic inputs while the runner is exercised separately with fixture data.

## File layout

```
benchmarks/
  runs.parquet
  tile_metrics.parquet
  predictions/
    {run_id}/
      {tile_id}.png         # binarised prediction mask (optional)
      {tile_id}.parquet     # extracted road graph (optional, for APLS reuse)
```

Both tables are append-only. Reruns produce new `run_id`s; they do not overwrite existing rows. A run that needs to be discarded is removed by `run_id` from both tables.

## `runs.parquet`

One row per evaluated checkpoint.

| column | type | description |
|---|---|---|
| `run_id` | string (UUID) | Primary key. Generated when the run starts. |
| `run_started_at` | timestamp (UTC) | Set at the start of inference. |
| `run_finished_at` | timestamp (UTC) | Set when all tiles complete. NaT if the run failed mid-way. |
| `model_name` | string | Architecture identifier, e.g. `unet_resnet50`, `terramind_v1_base`. Matches the key in the model registry. |
| `config_hash` | string | First 12 characters of the SHA-256 of the canonicalised training config dict. Two rows with the same `config_hash` came from identical configurations. |
| `config_yaml` | string | Full training config inline, serialised as YAML. Small (kilobytes), kept for reproducibility. |
| `seed` | int64 | Random seed used at training time. Several seeds per `config_hash` are expected and used to derive confidence intervals. |
| `loss_fn` | string | Loss function identifier, e.g. `focal_tversky`, `cldice_bce`, `dice`. Surfaced as its own column to make the loss-function pilot study trivial to query. |
| `loss_params` | string (JSON) | Parameters for the loss function (e.g. focal `alpha`, `gamma`, Tversky `beta`). JSON string so the structure can vary per loss. |
| `checkpoint_path` | string | Absolute path to the `.pth` file evaluated by this run. |
| `train_loss_final` | float64 | Training loss at the final epoch. |
| `val_loss_final` | float64 | Validation loss at the final epoch. |
| `val_loss_best` | float64 | Validation loss at the epoch the saved checkpoint was taken from. |
| `best_epoch` | int64 | Epoch index (0-based) at which `val_loss_best` was recorded. |
| `dataset_split` | string | Which split was evaluated: `train`, `val`, or `test`. Almost always `test`. |
| `resolution` | string | Ground-truth resolution at which pixel metrics were computed: `10m` or `2.5m`. One resolution per run; evaluating the same checkpoint at a second resolution produces a new `run_id`. APLS is always computed against the skeletonised 10m mask regardless of this field. |
| `threshold` | float64 | Sigmoid threshold used to binarise predictions. |
| `n_tiles` | int64 | Number of tiles in the run. Should equal `len(tile_metrics[tile_metrics.run_id == run_id])`. |

## `tile_metrics.parquet`

One row per `(run_id, tile_id)`. Long-form: every tile is its own row regardless of how many models are evaluated.

| column | type | description |
|---|---|---|
| `run_id` | string | Foreign key into `runs.parquet`. |
| `tile_id` | string | Tile identifier (filename stem, matching the entries in `data_split.json`). |
| `tp` | int64 | True positive pixel count. |
| `fp` | int64 | False positive pixel count. |
| `fn` | int64 | False negative pixel count. |
| `tn` | int64 | True negative pixel count. |
| `iou` | float64 | Intersection over union, computed from the counts in this row. |
| `f1` | float64 | F1 / Dice coefficient. |
| `precision` | float64 | Precision. |
| `recall` | float64 | Recall. |
| `accuracy` | float64 | Pixel accuracy. |
| `apls` | float64 | Average Path Length Similarity, computed on the predicted road graph against the ground-truth graph. **Nullable** — NaN when the graph metric was not computed (cheap pixel-only pass, or graph extraction failed). |
| `n_pred_nodes` | int64 | Number of nodes in the predicted graph. Helpful for diagnosing low or NaN APLS values. Nullable. |
| `n_pred_edges` | int64 | Number of edges in the predicted graph. Nullable. |
| `inference_ms` | float64 | Wall-clock time to run the model forward pass for this tile, in milliseconds. Excludes data loading and metric computation. |

The raw counts (`tp`, `fp`, `fn`, `tn`) are kept alongside the derived metrics deliberately. Any new pixel metric a downstream analysis wants — Matthews correlation, Cohen's kappa, balanced accuracy — can be recomputed from the counts without rerunning inference.

## Joining the tables

The canonical join:

```python
import pandas as pd

runs = pd.read_parquet("benchmarks/runs.parquet")
tiles = pd.read_parquet("benchmarks/tile_metrics.parquet")
df = runs.merge(tiles, on="run_id")
```

Once joined, every per-tile row carries its full run context (`model_name`, `seed`, `loss_fn`, etc.) and is ready for arbitrary slicing.

## Common queries

### Confidence interval across seeds for one config

For a given `(model_name, config_hash)`, take the dataset-level metric per seed and report a CI across seeds.

```python
# dataset-level IoU per seed (micro average from counts)
seed_iou = (
    df.groupby(["model_name", "config_hash", "seed"])
      .apply(lambda g: g["tp"].sum() / (g["tp"].sum() + g["fp"].sum() + g["fn"].sum()))
      .rename("iou")
      .reset_index()
)

# 95% CI across seeds, per config
seed_iou.groupby(["model_name", "config_hash"])["iou"].agg(
    mean="mean",
    lo=lambda x: x.quantile(0.025),
    hi=lambda x: x.quantile(0.975),
)
```

### Bootstrap CI on per-tile metrics

Sample `n_tiles` tiles with replacement, recompute the aggregate, repeat.

```python
import numpy as np

def bootstrap_iou(tile_df, n_boot=1000, sample_size=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    boot = np.empty(n_boot)
    arr = tile_df[["tp", "fp", "fn"]].to_numpy()
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), size=sample_size)
        tp, fp, fn = arr[idx].sum(axis=0)
        boot[i] = tp / (tp + fp + fn)
    return np.quantile(boot, [0.025, 0.5, 0.975])
```

### Wilcoxon paired comparison between two models

Wilcoxon needs paired observations on the same tile. Pivot to wide form first.

```python
from scipy.stats import wilcoxon

# pick one seed per model, or average within model first
paired = (
    df.query("model_name in ['unet_resnet50', 'terramind_v1_base'] and seed == 42")
      .pivot(index="tile_id", columns="model_name", values="iou")
      .dropna()
)
stat, p = wilcoxon(paired["unet_resnet50"], paired["terramind_v1_base"])
```

### Loss-function pilot study

Compare loss functions on shared tiles, holding the model fixed.

```python
(
    df.query("model_name == 'unet_resnet50'")
      .groupby("loss_fn")["iou"]
      .agg(["mean", "std", "count"])
)
```

A Friedman test (multi-group analogue of Wilcoxon) is appropriate when comparing more than two loss functions on the same tiles.

## Design notes

### One resolution per run

Pixel metrics for a single run are computed against a single ground-truth resolution. To evaluate the same checkpoint against 2.5m and 10m ground truth, run benchmarking twice and the two results land in separate rows of `runs.parquet` with different `run_id`s. This keeps `tile_metrics` columns flat (no `iou_10m` / `iou_2_5m` split) and makes comparison across resolutions a normal groupby on `resolution`.

The graph metric (`apls`) is always computed against the skeletonised 10m mask, independent of the `resolution` field.

### Mask paths live in the dataset, not here

`tile_metrics` does not store paths to the ground-truth masks. Those are dataset state and are expected to be resolved through a separate dataset manifest (produced by the sentinel2 pipeline alongside the masks themselves). A run records `dataset_split` and `resolution`; the runner uses those plus the manifest to find the right files. This avoids duplicating dataset metadata across many benchmark rows.

### Why parquet, not CSV

Parquet preserves int64/float64/datetime/nullable types, compresses well, and reads back into pandas in a single call without dtype hints. The downstream workload — bootstrap resampling, Wilcoxon and Friedman tests, joins between the two tables — is sensitive to dtype and NaN handling in ways that CSV makes painful. The benchmarking module hides the format behind a thin facade (`store.read` / `store.write`), so switching formats later remains a small change.

### Why long-form, not wide

Adding a new model, a new seed, or a new loss-function variant requires no schema migration; it produces new rows under the existing columns. Wide-form schemas (one column per model's IoU) couple the schema to the experiment matrix and break this property.

### Append-only, never overwrite

The runner refuses to write a `run_id` that already exists. Reruns are explicit new runs. This makes results auditable: every checkpoint evaluation is preserved, including the failed ones.

### `config_hash` semantics

`config_hash` is the SHA-256 (first 12 characters) of the training config dict after canonicalisation: keys sorted recursively, floats round-tripped through string form, augmentation pipelines serialised via `albumentations.to_dict`. Two runs with the same hash were produced by identical configurations and are directly comparable. The full `config_yaml` is kept inline for inspection without needing to look the run up elsewhere.
