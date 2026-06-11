# Benchmarking

Reference for the benchmarking module: what is implemented, how the modules fit together, the on-disk schema the full system will write, and how to use the statistical analysis surface.

## Status

The pixel-metrics and statistical analysis layers are implemented and tested. The orchestration layer (runner, store, model loader, data loader, CLI) is planned but not yet implemented.

| layer | module | status |
| ----- | ------ | ------ |
| pixel metrics | `confusion_matrix.py` | implemented |
| statistical analysis | `stats.py` | implemented |
| demo harness | `dummy_pipeline.py` | implemented |
| model loader | `model_loader.py` | planned |
| data loader | `data_loader.py` | planned |
| runner | `runner.py` | planned |
| parquet store | `store.py` | planned |
| CLI | `cli.py` | planned |

## Module overview

### `confusion_matrix.py`

Pure functions. No I/O, no state. Import directly or through `stats` (which re-exports everything).

`confusion_counts(output, target, threshold, from_logits, ignore_index) -> ConfusionCounts`
: Computes per-chip `(tp, fp, fn, tn)` for binary road segmentation. Positive class is road (`target == 1`). Pixels equal to `ignore_index` are excluded from all four counts. Each field of `ConfusionCounts` is a 1-D `LongTensor` of length B, one entry per chip in the batch. Accepts `[B,1,H,W]`, `[B,H,W]`, or `[H,W]` input.

`pixel_metrics_from_counts(c: ConfusionCounts) -> dict`
: Derives `iou`, `f1`, `precision`, `recall` from the counts. A metric with a zero denominator is returned as `NaN`, not 0 or 1. `NaN` is the honest "undefined here" value; downstream bootstrap and Wilcoxon already drop `NaN` pairs, so undefined tiles are excluded from comparisons rather than biasing them.

### `stats.py`

The combined analysis surface. Re-exports `ConfusionCounts`, `confusion_counts`, and `pixel_metrics_from_counts` from `confusion_matrix`, so this is the only import a caller needs.

**`bootstrap_paired_diff(df, model_a, model_b, metric, n_boot, rng, confidence)`**

Paired-bootstrap confidence interval on the per-tile metric difference `model_a - model_b`.

Each bootstrap iteration resamples *pairs* with replacement, never the two models independently. A constant per-tile offset therefore collapses the CI to a point regardless of how the underlying values vary across tiles. `diff_mean` is the observed (not bootstrapped) mean difference. Returns `{"diff_mean", "ci_lo", "ci_hi", "n_pairs"}`.

Input: long-form DataFrame with columns `(model_name, tile_id, <metric>)`, exactly one row per `(model_name, tile_id)`. Multi-seed data must be pre-aggregated before passing in.

**`wilcoxon_paired(df, model_a, model_b, metric)`**

Wilcoxon signed-rank test on the paired per-tile difference. Tests whether `model_a - model_b` is symmetric about zero. Returns `{"statistic", "p_value", "n_pairs"}`. Same pairing and NaN-dropping logic as `bootstrap_paired_diff`.

**`cross_seed_ci(df, config_filters, metric, aggregation, confidence)`**

t-interval of a dataset-level metric across seeds for one fixed configuration, quantifying training instability.

`config_filters` is a dict of `{column: value}` pairs that together identify one configuration, e.g. `{"model_name": "unet_resnet50", "loss_fn": "focal"}`. Each seed's per-tile rows are reduced to one scalar:

- `aggregation="macro"`: mean of per-tile metric values.
- `aggregation="micro"`: pool `tp/fp/fn/tn` across the seed's tiles, then derive the metric. Only valid for `iou`, `f1`, `precision`, `recall`, `accuracy`.

With a single seed, `std` and the CI bounds are `NaN` (undefined, not zero). Returns `{"mean", "std", "ci_lo", "ci_hi", "n_seeds", "per_seed_values"}`.

Input: same long-form DataFrame, plus a `seed` column (and count columns for micro).

### `dummy_pipeline.py`

End-to-end demo. Runs out of the box with no real model and no real image data. Shows exactly how the modules connect so you can verify the plumbing before swapping in a real checkpoint.

**Inference unit vs. evaluation unit.** These are deliberately different:

- *Inference* is always whole-tile. `tiled_inference` slides an overlapping window and blends the results into one seamless prediction, which avoids edge artifacts at chip boundaries.
- *Evaluation units* are non-overlapping chips. `score_in_chips()` dices the seamless stitched prediction and the mask into a grid and scores each chip independently, producing one row per chip. These per-chip rows are what the stats module resamples and pairs over.

The CapeTown mask (2553 x 2563) yields 144 chips of 224 px, which is the effective sample size for bootstrap and Wilcoxon. Adjacent chips from one image are spatially correlated, so they are not as independent as chips from separate satellite acquisitions. With a real multi-tile dataset you would resample over tiles; chips are the pragmatic proxy when you have one image.

Each chip row carries a `tile_id` of the form `{tile}_r{row}_c{col}`, mirroring the `(patch_row_id, patch_col_id)` grid in `DatasetManager`. Every row is stamped with `model_name`, `seed`, and `run_id` by `tag_run()`.

Two inference modes, controlled by `CONFIG.INFERENCE_MODE`:

**chip mode** (`"chip"`): feeds a batch of pre-sized chips through a `Predictor`, scores each chip with `confusion_counts` and `pixel_metrics_from_counts`, and returns a per-chip DataFrame. Data is synthetic by default; set `USE_SYNTHETIC_DATA = False` and implement the `ChipDataset` stubs to wire in real chips.

**tile mode** (`"tile"`): loads the real binary road mask from `TILE_MASK_PATH`, generates a synthetic image of matching spatial size, and runs `terratorch.tasks.tiled_inference.tiled_inference` (sliding-window with overlap blending) over the whole tile. `score_in_chips()` then dices the stitched prediction and mask into the chip grid. Defaults to `dummy_data/CapeTown_mask.tif` (2553 x 2563, ~21% road pixels) so the tile path runs immediately.

The `Predictor` class wraps any `nn.Module` that returns a `ModelOutput`-like object with a `.output` attribute (the TerraTorch convention). It handles both 1-channel (sigmoid) and N-channel (softmax, `POSITIVE_CLASS_INDEX`) output conventions. Swap the model inside; the benchmarking code never changes.

**Running multiple evaluations.** `model_name` and `seed` are set by environment variables so you can append several runs without editing the file:

```sh
BENCH_MODEL=dummy_a BENCH_SEED=0 python dummy_pipeline.py
BENCH_MODEL=dummy_b BENCH_SEED=0 python dummy_pipeline.py
BENCH_MODEL=dummy_a BENCH_SEED=1 python dummy_pipeline.py
```

With `APPEND_PARQUET = True` (the default), each run reads the existing parquet, concatenates, and rewrites it. Parquet files are immutable — there is no in-place append; read-concat-write is the correct pattern at this scale.

Each model at the same seed gets a distinct RNG state via `_effective_seed()`, which folds `model_name` into the seed. Without this, two different model names at seed 0 would produce byte-identical predictions, which makes paired differences all zero and breaks the bootstrap/Wilcoxon demo.

Sample parquet after three runs (dummy_a seed 0, dummy_b seed 0, dummy_a seed 1 — 432 rows):

```
  tile_id          patch_row_id  patch_col_id  tp   fp    fn   tn   iou    f1  precision  recall  model_name  seed  run_id
  CapeTown_r0_c0   0             0             ...  ...   ...  ...  0.14   0.24  0.20    0.37    dummy_a     0     a3d75c3b...
  ...
```

Precision (~0.20) matches the road base rate in the mask, which is the expected behaviour of a random predictor. `tp + fn` for the full tile sums to 1,344,724, the exact road-pixel count in the mask.

## System architecture

The full planned system:

```
                   Training (external)
                          |
                          | checkpoint + sidecar (train/val losses)
                          v
    CLI / YAML  -->  ModelLoader
    (config)         (registry)
                          |
                          | Predictor.predict
                          v
    sentinel2       Runner
    manifest   -->    per tile:
    masks (ext)        1. DataLoader yields (img, gt_mask)
                       2. logits = tiled_inference(model, img)   whole-tile, blended
                       3. probs  = logits_to_road_prob(logits)
                       4. per chip in score_in_chips(probs, mask):
                              counts = confusion_counts          } confusion_matrix.py
                              pixel  = pixel_metrics_from_counts
                       5. apls   = apls_metric (on full tile)
                       6. append chip rows to tile_metrics
                     finalise runs row
                          |
                          v
                       Store
                    (parquet I/O)
                          |
                          v
                  runs.parquet
                  tile_metrics.parquet   one row per (run_id, chip tile_id)
                          |
                          v
                       stats.py
                  cross_seed_ci           -> CIs for training instability
                  bootstrap_paired_diff   -> CI on model difference
                  wilcoxon_paired         -> p-value on model difference
```

The split between Runner (orchestration, side effects) and `confusion_matrix.py` / `stats.py` (pure, no I/O) is the key invariant. The metric and stats code can be tested in isolation against synthetic DataFrames; the runner is exercised separately with real fixture data.

## On-disk schema

### `runs.parquet`

One row per evaluated checkpoint.

| column | type | description |
| ------ | ---- | ----------- |
| `run_id` | string (UUID) | Primary key. Generated when the run starts. |
| `run_started_at` | timestamp (UTC) | Set at the start of inference. |
| `run_finished_at` | timestamp (UTC) | Set when all tiles complete. NaT if the run failed mid-way. |
| `model_name` | string | Architecture identifier, e.g. `unet_resnet50`, `terramind_v1_base`. Matches the key in the model registry. |
| `config_hash` | string | First 12 characters of the SHA-256 of the canonicalised training config dict. Two rows with the same `config_hash` came from identical configurations. |
| `config_yaml` | string | Full training config inline, serialised as YAML. Kept for reproducibility. |
| `seed` | int64 | Random seed used at training time. Several seeds per `config_hash` are expected and used to derive confidence intervals. |
| `loss_fn` | string | Loss function identifier, e.g. `focal_tversky`, `cldice_bce`, `dice`. Surfaced as its own column to make the loss-function pilot study trivial to query. |
| `loss_params` | string (JSON) | Parameters for the loss function. JSON string so the structure can vary per loss. |
| `checkpoint_path` | string | Absolute path to the `.pth` file evaluated by this run. |
| `train_loss_final` | float64 | Training loss at the final epoch. |
| `val_loss_final` | float64 | Validation loss at the final epoch. |
| `val_loss_best` | float64 | Validation loss at the epoch the saved checkpoint was taken from. |
| `best_epoch` | int64 | Epoch index (0-based) at which `val_loss_best` was recorded. |
| `dataset_split` | string | Which split was evaluated: `train`, `val`, or `test`. Almost always `test`. |
| `resolution` | string | Ground-truth resolution at which pixel metrics were computed: `10m` or `2.5m`. One resolution per run; evaluating the same checkpoint at a second resolution produces a new `run_id`. APLS is always computed against the skeletonised 10m mask regardless of this field. |
| `threshold` | float64 | Sigmoid threshold used to binarise predictions. |
| `n_tiles` | int64 | Number of tiles in the run. |

### `tile_metrics.parquet`

One row per `(run_id, tile_id)`. Long-form: every chip is its own row. `tile_id` identifies the chip, not the full satellite image — it is a string of the form `{image_stem}_r{patch_row_id}_c{patch_col_id}`, mirroring the `(patch_row_id, patch_col_id)` grid in `DatasetManager`.

| column | type | description |
| ------ | ---- | ----------- |
| `run_id` | string | Foreign key into `runs.parquet`. UUID generated at the start of each run. |
| `tile_id` | string | Chip identifier. Format `{image_stem}_r{row}_c{col}`, one per non-overlapping chip in the scoring grid. |
| `patch_row_id` | int64 | Row index of this chip in the scoring grid (0-based). |
| `patch_col_id` | int64 | Column index of this chip in the scoring grid (0-based). |
| `model_name` | string | Architecture identifier. Denormalised onto every chip row so the parquet is self-contained for stats queries without needing to join `runs.parquet`. |
| `seed` | int64 | Training seed. Denormalised for the same reason. |
| `tp` | int64 | True positive pixel count. |
| `fp` | int64 | False positive pixel count. |
| `fn` | int64 | False negative pixel count. |
| `tn` | int64 | True negative pixel count. |
| `iou` | float64 | Intersection over union. |
| `f1` | float64 | F1 / Dice coefficient. |
| `precision` | float64 | Precision. |
| `recall` | float64 | Recall. |
| `accuracy` | float64 | Pixel accuracy. |
| `apls` | float64 | Average Path Length Similarity. Nullable: NaN when graph metric was not computed. |
| `n_pred_nodes` | int64 | Nodes in the predicted graph. Nullable. |
| `n_pred_edges` | int64 | Edges in the predicted graph. Nullable. |
| `inference_ms` | float64 | Wall-clock time for the model forward pass, in milliseconds. Excludes data loading and metric computation. |

The raw counts (`tp`, `fp`, `fn`, `tn`) are kept alongside the derived metrics. Any new pixel metric (Matthews correlation, Cohen's kappa, balanced accuracy) can be recomputed from the counts without rerunning inference.

## Joining the tables

```python
import pandas as pd

runs = pd.read_parquet("benchmarks/runs.parquet")
tiles = pd.read_parquet("benchmarks/tile_metrics.parquet")
df = runs.merge(tiles, on="run_id")
```

Once joined, every per-tile row carries its full run context (`model_name`, `seed`, `loss_fn`, etc.) and is ready for arbitrary slicing.

## Using the stats module

The parquet written by `dummy_pipeline.py` is already in the right shape. Load it and you can call all three functions immediately:

```python
import numpy as np, pandas as pd
from benchmarking.stats import bootstrap_paired_diff, wilcoxon_paired, cross_seed_ci

df = pd.read_parquet("tile_metrics_dummy.parquet")
```

### Paired bootstrap CI between two models

`bootstrap_paired_diff` requires exactly one row per `(model_name, tile_id)`. If the parquet contains multiple seeds, filter to one seed first — otherwise the function raises on the duplicate `(model_name, tile_id)` pairs.

```python
seed0 = df[df["seed"] == 0]

out = bootstrap_paired_diff(
    seed0,
    model_a="dummy_a",
    model_b="dummy_b",
    metric="iou",
    n_boot=2000,
    rng=np.random.default_rng(42),
)
print(f"diff {out['diff_mean']:+.4f}  95% CI [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]  n={out['n_pairs']}")
```

### Wilcoxon signed-rank test

Same seed-filter requirement as bootstrap:

```python
out = wilcoxon_paired(seed0, "dummy_a", "dummy_b", metric="iou")
print(f"W={out['statistic']:.1f}  p={out['p_value']:.4f}  n={out['n_pairs']}")
```

### Cross-seed confidence interval (training instability)

`cross_seed_ci` is the complement: it *wants* multiple seeds for the same model. It aggregates each seed's chips to a scalar, then reports a t-interval across seeds.

```python
out = cross_seed_ci(
    df,
    config_filters={"model_name": "dummy_a"},
    metric="iou",
    aggregation="micro",   # pool tp/fp/fn/tn across chips, then derive IoU per seed
    confidence=0.95,
)
print(f"mean {out['mean']:.4f}  std {out['std']:.4f}  "
      f"95% CI [{out['ci_lo']:.4f}, {out['ci_hi']:.4f}]  n_seeds={out['n_seeds']}")
```

Use `aggregation="macro"` to average the per-chip metric values instead of pooling the counts. Macro and micro diverge when chips are imbalanced in road pixel count; micro weights by area, macro weights by chip.

### Loss-function pilot study

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

Pixel metrics for a single run are computed against a single ground-truth resolution. To evaluate the same checkpoint against 2.5m and 10m ground truth, run benchmarking twice; the two results land in separate rows of `runs.parquet` with different `run_ids`. This keeps `tile_metrics` columns flat and makes comparison across resolutions a normal groupby on `resolution`.

APLS is always computed against the skeletonised 10m mask, independent of the `resolution` field.

### Why parquet, not CSV

Parquet preserves int64/float64/datetime/nullable types, compresses well, and reads back into pandas in a single call without dtype hints. The downstream workload (bootstrap resampling, Wilcoxon and Friedman tests, joins between the two tables) is sensitive to dtype and NaN handling in ways that CSV makes painful.

### Why long-form, not wide

Adding a new model, a new seed, or a new loss-function variant requires no schema migration; it produces new rows under the existing columns. Wide-form schemas (one column per model's IoU) couple the schema to the experiment matrix and break this property.

### Append-only, never overwrite

The runner will refuse to write a `run_id` that already exists. Reruns are explicit new runs with new IDs. This makes results auditable: every checkpoint evaluation is preserved, including failed ones.

### `config_hash` semantics

`config_hash` is the SHA-256 (first 12 characters) of the training config dict after canonicalisation: keys sorted recursively, floats round-tripped through string form, augmentation pipelines serialised via `albumentations.to_dict`. Two runs with the same hash were produced by identical configurations and are directly comparable. The full `config_yaml` is kept inline for inspection.

### NaN semantics for pixel metrics

A metric with a zero denominator (e.g. a chip with no road in either prediction or ground truth) is stored as `NaN`, not 0 or 1. `NaN` is the honest "undefined here" value. Both `bootstrap_paired_diff` and `wilcoxon_paired` drop `NaN` pairs before analysis, so undefined tiles are excluded from comparisons rather than biasing them. The `n_pairs` field in the return value reflects how many pairs survived.
