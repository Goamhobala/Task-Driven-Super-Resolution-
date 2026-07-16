# Benchmarking

Reference for the benchmarking module: what is implemented, how the modules fit together, the on-disk schema the full system will write, and how to use the statistical analysis surface.

## Status

All layers are implemented and tested (`tests/test_stats.py`, `tests/test_runner_integration.py` — the latter runs real random-init checkpoints through `evaluate()` end to end). The data loader is implemented externally and is out of scope for this module; the runner reuses its reading/normalisation/mask helpers so eval cannot drift from training.

| layer | module | status |
| ----- | ------ | ------ |
| pixel metrics | `confusion_matrix.py` | implemented |
| statistical analysis | `stats.py` | implemented |
| demo harness | `dummy_pipeline.py` | implemented |
| demo analysis | `example.py` | implemented |
| runner (unet + sr families) | `runner.py` | implemented |
| sharded parquet store | `store.py` | implemented |
| CLI (eval/compare/variance/report) | `cli.py` | implemented |
| tile-metric plugin seam | `tile_metrics.py` | implemented |
| APLS (connectivity) | `graph_metrics.py` | implemented (`--tile-metric apls`, on by default in the HPC bench stages) |
| HPC integration (STAGE=bench) | `scripts/hpc/*/_stages.sh` | implemented |
| data loader | external | out of scope |

## Module overview

### `confusion_matrix.py`

Pure functions. No I/O, no state. Import directly or through `stats`, which re-exports everything.

`confusion_counts(output, target, threshold, from_logits, ignore_index) -> ConfusionCounts`
: Computes per-chip `(tp, fp, fn, tn)` for binary road segmentation. Positive class is road (`target == 1`). Pixels equal to `ignore_index` are excluded from all four counts. Each field of `ConfusionCounts` is a 1-D `LongTensor` of length B, one entry per chip in the batch. Accepts `[B,1,H,W]`, `[B,H,W]`, or `[H,W]` input.

`pixel_metrics_from_counts(c: ConfusionCounts) -> dict`
: Derives `iou`, `f1`, `precision`, `recall` from the counts. A metric with a zero denominator is returned as `NaN`, not 0 or 1. `NaN` is the honest "undefined here" value; downstream bootstrap and Wilcoxon drop `NaN` pairs, so undefined chips are excluded from comparisons rather than biasing them.

### `stats.py`

The combined analysis surface. Re-exports `ConfusionCounts`, `confusion_counts`, and `pixel_metrics_from_counts` from `confusion_matrix`, so this is the only import a caller needs.

**`bootstrap_paired_diff(df, model_a, model_b, metric, n_boot, rng, confidence)`**

Paired-bootstrap confidence interval on the per-chip metric difference `model_a - model_b`.

Each bootstrap iteration resamples *pairs* with replacement, never the two models independently. A constant per-chip offset therefore collapses the CI to a point regardless of how the underlying values vary across chips. `diff_mean` is the observed (not bootstrapped) mean difference. Returns `{"diff_mean", "ci_lo", "ci_hi", "n_pairs"}`.

Input: long-form DataFrame with columns `(model_name, chip_id, <metric>)`, exactly one row per `(model_name, chip_id)`. Multi-seed data must be pre-aggregated to one value per chip before passing in.

**`wilcoxon_paired(df, model_a, model_b, metric)`**

Wilcoxon signed-rank test on the paired per-chip difference. Tests whether `model_a - model_b` is symmetric about zero. Returns `{"statistic", "p_value", "n_pairs"}`. Same pairing and NaN-dropping logic as `bootstrap_paired_diff`.

**`cross_seed_ci(df, config_filters, metric, aggregation, confidence)`**

t-interval of a dataset-level metric across seeds for one fixed configuration, quantifying training instability.

`config_filters` is a dict of `{column: value}` pairs that together identify one configuration, e.g. `{"model_name": "unet_resnet50", "loss_fn": "focal"}`. Each seed's per-chip rows are reduced to one scalar:

- `aggregation="macro"`: mean of per-chip metric values.
- `aggregation="micro"`: pool `tp/fp/fn/tn` across the seed's chips, then derive the metric. Only valid for `iou`, `f1`, `precision`, `recall`, `accuracy`.

With a single seed, `std` and the CI bounds are `NaN` (undefined, not zero). Returns `{"mean", "std", "ci_lo", "ci_hi", "n_seeds", "per_seed_values"}`.

Input: same long-form DataFrame, plus a `seed` column (and count columns for micro).

### `runner.py`

`evaluate(dataset_dir, checkpoint, model_name, seed, store_dir, ...)` scores one checkpoint over a split's **footprint chips** and writes the sharded store. Two model families are supported through a predictor registry:

- `model="unet"` — `UNetLightning`; image + mask read at the same native (10 m) window; normalisation replays the checkpoint's frozen train stats from `hparams`.
- `model="sr"` — `JointSRUNetLightning`; the native window is fed as raw DN (the forward normalises internally, after super-resolution) and scored at 2.5 m against HR ground truth. `mask_source="graph"` rasterises the tile's `masks_graph` parquet, `"raster"` reads `<split>/<mask_dirname>/` COGs — the training dataloader's own helpers. SEN2SR's pinned 128 px LR input is handled by running each cell as a grid of pinned sub-windows and stitching the HR outputs; bicubic/sr4rs predict a cell in one pass. `--sen2sr-dir` overrides the weights path baked into `hparams` at training time.

**Footprint grid.** The evaluation chip is a ground cell of `cell_m` metres (default 2560 m = 256 px @ 10 m = 1024 px @ 2.5 m), derived per tile from the raster transform. `chip_id` (`{tile_stem}_r{ri}_c{ci}`) therefore names the same geography at every resolution — this is what lets the paired stats line up chips across model families.

**Edge chips.** Ragged border cells are padded up to a valid model input (next /32 for unet — the smp decoder constraint; the full zero-padded cell for sr, matching the SR training loader) and the logits are cropped back to the true extent *before* scoring, so counts never include pad pixels.

**Invariants.** The grid must partition each tile exactly (always checked), and the chips' summed `tp+fn` must equal the tile mask's road-pixel count, read independently of the chip loop (`check=first|all|off`, default `first`). A windowing/GT bug fails the run loudly instead of producing plausible wrong numbers.

**Timing.** `inference_ms` is `cuda.synchronize()`-bracketed and amortised over the batch (`batch_size` chips per forward pass for unet).

### `store.py`

One shard per run — `runs/<run_id>.parquet`, `chips/<run_id>.parquet`, `tiles/<run_id>.parquet` — written tmp-then-`os.replace` (atomic on one filesystem, Lustre included). Concurrent SLURM jobs sharing a `STORE_DIR` cannot lose each other's rows, and append-only comes free: an existing shard for a `run_id` is an error. Loaders (`load_runs` / `load_chips` / `load_tiles` / `load_joined`) glob the shards and still read the legacy flat files (`runs.parquet`, `chip_metrics.parquet`) for back-compat.

### `tile_metrics.py`

The seam graph metrics drop into. A plugin registered with `@register("name")` receives, per tile, the stitched binary prediction + GT at GT resolution, the geotransform, and the footprint grid, and returns tile-level values (-> `tiles/` shard) plus optional per-chip values (merged onto chip rows as nullable columns). Request plugins with repeatable `--tile-metric name`. `road_frac` is the trivial reference plugin.

### `graph_metrics.py` — APLS

`--tile-metric apls` scores Average Path Length Similarity (Van Etten et al. 2019): both masks are skeletonized and chain-traced into road graphs (pure numpy/networkx — no sknw/numba), control points are injected every 500 m along edges, snapped across graphs within 30 m, and shortest-path lengths compared in ground metres; the score is the harmonic mean of the two directions (CosmiQ convention). Emitted at BOTH levels, `road_frac`-style: **per-chip `apls`** merges onto the chip rows so the paired bootstrap/Wilcoxon resample the same `chip_id` unit as the pixel metrics (chips resolve first in the CLI; each 2560 m chip is scored independently — SpaceNet precedent used 400 m cells), and a **tile-level `apls`** row (plus the two directional scores and graph sizes) in `tiles/`. Chip APLS measures within-chip connectivity only — paths crossing a chip border are never sampled — so the tile row is the longer-range routing measure and the convenient rollup. NaN where a unit's masks are both road-free, 0.0 when exactly one is empty — the stats drop NaN pairs. Deterministic (sorted nodes, seeded subsampling). Defaults are GSD-aware via the transform; see the module docstring to retune (`SNAP_DIST_M`, `CONTROL_DELTA_M`, `MIN_SPUR_M`, `MAX_NODES`). Skeletonization noise is systematic — it hits every model and the GT identically — so between-model comparisons stay fair. Unit tests: `src/benchmarking/tests/test_graph_metrics.py` (torch-free).

### `cli.py`

`eval` / `compare` / `variance` / `report`. Metrics resolve against the chips table first, then the tiles table (tile metrics pair on `tile_id`). `compare` and `report` check the runs metadata and **refuse cross-GT pixel comparisons** (different `gt_res_m`, `label_source`, `mask_source`, `dataset_split`, or `cell_m`) unless `--force` — pixel metrics are only comparable within one ground truth; graph metrics are the cross-GT route. `report` takes repeatable `--metric` and exports markdown or CSV via `--out`.

### `dummy_pipeline.py`

End-to-end demo. Runs out of the box with no real model and no real image data. Shows exactly how the modules connect so you can verify the plumbing before swapping in a real checkpoint.

**Inference unit vs. evaluation unit.** These are deliberately different:

- Inference is always whole-tile. `tiled_inference` slides an overlapping window and blends the results into one seamless prediction, which avoids edge artifacts at chip boundaries.
- Evaluation units are non-overlapping chips. `score_in_chips()` dices the seamless stitched prediction and the mask into a grid and scores each chip independently, producing one row per chip. These per-chip rows are what the stats module resamples and pairs over.

The CapeTown mask (2553 x 2563) yields 144 chips of 224 px, which is the effective sample size for bootstrap and Wilcoxon. Adjacent chips from one image are spatially correlated, so they are not as independent as chips from separate satellite acquisitions. With a real multi-tile dataset you would resample over `tile_id` instead; chips are the pragmatic proxy when you have one image.

**Two keys per chip row:**

- `chip_id` is the evaluation unit and a foreign key into the dataset's per-patch metadata catalogue (`DatasetManager`'s `metadata.parquet`). Format: `{image_stem}_r{patch_row_id}_c{patch_col_id}`. Bootstrap and Wilcoxon pair on this key.
- `tile_id` is the parent image stem, denormalised onto every row so chips can be rolled up to tiles without joining the catalogue.

Every row is also stamped with `model_name`, `seed`, and `run_id` by `tag_run()`.

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

With `APPEND_PARQUET = True` (the default), each run reads the existing parquet, concatenates, and rewrites it. Parquet files are immutable and have no in-place append; read-concat-write is the correct pattern at this scale.

Each model at the same seed gets a distinct RNG state via `_effective_seed()`, which folds `model_name` into the seed. Without this, two different model names at seed 0 would produce byte-identical predictions and paired differences would all be zero.

Sample parquet after six runs (dummy_a and dummy_b, each at seeds 0, 1, 2; 864 rows total):

```
  chip_id         tile_id   patch_row_id  patch_col_id  tp   fp    fn   tn   iou    f1  ...  model_name  seed  run_id
  CapeTown_r0_c0  CapeTown  0             0             ...  ...   ...  ...  0.14  0.24  ...  dummy_a     0     a3d75c3b...
  ...
```

Precision is approximately 0.20, matching the road base rate in the mask, which is the expected behaviour of a random predictor. `tp + fn` for the full tile sums to 1,344,724, the exact road-pixel count in the mask.

### `example.py`

Worked analysis on the output of `dummy_pipeline.py`. Follows three steps:

1. Collapse multi-seed runs to one value per `(model_name, chip_id)` by averaging the per-chip metric across seeds. This is the pre-aggregation `bootstrap_paired_diff` and `wilcoxon_paired` require.
2. Paired bootstrap CI and Wilcoxon test on the seed-averaged per-chip values.
3. Per-chip mean and std across seeds to assess training stability. A large per-chip std means that chip's score swings when only the seed changes.

The example also prints the bootstrap unit count explicitly so the granularity is self-evident:

```
pairing on 144 chips across 1 tile(s) (chip is the bootstrap unit, tile_id is the parent image)
```

## System architecture

```
             Training (HPC: STAGE=tune -> STAGE=fit)
                          |
                          | checkpoint (+ best_params.yaml)
                          v
    CLI / YAML  -->    Runner  (STAGE=bench / benchmarking.cli eval)
    (config)             per tile:
                           1. footprint grid from the raster transform (cell_m)
                           2. per cell: predictor.predict -> sigmoid probs
                              (unet: native window; sr: raw-DN window -> 4x HR)
                           3. GT via the training loaders' own mask helpers
                           4. per cell:
                                  counts = confusion_counts          } confusion_matrix.py
                                  pixel  = pixel_metrics_from_counts
                           5. tile plugins (e.g. apls) on the stitched
                              prediction + GT -> tile rows / chip columns
                           6. tp+fn-vs-mask invariant
                         finalise runs row
                          |
                          v
                       Store  (sharded parquet, concurrency-safe)
                  runs/<run_id>.parquet
                  chips/<run_id>.parquet   one row per (run_id, chip_id)
                  tiles/<run_id>.parquet   one row per (run_id, tile_id)
                          |
                          v
                    stats.py  (via cli compare / variance / report)
                  cross_seed_ci           -> CIs for training instability
                  bootstrap_paired_diff   -> CI on model difference
                  wilcoxon_paired         -> p-value on model difference
```

The split between Runner (orchestration, side effects) and `confusion_matrix.py` / `stats.py` / `tile_metrics.py` plugins (pure, no I/O) is the key invariant. The metric and stats code test in isolation against synthetic DataFrames; the runner is exercised by `tests/test_runner_integration.py` with real (random-init) checkpoints over real GeoTIFF fixtures.

## Running on the HPC

Every experiment engine (`scripts/hpc/unet/_stages.sh`, `scripts/hpc/sr/_stages.sh`) has a `bench` stage beside `tune`/`fit`. It locates the experiment's fitted checkpoint and evaluates it into the **shared** store (`STORE_DIR`, default `/scratch/$USER/InstaRoad/benchmarks`) with `model_name = {family}_{EXP_TAG}` (e.g. `unet_cdngi`, `sr_r2a_cdngi`) — the name the stats pair and group on — plus `--exp-tag` / `--label-source` metadata for the comparability guardrail. The SR engine passes its `LABELS -> mask_source` mapping and `--sen2sr-dir`, so bench GT always matches training GT.

```sh
# score an already-trained checkpoint (no retraining)
sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=unet/cdngi.sh STAGE=bench SEED=0

# full pipeline in one allocation: tune -> fit -> bench
sbatch scripts/hpc/train_both.sbatch --SCRIPT=sr/r2a_cdngi.sh SEED=1

# afterwards, on any node with the venv
python -m benchmarking.cli report --store-dir /scratch/$USER/InstaRoad/benchmarks \
    --metric iou --metric f1 --out report.md
```

`STAGE=fit` still logs torchmetrics IoU/F1 to wandb (training-time telemetry); the store is the source of truth for model-wise comparison — every stored row flows through `confusion_counts` / `pixel_metrics_from_counts`, and the sharded store is safe under concurrent SLURM jobs.

## On-disk schema

Sharded: each run writes its own file under `runs/`, `chips/` and (when tile plugins ran) `tiles/`. The loaders concatenate the shards; the legacy flat files (`runs.parquet`, `chip_metrics.parquet`) are still read if present.

### `runs/<run_id>.parquet`

One row per evaluated checkpoint.

| column | type | description |
| ------ | ---- | ----------- |
| `run_id` | string (UUID) | Primary key. Generated when the run starts; also the shard filename. |
| `run_started_at` / `run_finished_at` | timestamp (UTC) | Wall-clock bounds of the run. |
| `model_name` | string | The identifier the stats pair/group on, `{family}_{exp_tag}` by convention (e.g. `unet_cdngi`, `sr_r2a_cdngi`). |
| `model_family` | string | Loader family: `unet` or `sr`. |
| `exp_tag` | string | Experiment tag (`cdngi`, `osm`, `r2a_cdngi`, ...). |
| `label_source` | string | GT label provenance (`cdngi`, `osm`, `overture`). Guardrail key. |
| `mask_source` | string | How GT was read: `csv` (unet), `graph` or `raster` (sr). Guardrail key. |
| `mask_dirname` | string | Alternative mask dir, when used (`mask_osm_10`, `mask_osm_2pt5`). |
| `config_hash` | string | First 12 hex chars of the SHA-256 of the canonicalised (parsed, key-sorted) `--config-yaml`. Empty when no config was passed. |
| `config_yaml` | string | The config file's full text, inline for reproducibility. |
| `seed` | int64 | Training seed. Several seeds per config feed the cross-seed CIs. |
| `checkpoint_path` | string | Absolute path to the evaluated `.ckpt`. |
| `dataset_dir` | string | Absolute dataset root. |
| `dataset_split` | string | Evaluated split. Almost always `test`. |
| `cell_m` | float64 | Footprint cell edge in metres (the chip unit). |
| `chip_px` | int64 | Cell edge in native pixels, derived from the transform (or `--chip-px`). |
| `gt_res_m` | float64 | Ground-truth resolution in metres: native pixel size / family scale (10.0 for unet, 2.5 for sr). Guardrail key. |
| `threshold` | float64 | Sigmoid threshold used to binarise predictions: the checkpoint's hparams, or the `--threshold` override (e.g. the θ* a loss-ablation run tuned on val — see `docs/loss_ablation.md`). |
| `batch_size` | int64 | Chips per forward pass (unet family). |
| `device` | string | `cuda` or `cpu`. |
| `tile_metrics` | string | Comma-joined plugin names that ran (empty if none). |
| `n_tiles` / `n_chips` | int64 | Tiles and chips scored in the run. |

### `chips/<run_id>.parquet`

One row per `(run_id, chip_id)`. Long-form: every chip is its own row.

`chip_id` has the format `{tile_stem}_r{ri}_c{ci}` where `ri`/`ci` index **footprint cells** (`cell_m` ground metres), so the same `chip_id` names the same geography for a 10 m unet run and a 2.5 m sr run — the cross-family pairing key. `tile_id` is denormalised onto every row as a convenience for rolling chips up to tiles or resampling at tile granularity.

| column | type | description |
| ------ | ---- | ----------- |
| `run_id` | string | Foreign key into `runs.parquet`. UUID generated at the start of each run. |
| `chip_id` | string | Chip identifier. Format `{image_stem}_r{patch_row_id}_c{patch_col_id}`. Foreign key into the dataset metadata catalogue. The unit `bootstrap_paired_diff` and `wilcoxon_paired` pair on. |
| `tile_id` | string | Parent image stem. Denormalised so chips roll up to tiles without joining the catalogue. |
| `patch_row_id` | int64 | Row index of this chip in the scoring grid (0-based). |
| `patch_col_id` | int64 | Column index of this chip in the scoring grid (0-based). |
| `model_name` | string | Architecture identifier. Denormalised so the parquet is self-contained for stats queries without joining `runs.parquet`. |
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
| `inference_ms` | float64 | `cuda.synchronize()`-bracketed forward-pass time, amortised over the batch. Excludes data loading and metric computation. |
| *(plugin columns)* | float64 | Per-chip values returned by tile-metric plugins (e.g. `apls`). Nullable: NaN where the metric is undefined on that chip. |

The raw counts (`tp`, `fp`, `fn`, `tn`) are kept alongside the derived metrics. Any new pixel metric (Matthews correlation, Cohen's kappa, balanced accuracy) can be recomputed from the counts without rerunning inference.

### `tiles/<run_id>.parquet`

One row per `(run_id, tile_id)`, written only when tile-metric plugins ran. Columns: `run_id`, `tile_id`, `model_name`, `seed`, plus whatever the plugins returned (e.g. `apls`, `pred_road_frac`). Tile metrics pair on `tile_id` in `compare`/`report` (the CLI resolves the table automatically from the metric name).

## Joining the tables

```python
from benchmarking.store import load_chips, load_joined, load_runs, load_tiles

df = load_joined("benchmarks")      # chips + their run context
tiles = load_tiles("benchmarks")    # tile-level plugin metrics
```

Once joined, every per-chip row carries its full run context (`model_name`, `seed`, `label_source`, `gt_res_m`, etc.) and is ready for arbitrary slicing.

To bring in chip-level attributes from the dataset catalogue:

```python
meta = pd.read_parquet("dataset/metadata.parquet")
df = df.merge(meta[["chip_id", "urbanisation_classification", "road_density"]], on="chip_id")
```

## Using the stats module

The parquet written by `dummy_pipeline.py` is already in the right shape for all three functions. See `example.py` for a complete worked analysis; the snippets below show each function in isolation.

```python
import numpy as np, pandas as pd
from benchmarking.stats import bootstrap_paired_diff, wilcoxon_paired, cross_seed_ci

df = pd.read_parquet("dummy_data/tile_metrics_dummy.parquet")
```

### Paired bootstrap CI between two models

`bootstrap_paired_diff` requires exactly one row per `(model_name, chip_id)`. With multi-seed data, average across seeds first so each chip contributes one value per model:

```python
avg = df.groupby(["model_name", "chip_id"], as_index=False)["f1"].mean()

out = bootstrap_paired_diff(
    avg,
    model_a="dummy_a",
    model_b="dummy_b",
    metric="f1",
    n_boot=2000,
    rng=np.random.default_rng(42),
)
print(f"diff {out['diff_mean']:+.4f}  95% CI [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]  n={out['n_pairs']}")
```

### Wilcoxon signed-rank test

Same pre-aggregation requirement as bootstrap:

```python
out = wilcoxon_paired(avg, "dummy_a", "dummy_b", metric="f1")
print(f"W={out['statistic']:.1f}  p={out['p_value']:.4f}  n={out['n_pairs']}")
```

### Cross-seed confidence interval (training instability)

`cross_seed_ci` is the complement: it wants multiple seeds for the same model. It aggregates each seed's chips to one scalar, then reports a t-interval across seeds.

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

Use `aggregation="macro"` to average the per-chip metric values instead of pooling the counts. Macro and micro diverge when chips are imbalanced in road pixel count; micro weights by chip area, macro weights by chip count.

### Loss-function pilot study

```python
(
    df.query("model_name == 'unet_resnet50'")
      .groupby("loss_fn")["iou"]
      .agg(["mean", "std", "count"])
)
```

A Friedman test (multi-group analogue of Wilcoxon) is appropriate when comparing more than two loss functions on the same chips.

## Design notes

### Chip is the evaluation unit; the footprint cell defines it

Scoring runs per non-overlapping footprint cell so there are enough units for the paired tests to resample over, and so the unit is *geographic* rather than pixel-based: `cell_m` metres of ground per chip regardless of resolution. Inference granularity is a family detail hidden behind the predictor (unet: the cell itself; SEN2SR: pinned 128 px sub-windows stitched to the cell). With few tiles, remember chips within one tile are spatially correlated; tile-level rollups (`tile_id`) are one groupby away.

### Native GT per family; guardrailed comparisons

Each family is scored against the ground truth it trained on, at its own resolution (unet: 10 m, sr: 2.5 m). Pixel metrics therefore compare directly only *within* one GT; `compare`/`report` enforce this via the runs metadata (`gt_res_m`, `label_source`, `mask_source`, ...) and `--force` overrides. Cross-family and cross-label claims ride on tile-level graph metrics (APLS), which are resolution-robust by construction.

### Chip attributes live in the dataset catalogue, not here

`chip_metrics` does not duplicate geometry, split, urbanisation class, or road density. Those belong to the dataset's `metadata.parquet` (owned by `DatasetManager`) and are joined in on demand. This prevents the benchmarking store from drifting out of sync with the dataset catalogue as experiments accumulate.

### One resolution per run

Pixel metrics for a single run are computed against a single ground-truth resolution (`gt_res_m`). To evaluate the same checkpoint against a second GT, run benchmarking twice; the two results land in separate run shards with different `run_id`s.

### Why parquet, not CSV

Parquet preserves int64/float64/datetime/nullable types, compresses well, and reads back into pandas in a single call without dtype hints. The downstream workload (bootstrap resampling, Wilcoxon and Friedman tests, joins between the two tables) is sensitive to dtype and NaN handling in ways that CSV makes painful.

### Why long-form, not wide

Adding a new model, a new seed, or a new loss-function variant requires no schema migration; it produces new rows under the existing columns. Wide-form schemas (one column per model's IoU) couple the schema to the experiment matrix and break this property.

### Append-only, sharded, never overwrite

Each run writes its own shard (tmp-file-then-`os.replace`, atomic on one filesystem); a `run_id` whose shard already exists is an error. Reruns are explicit new runs with new IDs, results are auditable, and concurrent SLURM jobs sharing a store cannot clobber each other — the old flat-file read-concat-rewrite append could lose rows under parallel writers.

### `config_hash` semantics

`config_hash` is the SHA-256 (first 12 characters) of the `--config-yaml` file after canonicalisation: parsed, keys sorted recursively, re-serialised. Formatting and key order don't change the hash; two runs with the same hash were produced by identical configurations and are directly comparable. The full `config_yaml` text is kept inline for inspection. The HPC bench stage passes the experiment's `best_params.yaml` automatically.

### NaN semantics for pixel metrics

A metric with a zero denominator (a chip with no road in either prediction or ground truth) is stored as `NaN`, not 0 or 1. `NaN` is the honest "undefined here" value. Both `bootstrap_paired_diff` and `wilcoxon_paired` drop `NaN` pairs before analysis, so undefined chips are excluded from comparisons rather than biasing them. The `n_pairs` field in the return value reflects how many pairs survived.
