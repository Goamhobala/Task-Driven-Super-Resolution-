# Task-Driven Super-Resolution for Road Extraction from Sentinel-2

Road extraction from 10 m Sentinel-2 imagery, with learned super-resolution as
the front end, fine-tuned by the segmentation loss alone.

Roads are the hard case for medium-resolution remote sensing: a residential
street is narrower than a Sentinel-2 pixel, so the structure that defines a
road network - thin, connected, continuous - falls below the sampling grid.
This repository contains the dataset pipeline, models, ablations, evaluation
suite and analysis used to study whether a super-resolution front end recovers
enough of that structure to be worth its cost, and how the answer depends on
the loss, the operating point and the SR architecture.

**Live demo:** https://instaroad-static.onrender.com/
(inference back end: https://huggingface.co/spaces/Goamhobala/instaroad-demo)

---

## What is here

### A purpose-built South African dataset

`src/sentinel2data/` builds the ROSA dataset from Google Earth Engine exports:
60 sampling zones of 25 km, stratified across 10 biomes and three urbanisation
classes, each exported as a 5x5 sheet of 512x512 tiles at 10 m. Every tile
carries 23 bands (Sentinel-2 optical, Sentinel-1 SAR ascending and descending,
topographic derivatives, and three built-up products) alongside 2.5 m road
masks rasterised from national road-centreline data.

1247 tiles in total. The splits are **site-disjoint** - 42 training sites, 9
validation, 9 test, with no zone appearing in two splits - so test performance
measures generalisation to unseen geography rather than to unseen crops of
seen geography.

### Segmentation baseline and loss ablation

`src/unet/` is the reference segmentation stack: a ResNet34 U-Net over the
ROSA splits, with the per-chip IoU/F1 metrics and checkpoint conventions that
every later experiment inherits unchanged, so treatments stay comparable.

`docs/loss_ablation.md` documents a systematic comparison of loss families -
distribution (BCE, weighted cross-entropies), region (Dice, Tversky) and
skeleton (clDice, Skeleton Recall) - under a greedy slot search over

    L = w_pix * L_pixel + w_reg * L_region + w_skel * L_skeleton

Thin-structure segmentation is exactly where the choice of loss stops being
cosmetic, and the ablation is run under identical conditions across arms.

### Super-resolution front ends

`src/sr/` holds the resolution-enhancement experiments. One LightningModule
covers every arm, subclassing the baseline so only the upsampler treatment
varies:

| Arm | Upsampler | Treatment |
| --- | --- | --- |
| R0 | bicubic x4 | parameter-free control |
| R1 | SEN2SR-Lite RGBN x4 | frozen SR preprocessing |
| R2 | SEN2SR-Lite RGBN x4 | fine-tuned by the segmentation loss alone |
| R4 | SR4RS | the same contrast on a GAN-trained generator |

R2 and R4 are *task-driven* super-resolution: there is no reconstruction,
perceptual or adversarial term at fine-tuning time. The only gradient reaching
the SR network comes from the segmentation loss, scaled by differential
learning rates (`lr_sr << lr`). The question is not whether the SR output looks
better but whether it carries more road evidence.

Two subsidiary studies fall out of this:

- **The hard-constraint 2x2** (`docs/hc_2x2_plan.md`) crosses the FFT
  low-frequency constraint with the generator architecture, separating "the
  constraint" from "the architecture" in the central claim. SEN2SR's shipped
  constraint pins each band's DC term to the bicubic input, which is what stops
  task-only fine-tuning from walking the generator off the reflectance scale -
  a drift SR4RS, which has no such anchor, exhibits measurably.
- **Adaptive post-SR normalisation** (`docs/adaptive_norm_plan.md`) tracks the
  SR output distribution with stop-gradient EMA buffers, because a frozen
  z-score is only valid while that distribution stays put.

### Linear probes on the SR front ends

`docs/sr_linear_probe.md` describes the RL-series. A 24 M-parameter U-Net can
compensate for a great deal, so `R2 - R1` measures "does joint fine-tuning help
*this decoder*", not "does the SR output carry more road evidence". Replacing
the decoder with a per-pixel logistic regression - no spatial context, no
capacity - removes the compensation channel and reads the front end directly.

The demo serves this arm alongside R2 for exactly that contrast: one full
decoder, one 1x1 convolution with four weights and a bias.

### Evaluation suite

`src/benchmarking/` is a standalone evaluation system, not a metrics helper.

| Layer | Module |
| --- | --- |
| Per-chip confusion counts | `confusion_matrix.py` |
| Bootstrap CIs, Wilcoxon, effect sizes | `stats.py` |
| Evaluation runner (unet and sr families) | `runner.py` |
| Sharded parquet result store | `store.py` |
| CLI (eval, sweep, compare, variance, report) | `cli.py` |
| APLS graph connectivity | `graph_metrics.py` |
| clDice skeleton metric | `skeleton_metrics.py` |

Design decisions worth naming:

- **Undefined metrics are NaN, never 0 or 1.** A chip with no road in the
  ground truth has no defined IoU; recording it as zero biases every downstream
  comparison. Bootstrap and Wilcoxon drop NaN pairs instead.
- **The operating point is selected, not assumed.** `sweep` scores a checkpoint
  at every threshold in a grid off a single inference pass and records the
  argmax as theta-star. Selection sweeps run on validation and are stamped
  `purpose: selection`; test sweeps are stamped `purpose: sensitivity` so a
  test-set sweep can never be mistaken for threshold selection.
- **Evaluation reuses the training path's reading, normalisation and mask
  helpers**, so the benchmark cannot drift from what was trained.
- **Connectivity is measured, not inferred from pixels.** Buffered F1 across
  tolerance radii, clDice and APLS all report on topology, which is what a road
  network is actually for.

### Visualisation

`src/sr/viz_*.py` renders the figures used throughout the analysis:

| Script | Renders |
| --- | --- |
| `viz_single.py` | one 128 px crop as a row: bicubic x4, SR image, prediction, ground truth |
| `viz_tile.py` | a complete tile - 512 px at 10 m in, 2048 px at 2.5 m out - as standalone full-resolution panels |
| `viz_grid.py` | a grid of crops across checkpoints |
| `viz_models.py` | several checkpoints on one crop, side by side |
| `viz_sr.py` | the SR output on its own |
| `viz_tolerance.py` | the buffered-F1 tolerance sweep |

Two conventions matter. Tiles are scored in the **same windows the benchmark
used**, not in one pass, because convolution borders differ between a 256 px
cell and a 512 px tile - a whole-tile pass produces a prediction that merely
resembles the benchmarked one. And the percentile stretch is computed once from
the original 10 m reflectance and applied to every panel, so no model looks
sharper purely because its histogram moved.

---

## Repository layout

    src/sentinel2data/   ROSA dataset construction, norm stats, rasterisation
    src/unet/            segmentation baseline, losses, LightningCLI
    src/sr/              SR arms, SR4RS torch port, probes, visualisation
    src/benchmarking/    evaluation suite, metrics, statistics, result store
    src/dlinknet/        D-LinkNet baseline
    src/terramind/       geospatial foundation-model baseline
    scripts/             cluster, Lightning, Kaggle and Modal run harnesses
    docs/                protocol documents, one per study
    tests/               15 test modules covering the above
    demo/                the public demo: manifests, serving code, static UI

---

## Setup

The repository uses submodules, so clone recursively:

    git clone --recursive https://github.com/Goamhobala/Task-Driven-Super-Resolution-.git

Requires Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/).

    uv sync --extra sr          # SR experiments (implies the baseline stack)
    uv sync --extra unet        # segmentation baseline only
    uv sync --extra benchmarking

Fetch the SEN2SR weights once, on a machine with internet - compute nodes
typically have none:

    python -c "from sr.sen2sr_loader import download_sen2sr; download_sen2sr('models/SEN2SRLite_RGBN')"

## Running

Train a joint SR and segmentation arm:

    python -m sr.cli fit --config src/sr/configs/joint_sr.yaml \
                         --config src/unet/configs/norm_stats.yaml

Select an operating point on validation, then evaluate on test at that point:

    python -m benchmarking.cli sweep --dataset-dir <ROSA_New/ROSADataset> \
        --checkpoint <ckpt> --model sr --split val --criterion iou
    python -m benchmarking.cli eval  --dataset-dir <ROSA_New/ROSADataset> \
        --checkpoint <ckpt> --model sr --split test --threshold <theta*>

Report across a result store:

    python -m benchmarking.cli report --store-dir <store> \
        --metric f1 --metric iou --metric apls --metric cldice --aggregation both

Render a tile figure:

    python -m sr.viz_tile --ckpt <ckpt> --dataset-dir <root> \
        --split test --tile <tile stem> --sr-dir models/SEN2SRLite_RGBN

Run harnesses for SLURM, Lightning Studio, Kaggle and Modal live under
`scripts/`; each wraps the same staged engine so an arm is defined once and
runs anywhere.

## Documentation

    mkdocs serve

Protocol documents in `docs/` are written per study and record what was
decided and why, including the design decisions that did not work out.

## Demo

`demo/` contains the deployment described at the top of this file: a static
map front end served from a CDN, and a GPU inference Space. The browser sends a
cell identifier rather than imagery, and the Space holds its own copy of the
tiles. `demo/space/parity.py` asserts that the serving path reproduces the
benchmarked windowing bit-for-bit, because the two known failure modes here -
reflectance-scale coupling and eval-collapsed convolutions - produce a
plausible-looking wrong mask rather than an error.

`docs/demo_hosting_plan.md` records the full design and the measured numbers.

## Acknowledgements

The dataset draws on Sentinel-1 and Sentinel-2 imagery via Google Earth Engine,
with road centrelines from national mapping data. SEN2SR and SR4RS are
third-party super-resolution models used as front ends; SR4RS was ported from
its original TensorFlow SavedModel to PyTorch here, with layer-by-layer parity
verified against the reference implementation.
