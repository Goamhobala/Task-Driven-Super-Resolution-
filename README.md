# Task-Driven Super-Resolution+

**Guiding Pretrained Super-Resolution to Enhance Road Segmentation from 10 m
Sentinel-2 Imagery**

Jing J. Yeh, Patrick Marais, Jonathan P. Shock — University of Cape Town

Sentinel-2 is free and revisits every five days, but at 10 m most roads are
only one to two pixels wide. Super-resolution is a natural remedy, yet
pretrained SR models used as a fixed preprocessing step have been found to
*underperform* bicubic interpolation for road segmentation: they are optimised
for human perceptual quality or pixel-level reconstruction, neither of which
necessarily correlates with what a downstream road extractor needs.

This repository contains the dataset pipeline, models, ablations, evaluation
suite and analysis for two task-driven approaches to adapting pretrained SR
models to road segmentation.

**Live demo:** https://instaroad-static.onrender.com/
(inference back end: https://huggingface.co/spaces/Goamhobala/instaroad-demo)

---

## Research questions

1. Can a pretrained super-resolution model be guided by road-segmentation
   supervision to produce outputs that improve downstream road extraction from
   10 m Sentinel-2 imagery?
2. How does task-driven adaptation affect the image fidelity and behaviour of
   the super-resolution model, and to what extent can architectural constraints
   preserve image-like outputs while retaining downstream task performance?
3. Is a high-capacity downstream road extractor necessary at all once the
   super-resolution model has been adapted for the task?

## TDSR-JT — Task-Driven Super-Resolution Joint Training

A pretrained SR model and a ResNet-34 U-Net are
optimised end to end under the segmentation objective alone, with differential
learning rates so the SR representation adapts more conservatively than the
segmentation network. There is no reconstruction, perceptual or adversarial
term at fine-tuning time. Unlike prior task-driven SR work, the framework
assumes no high-resolution reference imagery and no task network pretrained on
it — both are unavailable across much of the globe, and regional mismatch
corrupts the training signal rather than merely the initialisation.

## TDSR-LP-FT — Linear Probe then Fine-Tune

The U-Net is replaced by a
**five-parameter** logistic head: a single 1x1 convolution mapping four bands
to one logit. With the SR module frozen this is a linear probe, measuring how
much road evidence is linearly decodable from the SR output; released and
fine-tuned jointly with the warm-started probe, it follows the
linear-probe-then-fine-tune protocol. Because the head remains a five-parameter
pixelwise projection throughout, any task-relevant spatial representation must
be supplied by the SR model itself.

## Supporting mechanisms

- **Fourier-based Hard Constraint (FHC).** Task-only supervision leaves the SR
  output unanchored to the observed input — nothing in the objective requires
  the adapted SR to remain an image. FHC fuses the low-frequency component of
  the bicubically upsampled observation with the high-frequency component of
  the SR output via FFT, constraining adaptation while leaving learned detail
  adaptable. Without it, SR4RS grows out of bounds in early epochs, producing
  bright pseudo-colours before the segmentation gradients recalibrate it.
- **Adaptive standardisation (AdaptStd).** As the SR output evolves, the
  U-Net's precomputed per-band statistics go stale and segmentation degrades.
  AdaptStd updates them with an exponential moving average of minibatch
  statistics during SR training, and replaces the running estimates with a
  single-pass computation before inference. Unlike BatchNorm, the statistics
  are treated as constants and receive no gradients.

## Experiment arms

All arms predict 2.5 m road masks from 10 m Sentinel-2 RGB+NIR. Suffixes `a`
and `b` denote FHC-constrained and bare SR variants.

| ID | Configuration | SR model | FHC | SR weights | Extractor | Isolates |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | Bicubic 4x | — | — | — | U-Net (ResNet-34) | shared reference |
| R1a | Frozen SR | SEN2SR-Lite | yes | pretrained, frozen | U-Net | R1a-R0: pretrained SR (constrained) |
| R1b | Frozen SR | SEN2SR-Lite | no | pretrained, frozen | U-Net | R1b-R0: pretrained SR (bare) |
| R2a | TDSR-JT | SEN2SR-Lite | yes | fine-tuned jointly | U-Net | R2a-R1a: task-driven adaptation |
| R2b | TDSR-JT | SEN2SR-Lite | no | fine-tuned jointly | U-Net | R2b-R1b: task-driven adaptation |
| R3a | Frozen SR | SR4RS | yes | pretrained, frozen | U-Net | R3a-R0: pretrained SR (constrained) |
| R3b | Frozen SR | SR4RS | no | pretrained, frozen | U-Net | R3b-R0: pretrained SR (bare) |
| R4a | TDSR-JT | SR4RS | yes | fine-tuned jointly | U-Net | R4a-R3a: task-driven adaptation |
| R4b | TDSR-JT | SR4RS | no | fine-tuned jointly | U-Net | R4b-R3b: task-driven adaptation |

The design crosses the constraint against the architecture, so FHC's effect can
be separated from the choice of SR model. The two SR models are deliberately
opposed in objective: **SEN2SR-Lite** is a CNN trained on pixel-wise
reconstruction loss and ships with FHC; **SR4RS** is a GAN, which produces finer
detail but is more prone to hallucination.

TDSR-LP-FT (`docs/sr_linear_probe.md`) runs the same SR front ends read by the
five-parameter head instead of the U-Net.

## Dataset

ROSA pairs a cloud-free Sentinel-2 mosaic with road vectors from the South
African government's CD:NGI archive: **1,254 chips of 512x512 px across 60
scenes**, stratified by urbanisation (rural, peri-urban, urban) and by ten
classes following SANBI's VEGMAP project — nine official biomes plus Azonal
Vegetation. Urban locations were hand-picked; rural and peri-urban were drawn
at random from a custom stratification mask combining the World Settlement
Footprint and ESA urbanisation layers, then checked manually.

Labels are rasterised in both pixel and graph representations at 2.5 m, with
road width assigned by type (8 m for highways and primary roads, 5 m
otherwise). **The test split was manually cleaned for completeness and
correctness** — incorrect labels and incomplete graphs were repaired rather
than deleted, so no observer bias was introduced by removing inconvenient
tiles. Splits are geographically disjoint, so test performance measures
generalisation to unseen regions.

`src/sentinel2data/` builds the dataset; the Earth Engine export scripts live
under `scripts/GEE/`.

## Evaluation

Pixel F1 does not capture connectivity, which is what a road network is for.
The suite therefore reports:

- **APLS** (Average Path Length Similarity) — connectivity, computed over the
  extracted road graph rather than per pixel
- **Buffered F1** at tolerances of 1 to 5 px — the labels are rasterised from
  vector centrelines and are not pixel-perfect, so tolerance separates
  localisation error from genuinely wrong predictions
- **Average Precision** for tuning — threshold-independent, and better suited
  than AUROC to imbalanced binary problems

Statistics are non-parametric by design: with n=9 test regions, normality of
paired regional differences cannot be assumed, and the stratified sampling
means differences are drawn from a heterogeneous mixture rather than one
population. Results use **95% bootstrap confidence intervals (5000 resamples)**
alongside **exact Wilcoxon signed-rank p-values** over the nine regions, with
**Hochberg adjustment** within each research-question family. Following the ASA
statement on p-values, conclusions are not drawn from thresholds alone but
graded as **Strong**, **Suggestive** or **Limited** evidence by combining effect
magnitude, pointwise uncertainty, regional directional agreement and
multiplicity-adjusted rank evidence.

`src/benchmarking/` implements this end to end:

| Layer | Module |
| --- | --- |
| Per-chip confusion counts | `confusion_matrix.py` |
| Bootstrap CIs, Wilcoxon, effect sizes | `stats.py` |
| Evaluation runner | `runner.py` |
| Sharded parquet result store | `store.py` |
| CLI (eval, sweep, compare, variance, report) | `cli.py` |
| APLS | `graph_metrics.py` |

Two conventions worth naming. **Undefined metrics are NaN, never 0 or 1** — a
chip with no road has no defined IoU, and recording it as zero biases every
comparison; bootstrap and Wilcoxon drop NaN pairs instead. And **the operating
point is selected, not assumed**: `sweep` scores a checkpoint at every threshold
off a single inference pass, stamping validation sweeps `purpose: selection`
and test sweeps `purpose: sensitivity`, so a test sweep can never be mistaken
for threshold selection.

## Loss functions

A separate study (`docs/loss_ablation.md`) compares loss families on the U-Net
baseline: pixel-wise (BCE, weighted BCE), region (Dice and Tversky variants),
and road-specific attention-reweighted cross-entropies — **GapLoss**, which
raises the weight of pixels within a buffer of predicted road endpoints, and the
**Topological Loss** with its T2 and T4 extensions, which convolve the binarised
prediction with banks of rotated directional kernels to weight pixels lying in a
corridor between collinear road fragments. Both were developed and validated on
sub-metre aerial imagery; whether mechanisms operating on endpoints and
directional corridors remain meaningful at 10 m is untested a priori, which is
what the study establishes.

## Findings

TDSR-JT recovers much of the degradation caused by frozen SR preprocessing,
with suggestive evidence of improvement over the bicubic baseline in some
configurations, though gains are not consistent across SR models and
constraints. SEN2SR-Lite with FHC achieves the highest buffered F1 among
TDSR-JT configurations, while unconstrained SR4RS achieves the highest APLS.
FHC preserves radiometric consistency but provides no general downstream
advantage.

Under TDSR-LP-FT, SR4RS with only a five-parameter linear head surpasses the
bicubic + U-Net baseline and attains buffered F1 comparable to SR4RS + U-Net —
suggesting task supervision can reshape some pretrained SR models into strongly
road-discriminative representations without a high-capacity downstream
extractor.

## Visualisation

`src/sr/viz_*.py` renders the figures used throughout the analysis:

| Script | Renders |
| --- | --- |
| `viz_single.py` | one 128 px crop as a row: bicubic x4, SR image, prediction, ground truth |
| `viz_tile.py` | a complete tile — 512 px at 10 m in, 2048 px at 2.5 m out — as full-resolution panels |
| `viz_grid.py` | a grid of crops across checkpoints |
| `viz_models.py` | several checkpoints on one crop, side by side |
| `viz_sr.py` | the SR output alone |
| `viz_tolerance.py` | the buffered-F1 tolerance sweep |

Two conventions matter. Tiles are scored in **the same windows the benchmark
used**, not in one pass, because convolution borders differ between a 256 px
cell and a 512 px tile. And the percentile stretch is computed once from the
original 10 m reflectance and applied to every panel, so no model looks sharper
purely because its histogram moved.

---

## Repository layout

    src/sentinel2data/   ROSA construction, norm stats, rasterisation
    src/unet/            segmentation baseline, losses, LightningCLI
    src/sr/              SR arms, FHC, AdaptStd, SR4RS port, probes, visualisation
    src/benchmarking/    evaluation suite, metrics, statistics, result store
    src/dlinknet/        D-LinkNet baseline
    src/terramind/       geospatial foundation-model baseline
    scripts/GEE/         Earth Engine export scripts
    scripts/             SLURM, Lightning, Kaggle and Modal run harnesses
    docs/                protocol documents, one per study
    tests/               15 test modules
    demo/                the public demo: manifests, serving code, static UI

## Setup

Clone recursively — the repository uses submodules:

    git clone --recursive https://github.com/Goamhobala/Task-Driven-Super-Resolution-.git

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

    uv sync --extra sr            # SR experiments (implies the baseline stack)
    uv sync --extra unet          # segmentation baseline only
    uv sync --extra benchmarking

Fetch the SEN2SR weights once, on a machine with internet — compute nodes
typically have none:

    python -c "from sr.sen2sr_loader import download_sen2sr; download_sen2sr('models/SEN2SRLite_RGBN')"

## Running

Train a TDSR-JT arm (override `model.upsampler` and `freeze_sr` for the other
configurations):

    python -m sr.cli fit --config src/sr/configs/joint_sr.yaml \
                         --config src/unet/configs/norm_stats.yaml

Select an operating point on validation, then evaluate on test at that point:

    python -m benchmarking.cli sweep --dataset-dir <ROSA root> \
        --checkpoint <ckpt> --model sr --split val --criterion iou
    python -m benchmarking.cli eval  --dataset-dir <ROSA root> \
        --checkpoint <ckpt> --model sr --split test --threshold <theta*>

Report across a result store:

    python -m benchmarking.cli report --store-dir <store> \
        --metric f1 --metric iou --metric apls --aggregation both

Render a tile figure:

    python -m sr.viz_tile --ckpt <ckpt> --dataset-dir <root> \
        --split test --tile <tile stem> --sr-dir models/SEN2SRLite_RGBN

Run harnesses for SLURM, Lightning Studio, Kaggle and Modal live under
`scripts/`; each wraps the same staged engine, so an arm is defined once and
runs anywhere. Tuning used 30-trial searches on L40S with 10-epoch trial
budgets and 100-epoch refits across three seeds.

## Documentation

    mkdocs serve

Protocol documents in `docs/` are written per study and record what was decided
and why, including the decisions that did not work out.

## Demo

`demo/` contains the deployment linked above: a static map front end served
from a CDN, and a GPU inference Space. The browser sends a cell identifier
rather than imagery, and the Space holds its own copy of the tiles.
`demo/space/parity.py` asserts the serving path reproduces the benchmarked
windowing bit-for-bit, because the two known failure modes here — reflectance
scale coupling and eval-collapsed convolutions — produce a plausible-looking
wrong mask rather than an error. `docs/demo_hosting_plan.md` records the design
and the measured numbers.

## Acknowledgements

Imagery from Sentinel-1 and Sentinel-2 via Google Earth Engine; road
centrelines from the South African CD:NGI archive; biome boundaries from SANBI's
VEGMAP. SEN2SR-Lite and SR4RS are third-party super-resolution models used as
front ends. SR4RS was ported from its original TensorFlow SavedModel to PyTorch
here, with layer-by-layer parity verified against the reference implementation.
