# Loss-Function Ablation (Phase A–C)

Implementation of the loss-ablation protocol on the **UNet baseline pipeline**
(`src/unet/` + `RoadDataModule` over the ROSA split CSVs). The protocol
compares representative losses from each family — distribution (BCE, weighted
CEs), region (Dice/Tversky), skeleton (clDice, Skeleton Recall) — under
identical conditions, with the greedy slot-search strategy:

```
L = w_pix · L_pixel + w_reg · L_region + w_skel · L_skeleton
```

## Where things live

| piece | module |
| --- | --- |
| Slot losses + arm factory | `src/unet/losses.py` (`build_loss(arm, **hp)`) |
| Model integration | `src/unet/model.py` — `UNetLightning(loss_arm=..., gap_r=..., ...)`; `loss_arm=None` keeps the legacy Dice + pos-weighted BCE |
| Augmentation | `src/sentinel2data/dataset/augment.py`; toggles on `RoadDataModule` (`aug_flip`, `aug_sharpen`, `aug_noise`, `aug_blur`, `aug_colour`, `aug_p`) — train split only |
| Ablation trainer | `src/unet/train_ablation.py` (protocol-hardened `fit` + threshold sweep) |
| Stage-0 gates | `tests/test_losses.py` (unit), `scripts/stage0_sanity.py` (visual + scale parity) |
| HPC launchers | `scripts/hpc/loss/` — one script per arm (`l1_all.sh` … `l8_all.sh`, `la0_all.sh`, `la1_all.sh`) + the shared `_stages.sh` engine |
| Decision table | `scripts/phase_a_report.py` + `benchmarking.cli report` |

The arm and every loss hyperparameter are `UNetLightning` **hparams**, so each
checkpoint records exactly which loss trained it and loads through
`benchmarking`'s `UNetPredictor` like any other unet checkpoint. The arms are
equally drivable from the plain LightningCLI
(`python -m unet.cli fit ... --model.loss_arm gap_ce --model.gap_r 5
--data.aug_flip true`), but the ablation trainer below is the protocol path.

## Fairness rules (why `train_ablation` exists)

`python -m unet.train_ablation` reuses the exact `UNetLightning` +
`RoadDataModule` building blocks but hardens the run for cross-arm comparison:

* **Fixed epoch budget, no early stopping** — val loss is loss-dependent, so
  stopping on it would give each arm a different operating point.
* **Checkpoint selection on val F1 @ 0.5** (loss-independent, pre-registered);
  `val_loss` is logged, never used for selection.
* **§4.4 scale normalization** — every weighted CE is a weighted *mean*
  (`sum(W·ce)/sum(W)`), so "different loss" is never "different effective LR"
  and one screening LR (1e-3) is valid across arms.
* **§4.5 skeleton warmup** — skeleton-slot weight ramps from 0 inside the
  model (`on_train_epoch_start`), default epochs 30→40 of 100.
* **Plain CE in the arms** — `pos_weight` is itself a distribution-slot
  reweighting, so it only applies to the legacy loss.
* **Protocol-fixed augmentation** — D4 flips/rotations on the train crops
  (`--no-augment` to disable).
* Ends with a **val threshold sweep** (θ = 0.05…0.95 → `sweep.json`): the
  inference threshold is in scope for the protocol decision.

## Running on the HPC (staged, one script per arm)

The arms run through the same `scripts/hpc/` staging mechanism as the unet and
sr experiments. Under the cluster's concurrent-job cap, prefer
`scripts/hpc/train_pair.sbatch` — it runs TWO arms in one gpu:2 job (one per
GPU, each chaining fit→bench, per-side env via `A.KEY=V` / `B.KEY=V`):

```sh
sbatch scripts/hpc/train_pair.sbatch --A=loss/l1_all.sh --B=loss/la0_all.sh
sbatch scripts/hpc/train_pair.sbatch --A=loss/l2_all.sh --B=loss/l3_all.sh
sbatch scripts/hpc/train_pair.sbatch --A=loss/la0_all.sh --B=loss/la0_all.sh A.SEED=1 B.SEED=2
```
 `scripts/hpc/loss/` holds one self-contained script per
arm-table entry, plus the shared `_stages.sh` engine they source. Everything an
arm needs lives in its script; the only knobs meant to vary at submit time are
`SEED`, `STAGE`, and (for a within-arm hyperparameter grid) the relevant hp env
var. The dataset defaults to **`ROSA_all`** (the big dataset).

| script | arm | script | arm |
| --- | --- | --- | --- |
| `l1_all.sh` | `bce` (H1 floor) | `l6_all.sh` | `pstar_tversky` |
| `l2_all.sh` | `gap_ce` (grid `GAP_R∈{3,5,9}`) | `l7_all.sh` | `B*+cldice` |
| `l3_all.sh` | `tl_ce` (grid `TL_ELL∈{3,5,7}`) | `l8_all.sh` | `B*+skelrec` |
| `l4a/l4b_all.sh` | `t2_ce`/`t4_ce` (pending kernels) | `la0_all.sh` | `bce_dice` (anchor) |
| `l5_all.sh` | `pstar_dice` | `la1_all.sh` | `focal_tversky` |

Three stages (mirrors unet/sr, but there is **no Optuna** — the protocol fixes
the screening LR and screens the loss, not the optimiser):

* **`tune`** — deliberate no-op (loss ablation has nothing to search); exists so
  `train_both.sbatch`'s tune→fit→bench chain runs unchanged.
* **`fit`** — trains one arm at the fixed budget (`unet.train_ablation`).
* **`bench`** — scores the checkpoint into the loss store at the val-tuned θ\*.

```sh
# Stage 0 (any machine): unit gates + weight-map sanity render
pytest tests/test_losses.py -q
python scripts/stage0_sanity.py --out runs/stage0 --cross-check

# One arm, full pipeline in one allocation (fit -> bench). --gres=gpu:1 is
# enough (the no-op tune stage doesn't need the 2nd GPU train_both defaults to).
sbatch --gres=gpu:1 scripts/hpc/train_both.sbatch --SCRIPT=loss/l1_all.sh

# Grid a within-arm hyperparameter — each lands as a distinct model_name:
sbatch --gres=gpu:1 scripts/hpc/train_both.sbatch --SCRIPT=loss/l2_all.sh GAP_R=3
sbatch --gres=gpu:1 scripts/hpc/train_both.sbatch --SCRIPT=loss/l2_all.sh GAP_R=9

# Anchor seed-noise band (submit several seeds):
for s in 0 1 2; do
  sbatch --gres=gpu:1 scripts/hpc/train_both.sbatch --SCRIPT=loss/la0_all.sh SEED=$s
done

# Phase B/C reference the earlier winner (P* / B*) via env overrides:
sbatch --gres=gpu:1 scripts/hpc/train_both.sbatch --SCRIPT=loss/l5_all.sh PSTAR=gap_ce GAP_R=5
sbatch --gres=gpu:1 scripts/hpc/train_both.sbatch --SCRIPT=loss/l7_all.sh BSTAR=bce_dice

# Or drive a single stage yourself:
sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=loss/l1_all.sh STAGE=fit
sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=loss/l1_all.sh STAGE=bench
```

Each run dir (`runs/loss_<full_tag>_seed<n>/`) gets `checkpoints/best_f1.ckpt`,
`sweep.json`, `train_meta.json` and `config.yaml` (the resolved run config,
which doubles as the benchmark's `--config-yaml` input). The engine derives a
`model_name` that encodes the arm **and** its discriminating hyperparameter
(e.g. `l2_gap_ce_r5`), without the seed — so replicates of one config share a
`model_name` and pair through the `seed` column, while grid points land as
separate models.

## Benchmarking integration

The `bench` stage scores the selected checkpoint **on the val split at its tuned
θ\*** into the loss store (`STORE_DIR`, default
`/scratch/$USER/InstaRoad/benchmarks_loss`), via the `--threshold` override —
protocol decisions are made on val; test stays held out for the post-decision
confirmation run. Under the hood:

```sh
python -m benchmarking.cli eval \
    --dataset-dir $DATASET_DIR --checkpoint $RUN_DIR/checkpoints/best_f1.ckpt \
    --model unet --model-name l2_gap_ce_r5 --seed 0 \
    --store-dir /scratch/$USER/InstaRoad/benchmarks_loss \
    --split val --threshold 0.35 --exp-tag loss_l2_gap_ce --label-source all \
    --tile-metric apls --config-yaml $RUN_DIR/config.yaml
```

`--tile-metric apls` is on by default in the engine (`TILE_METRICS=apls`;
set `TILE_METRICS=""` to skip): every bench also writes APLS — the
protocol's connectivity metric (see `benchmarking.graph_metrics`) — at both
levels: per-chip onto the chip rows (so `compare`/`report` bootstrap and
Wilcoxon-pair it on `chip_id`, the same unit as the pixel metrics) and a
tile-level rollup with directional scores + graph diagnostics in `tiles/`.

Decision A/B/C then read from both sides:

```sh
python scripts/phase_a_report.py --runs /scratch/$USER/InstaRoad/runs
python -m benchmarking.cli report --store-dir /scratch/$USER/InstaRoad/benchmarks_loss \
    --metric f1 --metric iou --metric apls --out loss_report.md
```

`phase_a_report.py` prints the F1@θ\* table against the `bce_dice` seed-noise
band (margins inside the band = tie → prefer the simpler loss);
`benchmarking.cli report` adds paired bootstrap CIs + Wilcoxon over the
footprint chips — for pixel metrics and chip-level APLS alike, on the same
`chip_id` pairing. Decision A weighs pixel accuracy AND APLS jointly (the
protocol composite).

## Status / pending

* Phase A arms implemented: `bce`, `gap_ce`, `tl_ce`, plus anchors
  `bce_dice`, `pstar_dice`, `pstar_tversky`, `focal_tversky` and the Phase C
  compounds `+cldice` / `+skelrec`.
* `t2_ce` / `t4_ce` raise `NotImplementedError` until the curvature kernels
  from Giannini et al. (2026) are specified — the hook is
  `tl_weight_map(extra_kernels=...)`.
* APLS is implemented (`benchmarking.graph_metrics`, `--tile-metric apls`,
  default-on in the bench stage). Betti-0 and the clDice-metric remain open
  on the same `tile_metrics` plugin seam.
