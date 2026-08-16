#!/bin/bash
# SEED REFIT — sr_r0_new_lcdice_holdout
#
# Re-runs this arm's pilot fit at SEED=1 2 so the reported number can be a
# cross-seed mean +/- std rather than a single draw. Nothing else changes: same
# 50-epoch budget, same holdout protocol (val is NOT folded in), same theta*
# sweep on val at the end, same store. `model_name` carries no seed, so these
# rows group with the existing seed-0 row automatically under `report`.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-lcdice --time=12:00:00 \
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=loss/refit/lcdice.sh
#
# --SCRIPT resolves under $REPO_DIR/scripts/hpc/ whatever the cwd, so running
# from scripts/hpc keeps the command short. The headers in train.sbatch default
# to gpu:2 / cpus 4 / 12 h, hence the overrides. 12 h is generous: two seeds of
# fit (~1.5-2 h each on an L40S) plus four benches is ~5 h, but a chained job
# that dies at the wall clock loses the second seed entirely.
#
# Hyperparameters are seed 0's tune, copied verbatim from
#   sr_r0_new_lcdice_holdout_seed0_L4_lightning
# and baked in, so the cluster needs nothing from runslightning/ and the exact
# config a refit used is readable in the file that ran it.
#
# THE EXPORTS BELOW ARE NOT REDUNDANT WITH THE YAML. `_stages_tv.sh` appends
# --model.pstar / --model.gap_r / --model.warmup_start / ... AFTER the --config
# layers, so those flags OVERRIDE best_params.yaml with the engine's ENV
# defaults. For gap_tl_ce that silently swapped pstar bce->gap_t4_ce and the
# warmup schedule 15/5 -> 30/10. Exporting them pins the belt to the same values
# the overlay carries. (tl_theta / gap_theta / pos_weight are deliberately kept
# out of the belt by the engine, so the overlay alone is authoritative there.)
#
# Only ONE checkpoint is kept, by design: the refit overlay sets `monitor: null`,
# so `unet_s2rosa_jointsr_final.ckpt` IS the final-epoch weights rather than an
# argmax over epochs — there is no "best" to keep. `last.ckpt` holds the same
# weights and is removed once the seed is benched (KEEP_LAST=1 to retain it).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
USER_NAME="${USER:-$(whoami)}"
RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"

EXP_TAG="r0_new"
LOSS_ARM="lcdice"
export MODEL_NAME="sr_r0_new_lcdice_holdout"   # match seed 0 so the store groups the seeds
SEEDS="${SEEDS:-1 2}"

# --- fit-belt pins (see header) ---------------------------------------------
export PSTAR="bce"
export GAP_R="4"
export GAP_K="60.0"
export TL_ELL="5"
export TVERSKY_ALPHA="0.7"
export CL_ALPHA="0.3"
export CL_ITERS="5"
export SKEL_W="1.0"
export SKEL_RADIUS="1"
export WARMUP_START="15"
export WARMUP_RAMP="5"

read -r -d '' BEST_PARAMS <<'YAML' || true
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  upsampler: bicubic
  freeze_sr: false
  sr_pad: 0
  lr: 0.00023355483391598498
  loss_arm: lcdice
  pstar: bce
  gap_r: 4
  gap_k: 60.0
  tl_ell: 5
  tl_theta: 0.375
  gap_theta: 0.5
  tversky_alpha: 0.7
  cl_alpha: 0.3
  cl_iters: 5
  sr_w: 1.0
  sr_radius: 1
  warmup_start: 15
  warmup_ramp: 5
  lr_schedule: cosine
  sr_warmup_epochs: 1.0
  l2sp_lambda: 0.0
data:
  batch_size: 8
  mask_source: raster
  mask_dirname: mask_new_2pt5
trainer:
  precision: bf16-mixed
YAML

for SEED in $SEEDS; do
  RUN_DIR="${RUNS_ROOT}/sr_${EXP_TAG}_${LOSS_ARM}_holdout_seed${SEED}"
  mkdir -p "$RUN_DIR"
  # STAGE=fit refuses to start without this file, and a seed-N dir has never
  # been tuned — so the overlay is planted rather than looked up.
  printf '%s\n' "$BEST_PARAMS" > "$RUN_DIR/best_params.yaml"

  echo "########## sr_r0_new_lcdice_holdout  SEED=${SEED}  FIT ##########"
  env EXP_TAG="$EXP_TAG" LOSS_ARM="$LOSS_ARM" SEED="$SEED" STAGE=fit \
      bash "$REPO_DIR/scripts/hpc/loss/refit/_refit_arm.sh"

  echo "########## sr_r0_new_lcdice_holdout  SEED=${SEED}  BENCH val (selects theta*) ##########"
  env EXP_TAG="$EXP_TAG" LOSS_ARM="$LOSS_ARM" SEED="$SEED" STAGE=bench \
      BENCH_SPLIT=val \
      bash "$REPO_DIR/scripts/hpc/loss/refit/_refit_arm.sh"

  # Test at the SAME theta*: the bench stage reuses the sweep.json the val pass
  # just wrote (the sweep is always on val), so the operating point is still
  # chosen on val and test is only ever read. val -> loss selection for the R
  # series; test -> the reported number. RUN_TEST=0 to skip.
  if [ "${RUN_TEST:-1}" = "1" ]; then
    echo "########## sr_r0_new_lcdice_holdout  SEED=${SEED}  BENCH test (at val's theta*) ##########"
    env EXP_TAG="$EXP_TAG" LOSS_ARM="$LOSS_ARM" SEED="$SEED" STAGE=bench \
        BENCH_SPLIT=test \
        bash "$REPO_DIR/scripts/hpc/loss/refit/_refit_arm.sh"
  fi

  if [ "${KEEP_LAST:-0}" != "1" ]; then
    rm -f "$RUN_DIR/checkpoints/last.ckpt"   # same weights as the final ckpt
  fi
done

echo "=== sr_r0_new_lcdice_holdout: seeds ${SEEDS} done ==="
