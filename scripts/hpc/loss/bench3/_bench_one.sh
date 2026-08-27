#!/bin/bash
# Bench ONE (arm, seed) on the TEST split. No fitting, ever.
#
# The third test seed already exists as a trained checkpoint for every arm --
# what is missing is only its test ROW. So this drives STAGE=bench directly and
# refuses to run if the checkpoint is absent, rather than letting the engine's
# fit path quietly retrain a 50-epoch model because a directory was empty.
#
#   env MODEL_NAME=... EXP_TAG=... LOSS_ARM=... SEED=0 bash _bench_one.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
USER_NAME="${USER:-$(whoami)}"
RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"

: "${MODEL_NAME:?_bench_one.sh needs MODEL_NAME}"
: "${EXP_TAG:?_bench_one.sh needs EXP_TAG}"
: "${LOSS_ARM:?_bench_one.sh needs LOSS_ARM}"
: "${SEED:?_bench_one.sh needs SEED}"

RUN_DIR="${RUNS_ROOT}/sr_${EXP_TAG}_${LOSS_ARM}_holdout_seed${SEED}"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_jointsr_final.ckpt"

if [ ! -f "$CKPT" ]; then
  echo "SKIP ${MODEL_NAME} seed${SEED}: no checkpoint at ${CKPT}" >&2
  echo "     (seed-0 arms were fitted on Lightning/Modal — upload that run dir first)" >&2
  exit 3
fi
# theta comes from the VAL sweep. Without it _stages_tv.sh aborts rather than
# scoring at a silent 0.5, which is correct -- but as a pool we would rather
# name the arm than emit a wall of engine text.
if [ ! -f "${RUN_DIR}/sweep.json" ]; then
  echo "SKIP ${MODEL_NAME} seed${SEED}: no sweep.json (no operating point)" >&2
  exit 4
fi

# Match the columns the seed-1/2 test rows already carry. A ragged store makes
# cross_seed_ci silently drop whichever metric one seed happens to lack.
export MODEL_NAME EXP_TAG LOSS_ARM SEED
export STAGE=bench
export BENCH_SPLIT=test
export TILE_METRICS="${TILE_METRICS:-apls,cldice}"
export BUFFER_PX="${BUFFER_PX:-1,2,3,4,5}"
export AP_BINS="${AP_BINS:-101}"
export LABELS="${LABELS:-new}"
export TRAIN_SPLITS="train"     # holdout protocol: keeps MERGE_VAL off
export UPSAMPLER="bicubic"; export FREEZE_SR="false"; export SR_PAD=0
export ADAPTIVE_NORM="${ADAPTIVE_NORM:-0}"
export NORM_RECALIBRATE="${NORM_RECALIBRATE:-off}"
export STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_test3}"

echo "=== BENCH ${MODEL_NAME} seed${SEED} -> test  (theta from sweep.json) ==="
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
