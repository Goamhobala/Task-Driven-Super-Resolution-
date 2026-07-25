#!/bin/bash
# Shared tune/fit engine for the UNet experiment scripts. NOT submitted
# directly — each experiment script (cdngi.sh, osm.sh, ...) sets its config
# (EXP_TAG, DATASET_DIR, MASK_DIRNAME) and sources this file.
#
# STAGE=tune   Optuna search: one INDEPENDENT tuner process per GPU over a
#              shared sqlite study (no DDP anywhere — this is how both GPUs get
#              used despite Optuna x DDP not mixing). Default sbatch headers
#              (gpu:2) fit this stage.
# STAGE=fit    Refit the best config at full length on ONE GPU, then log test
#              metrics to wandb. Submit with --gres=gpu:1. Refits from scratch
#              by default; RESUME_FIT=1 continues a walltime-killed refit from
#              <run dir>/checkpoints/last.ckpt.
# STAGE=bench  Score the fitted checkpoint into the SHARED benchmark store
#              (per-chip confusion-matrix metrics -> compare/variance/report).
#              Works standalone on any existing checkpoint; train_both.sbatch
#              chains it after fit. Submit with --gres=gpu:1.
#
# Replication contract: everything an experiment needs lives in its script +
# this engine; the only knobs meant to vary at submit time are SEED and STAGE.
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-${INSTAROAD_ROOT}/.venv}"

: "${EXP_TAG:?experiment script must set EXP_TAG (e.g. cdngi, osm)}"
: "${DATASET_DIR:?experiment script must set DATASET_DIR}"
MASK_DIRNAME="${MASK_DIRNAME:-}"   # empty = the split CSVs' masks_raster

STAGE="${STAGE:-tune}"             # tune | fit
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"    # 0 = main process; GDAL/rasterio segfault in subprocesses
PRECISION="${PRECISION:-bf16-mixed}"

# --- Tune budget -------------------------------------------------------------
N_TRIALS="${N_TRIALS:-100}"        # TOTAL trials across all workers
SEARCH_GPUS="${SEARCH_GPUS:-1}"    # one tuner process per GPU; match the allocation
TUNE_EPOCHS="${TUNE_EPOCHS:-8}"
PATIENCE="${PATIENCE:-3}"
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-imagenet}"
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-1e-2}"
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-1.0}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-15.0}"
ENCODERS="${ENCODERS:-resnet18 resnet34 resnet50}"
BATCH_SIZES="${BATCH_SIZES:-8 16 32}"

# --- Fit budget --------------------------------------------------------------
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"      # 1 = no DDP (DDP notes: docs in git history)
WANDB_PROJECT="${WANDB_PROJECT:-unet_s2rosa_baseline}"
# =============================================================================

BASE_CONFIG="$REPO_DIR/src/unet/configs/unet.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"
RUN_DIR="${INSTAROAD_ROOT}/runs/unet_${EXP_TAG}_seed${SEED}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"
echo "host=$(hostname)  exp=unet/${EXP_TAG}  stage=${STAGE}  seed=${SEED}"
echo "DATASET_DIR=${DATASET_DIR}  mask_dirname=${MASK_DIRNAME:-<masks_raster>}"

# --- Fail fast ---------------------------------------------------------------
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is INSTAROAD_ROOT set correctly and the data present?" >&2
  exit 1
fi
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: ${NORM_CONFIG} missing — generate with sentinel2data.cli norm-stats." >&2
  exit 1
fi
# NOTE: no pre-flight check that <split>/${MASK_DIRNAME}/ exists — the data
# loader hits it within ~10s and raises a clear error anyway, so a bespoke
# check here only adds its own failure modes (a find|head|pipefail bug once
# killed osm.sh silently right here).

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
echo "python=$(which python)"

# Optional per-experiment CLI extras (arrays survive empty under set -u).
MASK_ARGS_TUNE=(); MASK_ARGS_FIT=()
if [ -n "${MASK_DIRNAME}" ]; then
  MASK_ARGS_TUNE=(--mask-dirname "${MASK_DIRNAME}")
  MASK_ARGS_FIT=(--data.mask_dirname "${MASK_DIRNAME}")
fi

# ============================== STAGE: tune ==================================
if [ "$STAGE" = "tune" ]; then
  # sqlite is fine for ONE job; for two concurrent jobs on one study use the
  # NFS-safe journal backend in BOTH + offset the 2nd job's sampler seeds
  # (STORAGE=journal://<path>, SAMPLER_OFFSET=500). See sr/_stages.sh.
  STORAGE="${STORAGE:-sqlite:///${RUN_DIR}/study.db}"
  SAMPLER_OFFSET="${SAMPLER_OFFSET:-0}"
  STUDY_NAME="unet_${EXP_TAG}_seed${SEED}"

  run_tuner () {   # $1=gpu id (empty = no pin)  $2=n-trials  $3=seed
    local gpu="$1" ntrials="$2" seed="$3" pin=""
    [ -n "$gpu" ] && pin="CUDA_VISIBLE_DEVICES=$gpu"
    env $pin python -m unet.tune \
      --base-config "$BASE_CONFIG" \
      --base-config "$NORM_CONFIG" \
      --dataset-dir "$DATASET_DIR" \
      ${MASK_ARGS_TUNE[@]+"${MASK_ARGS_TUNE[@]}"} \
      --out "$RUN_DIR" \
      --num-workers "$NUM_WORKERS" \
      --devices 1 \
      --n-trials "$ntrials" \
      --max-epochs "$TUNE_EPOCHS" \
      --patience "$PATIENCE" \
      --precision "$PRECISION" \
      --seed "$seed" \
      --train-seed "$SEED" \
      --study-name "$STUDY_NAME" \
      --storage "$STORAGE" \
      --encoder-weights "$ENCODER_WEIGHTS" \
      --lr-min "$LR_MIN" --lr-max "$LR_MAX" \
      --pos-weight-min "$POS_WEIGHT_MIN" --pos-weight-max "$POS_WEIGHT_MAX" \
      --encoders $ENCODERS \
      --batch-sizes $BATCH_SIZES
  }

  echo "=== OPTUNA SEARCH (n_trials=$N_TRIALS across ${SEARCH_GPUS} GPU(s), ${TUNE_EPOCHS} epochs/trial) ==="
  # Sampler seeds: SEED*1000+worker, so workers within a run differ (no
  # duplicate proposals) AND no sampler seed ever recurs across SEED runs
  # (SEED+g would make e.g. SEED=0/worker1 collide with SEED=1/worker0,
  # correlating the startup trials of nominally independent runs).
  if [ "$SEARCH_GPUS" -le 1 ]; then
    run_tuner "" "$N_TRIALS" "$(( SEED * 1000 + SAMPLER_OFFSET ))"
  else
    PER_WORKER=$(( (N_TRIALS + SEARCH_GPUS - 1) / SEARCH_GPUS ))
    echo "  fanning out ${SEARCH_GPUS} workers x ${PER_WORKER} trials each"
    pids=()
    for (( g=0; g<SEARCH_GPUS; g++ )); do
      run_tuner "$g" "$PER_WORKER" "$(( SEED * 1000 + SAMPLER_OFFSET + g ))" &
      pids+=($!)
      sleep 3   # stagger so worker 0 creates the study before the others attach
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    [ "$fail" -eq 0 ] || { echo "ERROR: an Optuna search worker failed (see log above)." >&2; exit 1; }
  fi
  echo "=== SEARCH DONE ===  best_params.yaml + study.db in $RUN_DIR"
  echo "Next: bash scripts/LightningStudio/run.sh unet/${EXP_TAG}.sh STAGE=fit SEED=${SEED}"
  exit 0
fi

# ============================== STAGE: bench =================================
# Score the fitted checkpoint into the shared benchmark store. Wandb keeps the
# torchmetrics test numbers (STAGE=fit); the store is the source of truth for
# model-wise comparison — every row flows through benchmarking.confusion_matrix,
# and the sharded store is safe under concurrent SLURM jobs.
if [ "$STAGE" = "bench" ]; then
  CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_best.ckpt"
  if [ ! -f "$CKPT" ]; then
    if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
      echo "WARN: best checkpoint missing; benchmarking last.ckpt instead." >&2
      CKPT="${RUN_DIR}/checkpoints/last.ckpt"
    else
      echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — run STAGE=fit first." >&2
      exit 1
    fi
  fi

  STORE_DIR="${STORE_DIR:-${INSTAROAD_ROOT}/benchmarks}"   # SHARED across experiments
  MODEL_NAME="${MODEL_NAME:-unet_${EXP_TAG}}"    # {family}_{exp}: what the stats pair/group on
  LABEL_SOURCE="${LABEL_SOURCE:-${EXP_TAG}}"     # unet exp tags ARE the label source (cdngi|osm)
  BENCH_SPLIT="${BENCH_SPLIT:-test}"
  TILE_METRICS="${TILE_METRICS:-apls}"           # comma-separated plugins; '' disables

  CONFIG_ARGS=()
  [ -f "${RUN_DIR}/best_params.yaml" ] && CONFIG_ARGS=(--config-yaml "${RUN_DIR}/best_params.yaml")
  MASK_ARGS_BENCH=()
  [ -n "${MASK_DIRNAME}" ] && MASK_ARGS_BENCH=(--mask-dirname "${MASK_DIRNAME}")
  METRIC_ARGS=()
  if [ -n "${TILE_METRICS}" ]; then
    IFS=',' read -r -a _TMS <<< "${TILE_METRICS}"
    for _tm in "${_TMS[@]}"; do METRIC_ARGS+=(--tile-metric "${_tm}"); done
  fi

  echo "=== BENCH (ckpt=$(basename "$CKPT"), model_name=${MODEL_NAME}, seed=${SEED}, tile_metrics=${TILE_METRICS:-none}) ==="
  python -m benchmarking.cli eval \
    --dataset-dir "$DATASET_DIR" \
    --checkpoint "$CKPT" \
    --model unet \
    --model-name "$MODEL_NAME" \
    --seed "$SEED" \
    --store-dir "$STORE_DIR" \
    --split "$BENCH_SPLIT" \
    --exp-tag "$EXP_TAG" \
    --label-source "$LABEL_SOURCE" \
    ${METRIC_ARGS[@]+"${METRIC_ARGS[@]}"} \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    ${MASK_ARGS_BENCH[@]+"${MASK_ARGS_BENCH[@]}"}

  echo "=== BENCH DONE ===  store: ${STORE_DIR}"
  echo "Report: python -m benchmarking.cli report --store-dir ${STORE_DIR}"
  exit 0
fi

# ============================== STAGE: fit ===================================
if [ "$STAGE" != "fit" ]; then
  echo "ERROR: STAGE must be tune, fit or bench, got '${STAGE}'." >&2
  exit 2
fi

BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_best.ckpt"
if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run STAGE=tune first." >&2
  exit 1
fi
echo "--- best hyperparameters ---"; cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

# Refit from scratch by DEFAULT (a stale last.ckpt is ignored, then overwritten).
# RESUME_FIT=1 instead continues a previous refit from its last.ckpt (e.g. a
# walltime-killed job): LightningCLI `fit --ckpt_path` restores the epoch,
# optimizer and the ModelCheckpoint best-score state, so the run finishes the
# remaining epochs with best-checkpoint tracking intact.
LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"
RESUME_ARGS=()
if [ "${RESUME_FIT:-0}" = "1" ]; then
  if [ -f "$LAST_CKPT" ]; then
    echo "=== RESUME_FIT=1: continuing the refit from ${LAST_CKPT} ==="
    RESUME_ARGS=(--ckpt_path "$LAST_CKPT")
  else
    echo "WARN: RESUME_FIT=1 but ${LAST_CKPT} not found — refitting from scratch." >&2
  fi
fi

echo "=== REFIT (best config, ${REFIT_EPOCHS} epochs, ${REFIT_GPUS} GPU) ==="
python -m unet.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  ${MASK_ARGS_FIT[@]+"${MASK_ARGS_FIT[@]}"} \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

# Log the test metrics to the SAME wandb run the refit just created.
if LATEST_RUN=$(readlink -f "$RUN_DIR/wandb/latest-run" 2>/dev/null) && [ -n "$LATEST_RUN" ]; then
  export WANDB_RUN_ID="${LATEST_RUN##*-}"   # .../run-<timestamp>-<id> -> <id>
  export WANDB_RESUME=must
  echo "resuming wandb run ${WANDB_RUN_ID} for the test split"
else
  echo "WARN: could not locate the refit's wandb run; test will log to a fresh run" >&2
fi

# Prefer the best checkpoint; fall back to last.ckpt; fail loudly otherwise.
if [ ! -f "$CKPT" ]; then
  if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
    echo "WARN: best checkpoint missing; testing last.ckpt instead." >&2
    CKPT="${RUN_DIR}/checkpoints/last.ckpt"
  else
    echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — refit produced none. Skipping test." >&2
    exit 1
  fi
fi

echo "=== BENCHMARK (test split, ckpt=$(basename "$CKPT")) ==="
python -m unet.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  ${MASK_ARGS_FIT[@]+"${MASK_ARGS_FIT[@]}"} \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
