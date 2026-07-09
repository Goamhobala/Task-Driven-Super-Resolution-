#!/bin/bash
# Stage 1 of 2: Optuna hyperparameter search ONLY (no refit -- run fit_only.sh
# afterwards). The search is embarrassingly parallel, so it fans out one tuner
# process per GPU over a SHARED sqlite study; each runs on a single GPU (no DDP).
#
# Run on a worker node with GPUs allocated, e.g.:
#   salloc --gres=gpu:2 ...
#   bash scripts/tune_only.sh
# Then, on a single-GPU allocation:
#   bash scripts/fit_only.sh
set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"

# Every knob below honours an environment override (VAR=... bash tune_only.sh),
# so a batch wrapper can call this script twice with different settings.
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"   # 0 = load in main process; GDAL/rasterio segfault in subprocesses
PRECISION="${PRECISION:-bf16-mixed}"

# --- Search budget ----------------------------------------------------------
N_TRIALS="${N_TRIALS:-100}"       # TOTAL trials across all workers (bump as high as you like)
SEARCH_GPUS="${SEARCH_GPUS:-2}"   # one tuner process per GPU; must match your allocation
TUNE_EPOCHS="${TUNE_EPOCHS:-8}"   # short per-trial budget
PATIENCE="${PATIENCE:-3}"         # per-trial EarlyStopping on val_iou (0 = off)

# Pretrained vs random init for the WHOLE search. 'imagenet' or 'none'/'random'.
# Use a DIFFERENT STUDY_TAG per setting so the two searches don't share a study.
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-imagenet}"
STUDY_TAG="${STUDY_TAG:-imagenet}"   # e.g. ENCODER_WEIGHTS=none STUDY_TAG=random for random init

# Label source. Empty (default) = the split CSVs' CDNGI masks_raster; set to a
# mask dir beside imagery (e.g. masks_osm_10m) to tune on OSM labels instead.
# The dir must exist as <split>/<MASK_DIRNAME>/{tile}.tif for every tile
# (generate with OpenStreetMapTest/dataset_hr_masks.py --scale 1). Give the OSM
# run a DIFFERENT STUDY_TAG so it doesn't share the CDNGI study, e.g.
#   MASK_DIRNAME=masks_osm_10m STUDY_TAG=osm bash tune_only.sh
MASK_DIRNAME="${MASK_DIRNAME:-}"

# Search space
LR_MIN=1e-5
LR_MAX=1e-2
POS_WEIGHT_MIN=1.0
POS_WEIGHT_MAX=15.0
ENCODERS="resnet18 resnet34 resnet50"
BATCH_SIZES="8 16 32"
# ===========================================================================

BASE_CONFIG="$REPO_DIR/src/unet/configs/unet.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
RUN_DIR="/scratch/${USER_NAME}/InstaRoad/runs/unet_optuna_${STUDY_TAG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# Log everything (interactive salloc+bash ignores #SBATCH --output).
LOG_FILE="${RUN_DIR}/tune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"

# Diagnose + fail fast.
echo "host=$(hostname)  USER_NAME=${USER_NAME}  REPO_DIR=${REPO_DIR}"
echo "DATASET_DIR=${DATASET_DIR}  encoder_weights=${ENCODER_WEIGHTS}  study_tag=${STUDY_TAG}  mask_dirname=${MASK_DIRNAME:-<CDNGI>}"
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  exit 1
fi
if [ -n "${MASK_DIRNAME}" ]; then
  # nullglob so a missing dir yields an empty array (NOT a find error that
  # pipefail+set -e would turn into a silent 0-second exit).
  shopt -s nullglob
  _alt=( "${DATASET_DIR}"/*/"${MASK_DIRNAME}"/*.tif )
  shopt -u nullglob
  if [ "${#_alt[@]}" -eq 0 ]; then
    echo "ERROR: MASK_DIRNAME=${MASK_DIRNAME} but no *.tif under <split>/${MASK_DIRNAME}/ in ${DATASET_DIR}." >&2
    _split="$(ls -d "${DATASET_DIR}"/*/ 2>/dev/null | head -1)"
    [ -n "${_split}" ] && echo "  label dirs present in ${_split}: $(ls -1 "${_split}" 2>/dev/null | tr '\n' ' ')" >&2
    echo "  (note: your folders are 'mask_osm_10' / 'mask_osm_2pt5' — singular 'mask'.)" >&2
    exit 1
  fi
fi
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: ${NORM_CONFIG} missing — generate with sentinel2data.cli norm-stats." >&2
  exit 1
fi

# Use the scratch venv + put src/ on PYTHONPATH so `unet` resolves (source package).
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1             # flush stdout live -> the log fills as it runs
echo "python=$(which python)"        # sanity: under $VENV_DIR, not base conda

STORAGE="sqlite:///${RUN_DIR}/study.db"
STUDY_NAME="unet_optuna_${STUDY_TAG}_seed${SEED}"

run_tuner () {   # $1=gpu id (empty = no pin)  $2=n-trials  $3=seed
  local gpu="$1" ntrials="$2" seed="$3" pin=""
  [ -n "$gpu" ] && pin="CUDA_VISIBLE_DEVICES=$gpu"
  env $pin python -m unet.tune \
    --base-config "$BASE_CONFIG" \
    --base-config "$NORM_CONFIG" \
    --dataset-dir "$DATASET_DIR" \
    --mask-dirname "$MASK_DIRNAME" \
    --out "$RUN_DIR" \
    --num-workers "$NUM_WORKERS" \
    --devices 1 \
    --n-trials "$ntrials" \
    --max-epochs "$TUNE_EPOCHS" \
    --patience "$PATIENCE" \
    --precision "$PRECISION" \
    --seed "$seed" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE" \
    --encoder-weights "$ENCODER_WEIGHTS" \
    --lr-min "$LR_MIN" --lr-max "$LR_MAX" \
    --pos-weight-min "$POS_WEIGHT_MIN" --pos-weight-max "$POS_WEIGHT_MAX" \
    --encoders $ENCODERS \
    --batch-sizes $BATCH_SIZES
}

echo "=== OPTUNA SEARCH (n_trials=$N_TRIALS across ${SEARCH_GPUS} GPU(s), ${TUNE_EPOCHS} epochs/trial) ==="
if [ "$SEARCH_GPUS" -le 1 ]; then
  run_tuner "" "$N_TRIALS" "$SEED"
else
  PER_WORKER=$(( (N_TRIALS + SEARCH_GPUS - 1) / SEARCH_GPUS ))
  echo "  fanning out ${SEARCH_GPUS} workers x ${PER_WORKER} trials each"
  pids=()
  for (( g=0; g<SEARCH_GPUS; g++ )); do
    run_tuner "$g" "$PER_WORKER" "$(( SEED + g ))" &
    pids+=($!)
    sleep 3   # stagger so worker 0 creates the study before the others attach
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [ "$fail" -eq 0 ] || { echo "ERROR: an Optuna search worker failed (see log above)." >&2; exit 1; }
fi

echo "=== SEARCH DONE ===  best_params.yaml + study.db in $RUN_DIR"
echo "Next: point fit_only.sh's RUN_DIR at ${RUN_DIR} and run it on a single-GPU allocation."
