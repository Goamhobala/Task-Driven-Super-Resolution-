#!/bin/bash
# Stage 1 of 2: Optuna search for the JOINT SR + UNet model ONLY (no refit --
# run sr_fit_only.sh afterwards). Mirrors tune_only.sh: the search is
# embarrassingly parallel, so it fans out one INDEPENDENT tuner process per GPU
# over a SHARED sqlite study -- each on a single GPU, no DDP anywhere. This is
# how both GPUs get used during tuning despite Optuna x DDP not mixing.
#
# The search's job here is the joint LR pair: UNet `lr` and SEN2SR `lr_sr` are
# sampled independently (lr_sr's range sits well below lr's; the ratio is the
# task-driven-SR alpha).
#
# Submit through the generic dispatcher (default headers are gpu:2):
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr_tune_only.sh
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr_tune_only.sh MASK_SOURCE=raster STUDY_TAG=osm
# Then, on a separate allocation:
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr_fit_only.sh STUDY_TAG=<tag>
set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
# Every knob honours an environment override (VAR=... bash sr_tune_only.sh),
# so train.sbatch can drive it with KEY=VALUE tokens.
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN}"

SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"   # 0 = load in main process; GDAL/rasterio segfault in subprocesses
PRECISION="${PRECISION:-bf16-mixed}"

# --- Search budget ----------------------------------------------------------
N_TRIALS="${N_TRIALS:-200}"        # TOTAL trials across all workers (SR trials are slower than unet's)
SEARCH_GPUS="${SEARCH_GPUS:-2}"   # one INDEPENDENT tuner process per GPU; match your allocation
TUNE_EPOCHS="${TUNE_EPOCHS:-8}"   # short per-trial budget
PATIENCE="${PATIENCE:-3}"         # per-trial EarlyStopping on val_iou (0 = off)

# Label source: graph = pipeline (CDNGI) via masks_graph parquet; raster = OSM
# HR masks (<split>/masks_osm_2pt5m/, dataset_hr_masks.py --scale 4).
# Use a DIFFERENT STUDY_TAG per setting so searches don't share a study.
MASK_SOURCE="${MASK_SOURCE:-graph}"
STUDY_TAG="${STUDY_TAG:-${MASK_SOURCE}}"   # e.g. MASK_SOURCE=raster STUDY_TAG=osm

ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-imagenet}"

# Search space — the LR pair is the point (both log-uniform).
LR_MIN="${LR_MIN:-1e-5}"          # UNet lr
LR_MAX="${LR_MAX:-1e-2}"
LR_SR_MIN="${LR_SR_MIN:-1e-7}"    # SEN2SR lr_sr: 'nearly frozen' .. UNet scale
LR_SR_MAX="${LR_SR_MAX:-1e-3}"
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-1.0}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-15.0}"
ENCODERS="${ENCODERS:-resnet34}"  # deliberately NOT searched: encoder constancy is the control
BATCH_SIZES="${BATCH_SIZES:-2 4 8}"   # 512px UNet stage is memory-heavy
# ===========================================================================

BASE_CONFIG="$REPO_DIR/src/sr/configs/joint_sr.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
RUN_DIR="/scratch/${USER_NAME}/InstaRoad/runs/sr_optuna_${STUDY_TAG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# Log everything (interactive salloc+bash ignores #SBATCH --output).
LOG_FILE="${RUN_DIR}/tune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"

# Diagnose + fail fast.
echo "host=$(hostname)  USER_NAME=${USER_NAME}  REPO_DIR=${REPO_DIR}"
echo "DATASET_DIR=${DATASET_DIR}  SEN2SR_DIR=${SEN2SR_DIR}  mask_source=${MASK_SOURCE}  study_tag=${STUDY_TAG}"
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  exit 1
fi
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: ${NORM_CONFIG} missing — generate with sentinel2data.cli norm-stats." >&2
  exit 1
fi
if [ ! -f "${SEN2SR_DIR}/model.safetensor" ]; then
  echo "ERROR: SEN2SR weights not at ${SEN2SR_DIR} — prefetch ONCE on a login node:" >&2
  echo "  python -c \"from sr.sen2sr_loader import download_sen2sr; download_sen2sr('${SEN2SR_DIR}')\"" >&2
  exit 1
fi
if [ "${MASK_SOURCE}" = "raster" ]; then
  n_osm=$(find "${DATASET_DIR}"/*/masks_osm_2pt5m -maxdepth 1 -name '*.tif' 2>/dev/null | head -n 100 | wc -l)
  if [ "${n_osm}" -eq 0 ]; then
    echo "ERROR: MASK_SOURCE=raster but no masks under <split>/masks_osm_2pt5m/." >&2
    echo "  Generate with OpenStreetMapTest/dataset_hr_masks.py --scale 4" >&2
    exit 1
  fi
fi

# Use the scratch venv + put src/ on PYTHONPATH so `sr` / `unet` resolve.
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1             # flush stdout live -> the log fills as it runs
echo "python=$(which python)"

STORAGE="sqlite:///${RUN_DIR}/study.db"
STUDY_NAME="sr_optuna_${STUDY_TAG}_seed${SEED}"

run_tuner () {   # $1=gpu id (empty = no pin)  $2=n-trials  $3=seed
  local gpu="$1" ntrials="$2" seed="$3" pin=""
  [ -n "$gpu" ] && pin="CUDA_VISIBLE_DEVICES=$gpu"
  env $pin python -m sr.tune \
    --base-config "$BASE_CONFIG" \
    --base-config "$NORM_CONFIG" \
    --dataset-dir "$DATASET_DIR" \
    --sen2sr-dir "$SEN2SR_DIR" \
    --mask-source "$MASK_SOURCE" \
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
    --lr-sr-min "$LR_SR_MIN" --lr-sr-max "$LR_SR_MAX" \
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
echo "Next: sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr_fit_only.sh STUDY_TAG=${STUDY_TAG}"
