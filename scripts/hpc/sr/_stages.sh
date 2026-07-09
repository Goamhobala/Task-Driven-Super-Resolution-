#!/bin/bash
# Shared tune/fit engine for the joint-SR experiment scripts. NOT submitted
# directly — each experiment script (r0_cdngi.sh, r2a_cdngi.sh, ...) sets its
# config and sources this file.
#
# Experiment scripts must set:
#   EXP_TAG      e.g. r2a_cdngi (drives the run dir + study name)
#   LABELS       cdngi | overture | osm  (label SOURCE naming — never "graph",
#                which collides with the graph-model thread; cdngi/overture map
#                to the pipeline's masks_graph parquets of the matching dataset,
#                osm maps to the pre-generated <split>/mask_osm_2pt5 rasters)
#   UPSAMPLER    sen2sr | bicubic
#   FREEZE_SR    true | false
#   SR_PAD       reflect-pad in native px (0 = off, 8 = border-artifact fix)
#
# STAGE=tune  Optuna search, one INDEPENDENT tuner per GPU, shared sqlite study
#             (no DDP — that's the Optuna constraint). Default headers = gpu:2.
# STAGE=fit   Refit best config on ONE GPU + test. Submit with --gres=gpu:1.
#
# Replication contract: only SEED and STAGE are meant to vary at submit time.
set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"

: "${EXP_TAG:?experiment script must set EXP_TAG}"
: "${LABELS:?experiment script must set LABELS (cdngi|overture|osm)}"
: "${UPSAMPLER:?experiment script must set UPSAMPLER (sen2sr|bicubic)}"
: "${FREEZE_SR:?experiment script must set FREEZE_SR (true|false)}"
: "${SR_PAD:?experiment script must set SR_PAD (0 = off)}"

STAGE="${STAGE:-tune}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRECISION="${PRECISION:-bf16-mixed}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN}"

# LABELS -> dataset dir + code-level mask_source
case "$LABELS" in
  cdngi)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
    MASK_SOURCE="graph" ;;   # = the CDNGI dataset's own masks_graph parquets
  overture)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_Overture}"
    MASK_SOURCE="graph" ;;   # = the Overture dataset's own masks_graph parquets
  osm)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
    MASK_SOURCE="raster" ;;  # = <split>/mask_osm_2pt5 rasters
  *)
    echo "ERROR: LABELS must be cdngi|overture|osm, got '${LABELS}'." >&2; exit 2 ;;
esac

# --- Tune budget -------------------------------------------------------------
N_TRIALS="${N_TRIALS:-200}"
SEARCH_GPUS="${SEARCH_GPUS:-2}"
TUNE_EPOCHS="${TUNE_EPOCHS:-8}"
PATIENCE="${PATIENCE:-3}"
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-imagenet}"
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-1e-2}"
LR_SR_MIN="${LR_SR_MIN:-1e-7}"   # searched only when UPSAMPLER=sen2sr && !FREEZE_SR
LR_SR_MAX="${LR_SR_MAX:-1e-3}"
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-1.0}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-15.0}"
ENCODERS="${ENCODERS:-resnet34}"      # NOT searched: encoder constancy is the control
BATCH_SIZES="${BATCH_SIZES:-2 4 8}"   # 512px UNet stage is memory-heavy

# --- Fit budget --------------------------------------------------------------
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_joint}"
# =============================================================================

BASE_CONFIG="$REPO_DIR/src/sr/configs/joint_sr.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"
RUN_DIR="/scratch/${USER_NAME}/InstaRoad/runs/sr_${EXP_TAG}_seed${SEED}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"
echo "host=$(hostname)  exp=sr/${EXP_TAG}  stage=${STAGE}  seed=${SEED}"
echo "labels=${LABELS} (mask_source=${MASK_SOURCE})  upsampler=${UPSAMPLER}  freeze_sr=${FREEZE_SR}  sr_pad=${SR_PAD}"
echo "DATASET_DIR=${DATASET_DIR}"

# --- Fail fast ---------------------------------------------------------------
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  exit 1
fi
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: ${NORM_CONFIG} missing — generate with sentinel2data.cli norm-stats." >&2
  exit 1
fi
if [ "${UPSAMPLER}" = "sen2sr" ] && [ ! -f "${SEN2SR_DIR}/model.safetensor" ]; then
  echo "ERROR: SEN2SR weights not at ${SEN2SR_DIR} — prefetch ONCE on a login node:" >&2
  echo "  python -c \"from sr.sen2sr_loader import download_sen2sr; download_sen2sr('${SEN2SR_DIR}')\"" >&2
  exit 1
fi
if [ "${MASK_SOURCE}" = "raster" ]; then
  # -print -quit: no pipe to `head`, so `find` can't die of SIGPIPE and trip
  # `set -o pipefail` (that silently killed the unet osm.sh check).
  first_osm=$(find "${DATASET_DIR}"/*/mask_osm_2pt5 -maxdepth 1 -name '*.tif' -print -quit 2>/dev/null)
  if [ -z "${first_osm}" ]; then
    echo "ERROR: LABELS=osm but no masks under <split>/mask_osm_2pt5/." >&2
    echo "  Generate with OpenStreetMapTest/dataset_hr_masks.py --scale 4" >&2
    exit 1
  fi
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
echo "python=$(which python)"

# ============================== STAGE: tune ==================================
if [ "$STAGE" = "tune" ]; then
  STORAGE="sqlite:///${RUN_DIR}/study.db"
  STUDY_NAME="sr_${EXP_TAG}_seed${SEED}"

  run_tuner () {   # $1=gpu id (empty = no pin)  $2=n-trials  $3=seed
    local gpu="$1" ntrials="$2" seed="$3" pin=""
    [ -n "$gpu" ] && pin="CUDA_VISIBLE_DEVICES=$gpu"
    env $pin python -m sr.tune \
      --base-config "$BASE_CONFIG" \
      --base-config "$NORM_CONFIG" \
      --dataset-dir "$DATASET_DIR" \
      --sen2sr-dir "$SEN2SR_DIR" \
      --mask-source "$MASK_SOURCE" \
      --upsampler "$UPSAMPLER" \
      --freeze-sr "$FREEZE_SR" \
      --sr-pad "$SR_PAD" \
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
      --lr-sr-min "$LR_SR_MIN" --lr-sr-max "$LR_SR_MAX" \
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
    run_tuner "" "$N_TRIALS" "$(( SEED * 1000 ))"
  else
    PER_WORKER=$(( (N_TRIALS + SEARCH_GPUS - 1) / SEARCH_GPUS ))
    echo "  fanning out ${SEARCH_GPUS} workers x ${PER_WORKER} trials each"
    pids=()
    for (( g=0; g<SEARCH_GPUS; g++ )); do
      run_tuner "$g" "$PER_WORKER" "$(( SEED * 1000 + g ))" &
      pids+=($!)
      sleep 3   # stagger so worker 0 creates the study before the others attach
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    [ "$fail" -eq 0 ] || { echo "ERROR: an Optuna search worker failed (see log above)." >&2; exit 1; }
  fi
  echo "=== SEARCH DONE ===  best_params.yaml + study.db in $RUN_DIR"
  echo "Next: sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/${EXP_TAG}.sh STAGE=fit SEED=${SEED}"
  exit 0
fi

# ============================== STAGE: fit ===================================
if [ "$STAGE" != "fit" ]; then
  echo "ERROR: STAGE must be tune or fit, got '${STAGE}'." >&2
  exit 2
fi

BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_jointsr_best.ckpt"
if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run STAGE=tune first." >&2
  exit 1
fi
echo "--- best hyperparameters ---"; cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

# The experiment's SR treatment is passed explicitly (belt) even though the
# best_params overlay records it too (braces) — drift is impossible.
MODEL_ARGS=(--model.upsampler "$UPSAMPLER" --model.freeze_sr "$FREEZE_SR"
            --model.sr_pad "$SR_PAD" --model.sen2sr_dir "$SEN2SR_DIR")

echo "=== REFIT (best config, ${REFIT_EPOCHS} epochs, ${REFIT_GPUS} GPU) ==="
python -m sr.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  "${MODEL_ARGS[@]}" \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED"

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
python -m sr.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  "${MODEL_ARGS[@]}" \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
