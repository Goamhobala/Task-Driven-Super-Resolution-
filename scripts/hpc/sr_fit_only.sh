#!/bin/bash
# Stage 2 of 2: resume from an existing sr.tune search — refit the best joint
# SR + UNet config at full length, then benchmark on the test split. Mirrors
# fit_only.sh; reuses the best_params.yaml sr.tune wrote into RUN_DIR.
#
# Submit on a single-GPU allocation (the search stage owns the 2-GPU pattern):
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr_fit_only.sh STUDY_TAG=graph
set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
# Every knob honours an environment override (VAR=... bash sr_fit_only.sh),
# so train.sbatch can drive it with KEY=VALUE tokens.
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN}"

SEED="${SEED:-0}"
MASK_SOURCE="${MASK_SOURCE:-graph}"      # must match the search's label source
STUDY_TAG="${STUDY_TAG:-${MASK_SOURCE}}" # must match sr_tune_only.sh's STUDY_TAG
RUN_DIR="${RUN_DIR:-/scratch/${USER_NAME}/InstaRoad/runs/sr_optuna_${STUDY_TAG}_seed${SEED}}"  # holds best_params.yaml
NUM_WORKERS="${NUM_WORKERS:-0}"
PRECISION="${PRECISION:-bf16-mixed}"
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"        # 1 = single GPU (no DDP). See notes in train_unet_optuna.sh.
WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_joint}"
# ===========================================================================

BASE_CONFIG="$REPO_DIR/src/sr/configs/joint_sr.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"
BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_jointsr_best.ckpt"

# Use the scratch venv + put src/ on PYTHONPATH so `sr` / `unet` resolve.
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1             # flush stdout live -> the log fills as it runs
echo "python=$(which python)"

if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run the search first (sr_tune_only.sh)." >&2
  exit 1
fi
if [ ! -f "${SEN2SR_DIR}/model.safetensor" ]; then
  echo "ERROR: SEN2SR weights not at ${SEN2SR_DIR} — prefetch on a login node." >&2
  exit 1
fi
echo "--- best hyperparameters ---"; cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

echo "=== REFIT (best config, ${REFIT_EPOCHS} epochs, ${REFIT_GPUS} GPU) ==="
python -m sr.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  --model.sen2sr_dir "$SEN2SR_DIR" \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED"

# Log the test metrics to the SAME wandb run the refit just created (see
# fit_only.sh for the rationale: without this, test_iou/f1 only hit stdout).
if LATEST_RUN=$(readlink -f "$RUN_DIR/wandb/latest-run" 2>/dev/null) && [ -n "$LATEST_RUN" ]; then
  export WANDB_RUN_ID="${LATEST_RUN##*-}"   # .../run-<timestamp>-<id> -> <id>
  export WANDB_RESUME=must
  echo "resuming wandb run ${WANDB_RUN_ID} for the test split"
else
  echo "WARN: could not locate the refit's wandb run; test will log to a fresh run" >&2
fi

echo "=== BENCHMARK (test split) ==="
python -m sr.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  --model.sen2sr_dir "$SEN2SR_DIR" \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
