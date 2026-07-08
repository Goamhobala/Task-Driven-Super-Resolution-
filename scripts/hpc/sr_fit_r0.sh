#!/bin/bash
# R0 refit: resume from an sr_tune_r0.sh search, refit the best bicubic config
# at full length, then benchmark on the test split. Same as sr_fit_only.sh but
# for the parameter-free R0 baseline — so it needs NO SEN2SR weights and forces
# the bicubic upsampler regardless of what the base config defaults to.
#
# Submit on a single-GPU allocation:
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr_fit_r0.sh STUDY_TAG=bicubic
set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"

SEED="${SEED:-0}"
MASK_SOURCE="${MASK_SOURCE:-graph}"      # must match the search's label source
STUDY_TAG="${STUDY_TAG:-bicubic}"        # must match sr_tune_r0.sh's STUDY_TAG
RUN_DIR="${RUN_DIR:-/scratch/${USER_NAME}/InstaRoad/runs/sr_optuna_${STUDY_TAG}_seed${SEED}}"  # holds best_params.yaml
NUM_WORKERS="${NUM_WORKERS:-0}"
PRECISION="${PRECISION:-bf16-mixed}"
REFIT_EPOCHS="${REFIT_EPOCHS:-50}"
REFIT_GPUS="${REFIT_GPUS:-1}"        # 1 = single GPU (no DDP).
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
export PYTHONUNBUFFERED=1
echo "python=$(which python)"

if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run the search first (sr_tune_r0.sh)." >&2
  exit 1
fi
# (No SEN2SR weight check: R0 is parameter-free bicubic.)
echo "--- best hyperparameters ---"; cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

echo "=== REFIT — R0 bicubic (best config, ${REFIT_EPOCHS} epochs, ${REFIT_GPUS} GPU) ==="
python -m sr.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --model.upsampler bicubic \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED"

# Log the test metrics to the SAME wandb run the refit just created.
if LATEST_RUN=$(readlink -f "$RUN_DIR/wandb/latest-run" 2>/dev/null) && [ -n "$LATEST_RUN" ]; then
  export WANDB_RUN_ID="${LATEST_RUN##*-}"
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
  --model.upsampler bicubic \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
