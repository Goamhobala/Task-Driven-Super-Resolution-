#!/bin/bash
# Resume from an existing Optuna search: refit the best config at full length,
# then benchmark on the test split. Skips the 2-hour search entirely -- it just
# reuses best_params.yaml that unet.tune already wrote into RUN_DIR.
#
# Run on a worker node with a GPU allocated (salloc --gres=gpu:1), then:
#   bash scripts/fit_only.sh
set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="/scratch/${USER_NAME}/InstaRoad/.venv"
DATASET_DIR="/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI"

SEED=0
STUDY_TAG="imagenet"   # must match tune_only.sh's STUDY_TAG (e.g. imagenet | random)
RUN_DIR="/scratch/${USER_NAME}/InstaRoad/runs/unet_optuna_${STUDY_TAG}_seed${SEED}"  # holds best_params.yaml
NUM_WORKERS=0
PRECISION="bf16-mixed"
REFIT_EPOCHS=50
REFIT_GPUS=1          # 1 = single GPU (no DDP). See notes in train_unet_optuna.sh.
WANDB_PROJECT="unet_s2rosa_baseline"
# ===========================================================================

BASE_CONFIG="$REPO_DIR/src/unet/configs/unet.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"
BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_best.ckpt"

# --- THE PART THAT WAS MISSING: use the scratch venv + put src/ on PYTHONPATH so
#     `python -m unet.cli` resolves (unet is a source package, not pip-installed).
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
echo "python=$(which python)"        # sanity: should be under $VENV_DIR, not miniconda

if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run the search first (train_unet_optuna.sh)." >&2
  exit 1
fi
echo "--- best hyperparameters ---"; cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

echo "=== REFIT (best config, ${REFIT_EPOCHS} epochs, ${REFIT_GPUS} GPU) ==="
python -m unet.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED"

echo "=== BENCHMARK (test split) ==="
python -m unet.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --trainer.devices 1 \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
