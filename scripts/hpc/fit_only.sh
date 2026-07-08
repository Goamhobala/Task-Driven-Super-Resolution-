#!/bin/bash
# Resume from an existing Optuna search: refit the best config at full length,
# then benchmark on the test split. Skips the 2-hour search entirely -- it just
# reuses best_params.yaml that unet.tune already wrote into RUN_DIR.
#
# Run on a worker node with a GPU allocated (salloc --gres=gpu:1), then:
#   bash scripts/fit_only.sh
set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
# Every knob honours an environment override (VAR=... bash fit_only.sh), so the
# train.sbatch wrapper can drive it with `KEY=VALUE` tokens.
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"

SEED="${SEED:-0}"
STUDY_TAG="${STUDY_TAG:-imagenet}"   # must match tune_only.sh's STUDY_TAG (e.g. imagenet | random | osm)
RUN_DIR="${RUN_DIR:-/scratch/${USER_NAME}/InstaRoad/runs/unet_optuna_${STUDY_TAG}_seed${SEED}}"  # holds best_params.yaml
NUM_WORKERS="${NUM_WORKERS:-0}"
PRECISION="${PRECISION:-bf16-mixed}"
REFIT_EPOCHS="${REFIT_EPOCHS:-50}"
REFIT_GPUS="${REFIT_GPUS:-1}"        # 1 = single GPU (no DDP). See notes in train_unet_optuna.sh.
WANDB_PROJECT="${WANDB_PROJECT:-unet_s2rosa_baseline}"

# Label source. Leave EMPTY to inherit whatever the search used -- unet.tune now
# pins data.mask_dirname into best_params.yaml, so the refit reproduces it
# automatically. Only set this to OVERRIDE the overlay (e.g. cross-evaluate on a
# different label set); it must name a mask dir beside imagery, e.g. mask_osm_10.
MASK_DIRNAME="${MASK_DIRNAME:-}"
# ===========================================================================

# Pass --data.mask_dirname ONLY when set: an empty value would misresolve to
# <split>/<tile>.tif under LightningCLI. Empty -> keep the CSVs' CDNGI masks.
MASK_ARGS=()
[ -n "$MASK_DIRNAME" ] && MASK_ARGS=(--data.mask_dirname "$MASK_DIRNAME")

BASE_CONFIG="$REPO_DIR/src/unet/configs/unet.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"
BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_best.ckpt"

# --- THE PART THAT WAS MISSING: use the scratch venv + put src/ on PYTHONPATH so
#     `python -m unet.cli` resolves (unet is a source package, not pip-installed).
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1             # flush stdout live -> the log fills as it runs
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
  ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED"

# Log the test metrics to the SAME wandb run the refit just created. The base
# config uses the default (TensorBoard) logger, so without the wandb overlay the
# `test` subcommand ran logger-less and test_iou/test_f1 reached only stdout ->
# the slurm log. wandb reads the run id + resume mode from the environment, so no
# extra jsonargparse plumbing on the WandbLogger is needed.
if LATEST_RUN=$(readlink -f "$RUN_DIR/wandb/latest-run" 2>/dev/null) && [ -n "$LATEST_RUN" ]; then
  export WANDB_RUN_ID="${LATEST_RUN##*-}"   # .../run-<timestamp>-<id> -> <id>
  export WANDB_RESUME=must
  echo "resuming wandb run ${WANDB_RUN_ID} for the test split"
else
  echo "WARN: could not locate the refit's wandb run; test will log to a fresh run" >&2
fi

echo "=== BENCHMARK (test split) ==="
python -m unet.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
