#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="unet-optuna"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out
#SBATCH --mem-per-cpu=8G

# End-to-end UNet finetuning: Optuna searches the hyperparameters, then the best
# config is refit at full length and benchmarked on the held-out test tiles.
# Adapted from scripts/train_baseline.sh -- same conventions, but drives the
# config-driven `unet` module (LightningCLI) instead of baseline.train/benchmark.
#
#   1. python -m unet.tune  -> best_params.yaml  (Optuna, short per-trial budget)
#   2. python -m unet.cli fit  --config <base> --config best_params.yaml  (full refit)
#   3. python -m unet.cli test --config <base> --config best_params.yaml  (per-crop IoU/F1)
#
# Usage:  sbatch scripts/train_unet_optuna.sh
# Edit the CONFIG block below; nothing else should need touching.

set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
# $USER can be empty in a batch shell; fall back to whoami so DATA_DIR never
# collapses to /scratch//InstaRoad (a real cause of "found 0 tiles" under sbatch).
USER_NAME="${USER:-$(whoami)}"

# Repo root. Set explicitly: under `sbatch`, SLURM copies this script to a spool
# dir, so deriving the path from $BASH_SOURCE points at the copy, not the repo.
# Override at submit time with:  REPO_DIR=/path sbatch scripts/train_unet_optuna.sh
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="/scratch/${USER_NAME}/InstaRoad/.venv"     # prebuilt venv on scratch
# One-time, into that venv:  uv pip install -e "$REPO_DIR[unet]"

# S2ROSA dataset root (holds the train/val/test tile splits the RoadDataModule reads).
DATASET_DIR="/scratch/${USER_NAME}/InstaRoad/S2ROSA_V2"

# Canonical Lightning config(s). unet.yaml is the base; norm_stats.yaml layers the
# frozen per-band train stats over it (data.norm_mean/std). Order matters (last wins).
BASE_CONFIG="$REPO_DIR/src/unet/configs/unet.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"   # logger overlay -> Weights & Biases

SEED=0
NUM_WORKERS=1        # 0 = load in main process; GDAL/rasterio segfault in subprocesses
PRECISION="bf16-mixed"

# --- Optuna search budget ---------------------------------------------------
N_TRIALS=30
TUNE_EPOCHS=8         # short per-trial budget; the winner is refit at full length
PATIENCE=3           # per-trial EarlyStopping on val_iou (0 = off)
# Search space (see unet.tune for defaults; override here if desired)
LR_MIN=1e-5
LR_MAX=1e-2
POS_WEIGHT_MIN=1.0
POS_WEIGHT_MAX=15.0
ENCODERS="resnet18 resnet34 resnet50"
BATCH_SIZES="8 16 32"

# --- Full refit budget (after the search picks the winner) ------------------
REFIT_EPOCHS=50

WANDB_PROJECT="unet_s2rosa_baseline"
# export WANDB_API_KEY=...   # set in your shell / ~/.bashrc before sbatch for online logging
# ===========================================================================

RUN_DIR="/scratch/${USER_NAME}/InstaRoad/runs/unet_optuna_seed${SEED}"
CKPT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

# --- Diagnose the environment IN THE JOB LOG, then fail fast with a clear
#     message if the data isn't visible from this node (instead of dying later
#     as a cryptic num_samples=0 inside the DataLoader). ----------------------
echo "host=$(hostname)  USER_NAME=${USER_NAME}  REPO_DIR=${REPO_DIR}"
echo "DATASET_DIR=${DATASET_DIR}"
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted on this node?" >&2
  exit 1
fi
n_tif=$(find "${DATASET_DIR}" -name '*.tif' 2>/dev/null | head -n 1000 | wc -l)
echo "  .tif count (capped at 1000): ${n_tif}"
if [ "${n_tif}" -eq 0 ]; then
  echo "ERROR: no .tif tiles under ${DATASET_DIR}." >&2
  exit 1
fi
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: ${NORM_CONFIG} missing — generate it with:" >&2
  echo "  python -m sentinel2data.cli norm-stats --dataset-dir ${DATASET_DIR} --out ${NORM_CONFIG}" >&2
  exit 1
fi

# Use the scratch venv directly; src/ on PYTHONPATH so the top-level packages import.
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "=== OPTUNA SEARCH (n_trials=$N_TRIALS, ${TUNE_EPOCHS} epochs/trial) ==="
python -m unet.tune \
  --base-config "$BASE_CONFIG" \
  --base-config "$NORM_CONFIG" \
  --dataset-dir "$DATASET_DIR" \
  --out "$RUN_DIR" \
  --num-workers "$NUM_WORKERS" \
  --n-trials "$N_TRIALS" \
  --max-epochs "$TUNE_EPOCHS" \
  --patience "$PATIENCE" \
  --precision "$PRECISION" \
  --seed "$SEED" \
  --study-name "unet_optuna_seed${SEED}" \
  --storage "sqlite:///${RUN_DIR}/study.db" \
  --lr-min "$LR_MIN" --lr-max "$LR_MAX" \
  --pos-weight-min "$POS_WEIGHT_MIN" --pos-weight-max "$POS_WEIGHT_MAX" \
  --encoders $ENCODERS \
  --batch-sizes $BATCH_SIZES

BEST_CONFIG="${RUN_DIR}/best_params.yaml"
if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: Optuna did not write ${BEST_CONFIG}." >&2
  exit 1
fi
echo "--- best hyperparameters ---"; cat "$BEST_CONFIG"

# Refit + test from inside RUN_DIR so the base config's relative `checkpoints/`
# dirpath lands under the run dir (keeps configs absolute, so cd is safe).
cd "$RUN_DIR"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_best.ckpt"

echo "=== REFIT (best config, ${REFIT_EPOCHS} epochs) ==="
python -m unet.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --trainer.max_epochs "$REFIT_EPOCHS" \
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
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
