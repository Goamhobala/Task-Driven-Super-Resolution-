#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="ssmt-train"
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out
#SBATCH --mem-per-cpu=8G

# End-to-end baseline: train the UNet++ road model, then benchmark the best
# checkpoint on the held-out test sites (writes the per-chip metrics parquet).
#
# Usage:  sbatch scripts/train_baseline.sh
# Edit the CONFIG block below; nothing else should need touching.

set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
# $USER can be empty in a batch shell; fall back to whoami so DATA_DIR never
# collapses to /scratch//InstaRoad (a real cause of "found 0 tiles" under sbatch).
USER_NAME="${USER:-$(whoami)}"

# Repo root. Set explicitly: under `sbatch`, SLURM copies this script to a spool
# dir, so deriving the path from $BASH_SOURCE points at the copy, not the repo.
# Override at submit time with:  REPO_DIR=/path sbatch scripts/train_baseline.sh
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
DATA_DIR="/scratch/${USER_NAME}/InstaRoad"          # holds imagery/, mask_10m/, Data.npz
VENV_DIR="${DATA_DIR}/.venv"                         # prebuilt venv on scratch
# One-time, into that venv:  uv pip install -e "$REPO_DIR[baseline]"

# Data sub-paths passed EXPLICITLY to the trainer (don't rely on its defaults).
IMAGERY_DIR="${DATA_DIR}/imagery"
MASK_DIR="${DATA_DIR}/mask_10m"
STATS_NPZ="${DATA_DIR}/Data.npz"

CHANNEL_CONFIG="M3"   # M0=RGB+NIR(4)  M1=+S2 20m(10)  M2=RGB+NIR+S1(8)  M3=all 14 bands
EPOCHS=50
BATCH_SIZE=8
LR=0.001
PATCH_SIZE=256
SEED=0
NUM_WORKERS=0         # 0 = load in main process; GDAL/rasterio segfault in subprocesses

WANDB_PROJECT="instaroad-baseline"
# export WANDB_API_KEY=...   # set in your shell / ~/.bashrc before sbatch for online logging
WANDB_MODE="online"
# ===========================================================================

RUN_DIR="$DATA_DIR/runs/baseline_${CHANNEL_CONFIG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# --- Diagnose the environment IN THE JOB LOG, then fail fast with a clear
#     message if the data isn't visible from this node (instead of dying later
#     as a cryptic num_samples=0 inside the DataLoader). ----------------------
echo "host=$(hostname)  USER_NAME=${USER_NAME}  REPO_DIR=${REPO_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "imagery=${IMAGERY_DIR}"
if [ ! -d "${IMAGERY_DIR}" ]; then
  echo "ERROR: ${IMAGERY_DIR} not visible on $(hostname). Is /scratch mounted on this node?" >&2
  exit 1
fi
n_tif=$(find "${IMAGERY_DIR}" -maxdepth 1 -name '*.tif' | wc -l)
echo "  .tif count: ${n_tif}"
if [ "${n_tif}" -eq 0 ]; then
  echo "ERROR: no .tif COGs under ${IMAGERY_DIR}." >&2
  exit 1
fi

# Use the scratch venv directly; src/ on PYTHONPATH so the top-level packages import.
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "=== TRAIN (config=$CHANNEL_CONFIG) ==="
python -m baseline.train \
  --data "$DATA_DIR" \
  --imagery "$IMAGERY_DIR" \
  --masks "$MASK_DIR" \
  --stats "$STATS_NPZ" \
  --config "$CHANNEL_CONFIG" \
  --out "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch-size "$PATCH_SIZE" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE"

echo "=== BENCHMARK (test split) ==="
python -m baseline.benchmark \
  --ckpt "$RUN_DIR/best.pth" \
  --data "$DATA_DIR" \
  --imagery "$IMAGERY_DIR" \
  --masks "$MASK_DIR" \
  --stats "$STATS_NPZ" \
  --out "$RUN_DIR" \
  --num-workers "$NUM_WORKERS"

echo "=== DONE ===  outputs in $RUN_DIR"
