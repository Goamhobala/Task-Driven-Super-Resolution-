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
# Repo root. Set explicitly: under `sbatch`, SLURM copies this script to a spool
# dir, so deriving the path from $BASH_SOURCE points at the copy, not the repo.
# Override at submit time with:  REPO_DIR=/path sbatch scripts/train_baseline.sh
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
DATA_DIR="/scratch/$USER/InstaRoad"                 # holds imagery/, mask_10m/, Data.npz
VENV_DIR="/scratch/$USER/InstaRoad/.venv"           # prebuilt venv on scratch
# One-time, into that venv:  uv pip install -e "$REPO_DIR[baseline]"

CHANNEL_CONFIG="M3"   # M0=RGB+NIR(4)  M1=+S2 20m(10)  M2=RGB+NIR+S1(8)  M3=all 14 bands
EPOCHS=50
BATCH_SIZE=8
LR=0.001
PATCH_SIZE=256
SEED=0
NUM_WORKERS=8

WANDB_PROJECT="instaroad-baseline"
# export WANDB_API_KEY=...   # set in your shell / ~/.bashrc before sbatch for online logging
WANDB_MODE="online"
# ===========================================================================

RUN_DIR="$DATA_DIR/runs/baseline_${CHANNEL_CONFIG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# Use the scratch venv directly; src/ on PYTHONPATH so the top-level packages import.
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "=== TRAIN (config=$CHANNEL_CONFIG) ==="
python -m baseline.train \
  --data "$DATA_DIR" \
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
  --out "$RUN_DIR" \
  --num-workers "$NUM_WORKERS"

echo "=== DONE ===  outputs in $RUN_DIR"
