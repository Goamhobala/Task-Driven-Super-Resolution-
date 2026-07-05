#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="sr-train"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out
#SBATCH --mem-per-cpu=8G

# Resolution-enhancement experiments (RQ A2): R0 bicubic / R1 frozen SEN2SR /
# R2 joint task-driven SEN2SR + U-Net. One trainer covers all three; pick via
# EXPERIMENT below.
#
# One-time setup (login node — compute nodes have no internet):
#   uv pip install -e "$REPO_DIR[sr]"                       # into the scratch venv
#   python -m sr.train --data "$DATA_DIR" --experiment R1 \
#       --out /tmp/dl --download-sen2sr --epochs 0          # or any way to prefetch
#   (downloads SEN2SR weights to $DATA_DIR/models/SEN2SRLite_RGBN)
#
# Usage:  sbatch scripts/train_sr.sh
# Edit the CONFIG block below; nothing else should need touching.

set -euo pipefail

# ============================ CONFIG — EDIT HERE ============================
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
DATA_DIR="/scratch/${USER_NAME}/InstaRoad"     # holds imagery/, mask_2pt5m/, Data.npz
VENV_DIR="${DATA_DIR}/.venv"

IMAGERY_DIR="${DATA_DIR}/imagery"
MASK_HR_DIR="${DATA_DIR}/mask_2pt5m"           # 2.5 m masks (exactly 4x imagery dims)
STATS_NPZ="${DATA_DIR}/Data.npz"
SEN2SR_DIR="${DATA_DIR}/models/SEN2SRLite_RGBN"

EXPERIMENT="R2"       # R0=bicubic  R1=SEN2SR frozen  R2=SEN2SR joint (task-driven)
EPOCHS=50
BATCH_SIZE=4          # 512x512 U-Net stage; 4 fits comfortably on an L40S at bf16
LR_SEG=0.001          # U-Net LR (baseline default)
LR_SR=0.00001         # SEN2SR LR; alpha = LR_SR/LR_SEG — keep well below LR_SEG
FREEZE_SR_STEPS=500   # hold SEN2SR at lr=0 early to protect pretrained weights
SR_LR_RAMP_STEPS=500  # then ramp linearly to LR_SR
SCHEDULER="none"      # none | cosine
PATCH_SIZE=128        # pinned to 128 by SEN2SR's shipped FFT mask
ENCODER="resnet34"    # keep constant across R0/R1/R2 — the ablation's control
SEED=0
NUM_WORKERS=0         # 0 = load in main process; GDAL/rasterio segfault in forked subprocesses
PRECISION="bf16-mixed"

WANDB_PROJECT="instaroad-baseline"
# export WANDB_API_KEY=...   # set in your shell / ~/.bashrc before sbatch for online logging
WANDB_MODE="online"
# ===========================================================================

RUN_DIR="$DATA_DIR/runs/sr_${EXPERIMENT}_seed${SEED}"
mkdir -p "$RUN_DIR"

# Fail fast, in the job log, if the data isn't visible from this node.
echo "host=$(hostname)  USER_NAME=${USER_NAME}  REPO_DIR=${REPO_DIR}"
echo "DATA_DIR=${DATA_DIR}  imagery=${IMAGERY_DIR}  masks_hr=${MASK_HR_DIR}"
for d in "$IMAGERY_DIR" "$MASK_HR_DIR"; do
  if [ ! -d "$d" ]; then
    echo "ERROR: $d not visible on $(hostname). Is /scratch mounted on this node?" >&2
    exit 1
  fi
done
n_tif=$(find "${IMAGERY_DIR}" -maxdepth 1 -name '*.tif' | wc -l)
n_msk=$(find "${MASK_HR_DIR}" -maxdepth 1 -name '*.tif' | wc -l)
echo "  imagery .tif count: ${n_tif}   2.5m mask .tif count: ${n_msk}"
if [ "${n_tif}" -eq 0 ] || [ "${n_msk}" -eq 0 ]; then
  echo "ERROR: missing COGs or 2.5m masks." >&2
  exit 1
fi
if [ "${EXPERIMENT}" != "R0" ] && [ ! -f "${SEN2SR_DIR}/model.safetensor" ]; then
  echo "ERROR: SEN2SR weights not at ${SEN2SR_DIR} — prefetch on a login node (see header)." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "=== TRAIN (experiment=$EXPERIMENT) ==="
python -m sr.train \
  --data "$DATA_DIR" \
  --imagery "$IMAGERY_DIR" \
  --masks-hr "$MASK_HR_DIR" \
  --stats "$STATS_NPZ" \
  --sen2sr-dir "$SEN2SR_DIR" \
  --experiment "$EXPERIMENT" \
  --out "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr-seg "$LR_SEG" \
  --lr-sr "$LR_SR" \
  --freeze-sr-steps "$FREEZE_SR_STEPS" \
  --sr-lr-ramp-steps "$SR_LR_RAMP_STEPS" \
  --scheduler "$SCHEDULER" \
  --patch-size "$PATCH_SIZE" \
  --encoder "$ENCODER" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --precision "$PRECISION" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE"

echo "=== DONE ===  outputs in $RUN_DIR"
