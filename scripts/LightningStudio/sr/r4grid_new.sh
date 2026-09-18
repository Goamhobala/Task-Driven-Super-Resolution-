#!/bin/bash
# R4GRID — the SR4RS lr_sr x hard-constraint grid cell. Lightning Studio twin of
# scripts/hpc/sr/r4grid_new.sh. The rationale for every setting lives in the HPC
# twin; this file must stay setting-for-setting identical to it, with only the
# /scratch paths swapped for $INSTAROAD_ROOT (the same substitution
# sync_engine.py applies to the engine).
#
#   S=sr/r4grid_new.sh; C="HC=on LRSR=1e-5"
#   bash scripts/LightningStudio/run.sh $S STAGE=fit   $C
#   bash scripts/LightningStudio/run.sh $S STAGE=bench $C
#
# Resuming a partial cell (checkpoint normalisation, planted overlay, auto-resume
# after an interruptible preemption): scripts/LightningStudio/sr/grid/resume_r4grid.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

# --- the two cell coordinates ------------------------------------------------
HC="${HC:?set HC=on|off — the hard-constraint lane (r2a lane vs r2b lane)}"
LRSR="${LRSR:?set LRSR=<lr_sr> — the pinned adaptation rate, e.g. 1e-4}"

case "$HC" in
on | off) : ;;
*)
  echo "ERROR: HC must be on|off, got '${HC}'." >&2
  exit 2
  ;;
esac

LS_TAG=$(awk -v v="$LRSR" 'BEGIN {
  if (v + 0 != v || v <= 0) exit 1
  split(sprintf("%.1e", v), a, "e")
  m = a[1]; sub(/\.0$/, "", m)
  printf "%se%d", m, a[2] + 0
}' </dev/null) || {
  echo "ERROR: LRSR must be a positive number, got '${LRSR}'." >&2
  exit 2
}

EXP_TAG="r4grid_${HC}_ls${LS_TAG}"

# --- the lane: HC on = r2a's configuration, HC off = r2b's -------------------
if [ "$HC" = "on" ]; then
  SR_HC="on"
  SR_PAD=8
  export HC_MASK_PATH="${HC_MASK_PATH:-${INSTAROAD_ROOT}/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"
else
  SR_HC="off"
  SR_PAD=0
fi

# --- everything else = the formal r2a/r2b configuration ----------------------
LABELS="new"
UPSAMPLER="sr4rs"
export SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SR4RS_RGBN}"
FREEZE_SR="false"
REG="${REG:-true}"
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

# --- the pinned axes ---------------------------------------------------------
export LR_SR_MIN="$LRSR"
export LR_SR_MAX="$LRSR"
export LR_MIN="${LR_MIN:-2e-4}"
export LR_MAX="${LR_MAX:-2e-4}"
N_TRIALS="${N_TRIALS:-1}"
TUNE_EPOCHS="${TUNE_EPOCHS:-1}"
PATIENCE="${PATIENCE:-5}"
SEARCH_GPUS="${SEARCH_GPUS:-1}"
BATCH_SIZES="${BATCH_SIZES:-4}" # between-arm constant of the whole SR series

# --- fit: the pre-registered budget, ONE seed, no replicates -----------------
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"

# --- band-guard policy: LOOSENED RAILS, NOT DISABLED -------------------------
export STD_BAND_RAISE_LO="${STD_BAND_RAISE_LO:-0.01}"
export STD_BAND_RAISE_HI="${STD_BAND_RAISE_HI:-100}"
export STD_BAND_ACTION="${STD_BAND_ACTION:-warn}"

# --- the loss: a frozen control, copied from the formal arm ------------------
LOSS_ARM="${LOSS_ARM:-gap_ce}"
export PSTAR="${PSTAR:-gap_ce}"
export GAP_R="${GAP_R:-4}"
export GAP_K="${GAP_K:-60.0}"
export TL_ELL="${TL_ELL:-5}"
export TL_THETA="${TL_THETA:-0.375}"
export GAP_THETA="${GAP_THETA:-0.55836}"
export MIX_W="${MIX_W:-0.6075946831862098}"
export TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
export CL_ALPHA="${CL_ALPHA:-0.3}"
export CL_ITERS="${CL_ITERS:-5}"
export SKEL_W="${SKEL_W:-1.0}"
export SKEL_RADIUS="${SKEL_RADIUS:-1}"
export WARMUP_START="${WARMUP_START:-30}"
export WARMUP_RAMP="${WARMUP_RAMP:-10}"
export SEARCH_THETAS="${SEARCH_THETAS:-false}"
export SEARCH_MIX_W="${SEARCH_MIX_W:-false}"
export POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-4.77222}"
export POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-4.77222}"

# --- protocol: the final one, unchanged --------------------------------------
export TRAIN_SPLITS="${TRAIN_SPLITS:-train val}"
export MONITOR="${MONITOR:-val_ap}"
export ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
export ADAPTIVE_NORM_M="${ADAPTIVE_NORM_M:-0.01}"
export NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"
export BENCH_SPLIT="${BENCH_SPLIT:-test}"
export SWEEP_CRITERION="${SWEEP_CRITERION:-iou}"
export AP_BINS="${AP_BINS:-101}"
export TILE_METRICS="${TILE_METRICS:-apls,cldice}"

# --- wandb: one project for the whole grid -----------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_lrsr_grid}"
export WANDB_NAME="${WANDB_NAME:-${EXP_TAG}_seed${SEED:-0}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-r4grid_${HC}}"

echo "=== SR4RS lr_sr GRID CELL: HC=${HC} (sr_hc=${SR_HC}, sr_pad=${SR_PAD})  lr_sr=${LRSR} (pinned) ==="
echo "    exp_tag=${EXP_TAG}  rails=[${STD_BAND_RAISE_LO}x, ${STD_BAND_RAISE_HI}x]  lr=2e-4 pinned  descriptive-only"

source "$LS_DIR/sr/_stages_tv.sh"
