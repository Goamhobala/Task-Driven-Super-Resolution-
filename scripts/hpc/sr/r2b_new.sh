#!/bin/bash
# R2b — COLD joint task-driven SEN2SR fine-tuning WITHOUT padding, ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> test).
# vs r2a: padding on/off separates "fine-tuning fixes the border" from genuine
# task-driven adaptation.
#
#   bash scripts/hpc/submit.sh sr/r2b_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2b_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2b_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2b_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=0

REG="${REG:-true}"
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
