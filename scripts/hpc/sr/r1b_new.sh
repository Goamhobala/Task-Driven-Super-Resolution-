#!/bin/bash
# R1b — FROZEN SEN2SR preprocessing WITHOUT padding, ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> test).
# lr_sr is auto-skipped (frozen SR). Stage 1 for r7b_new.sh.
#
#   bash scripts/hpc/submit.sh sr/r1b_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/r1b_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/r1b_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1b_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
