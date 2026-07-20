#!/bin/bash
# R2b — COLD joint task-driven SEN2SR fine-tuning WITHOUT padding, ROSA_all.
# vs r2a: padding on/off separates "fine-tuning fixes the border" from genuine
# task-driven adaptation.
#
#   bash scripts/hpc/submit.sh sr/r2b_all.sh STAGE=tune [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2b_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2b_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
