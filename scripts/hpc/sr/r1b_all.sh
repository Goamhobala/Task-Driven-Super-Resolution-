#!/bin/bash
# R1b — FROZEN SEN2SR preprocessing WITHOUT padding, ROSA_all.
# lr_sr is auto-skipped (frozen SR). Stage 1 for r7b_all.sh.
#
#   bash scripts/hpc/submit.sh sr/r1b_all.sh STAGE=tune [SEED=n]
#   bash scripts/hpc/submit.sh sr/r1b_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1b_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
