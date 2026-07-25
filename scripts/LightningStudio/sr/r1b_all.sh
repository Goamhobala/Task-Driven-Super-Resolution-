#!/bin/bash
# R1b — FROZEN SEN2SR preprocessing WITHOUT padding, ROSA_all.
# lr_sr is auto-skipped (frozen SR). Stage 1 for r7b_all.sh.
#
#   bash scripts/LightningStudio/run.sh sr/r1b_all.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r1b_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1b_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=0

source "$LS_DIR/sr/_stages.sh"
