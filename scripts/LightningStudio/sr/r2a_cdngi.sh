#!/bin/bash
# R2a — joint task-driven SEN2SR fine-tuning WITH reflect-padding (8 px),
# CDNGI labels. The tune stage searches the joint LR pair (lr, lr_sr).
# vs r2b: padding on/off separates "fine-tuning fixes the border" from
# genuine task-driven adaptation.
#
#   bash scripts/LightningStudio/run.sh sr/r2a_cdngi.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r2a_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2a_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

source "$LS_DIR/sr/_stages.sh"
