#!/bin/bash
# R1b — frozen SEN2SR preprocessing WITHOUT padding, CDNGI labels.
# vs r1a: same frozen SR, padding on/off isolates the FFT border artifact.
# Frozen SR: lr_sr is not searched.
#
#   bash scripts/LightningStudio/run.sh sr/r1b_cdngi.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r1b_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1b_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=0

source "$LS_DIR/sr/_stages.sh"
