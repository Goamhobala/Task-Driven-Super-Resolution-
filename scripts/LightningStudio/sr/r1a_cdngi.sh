#!/bin/bash
# R1a — frozen SEN2SR preprocessing WITH reflect-padding (8 px), CDNGI labels.
# vs r1b: same frozen SR, padding on/off isolates the FFT border artifact.
# Frozen SR: lr_sr is not searched.
#
#   bash scripts/LightningStudio/run.sh sr/r1a_cdngi.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r1a_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1a_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=8

source "$LS_DIR/sr/_stages.sh"
