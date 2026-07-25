#!/bin/bash
# R0 — bicubic x4 upsampling (deterministic baseline), CDNGI labels.
# No SR params: lr_sr is not searched; SR_PAD is irrelevant (no FFT constraint).
#
#   bash scripts/LightningStudio/run.sh sr/r0_cdngi.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r0_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r0_cdngi"
LABELS="cdngi"
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

source "$LS_DIR/sr/_stages.sh"
