#!/bin/bash
# R1a — FROZEN SEN2SR preprocessing WITH reflect-padding (8 px), ROSA_all.
# lr_sr is auto-skipped (frozen SR). Stage 1 of the staged SEN2SR protocol:
# r7a_all.sh warm-starts its UNet from this arm's fitted ckpt.
# vs r1b: padding on/off re-establishes the FFT border effect post units-fix.
#
#   bash scripts/LightningStudio/run.sh sr/r1a_all.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r1a_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1a_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=8

source "$LS_DIR/sr/_stages.sh"
