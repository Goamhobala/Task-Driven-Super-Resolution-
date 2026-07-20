#!/bin/bash
# R1a — FROZEN SEN2SR preprocessing WITH reflect-padding (8 px), ROSA_all.
# lr_sr is auto-skipped (frozen SR). Stage 1 of the staged SEN2SR protocol:
# r7a_all.sh warm-starts its UNet from this arm's fitted ckpt.
# vs r1b: padding on/off re-establishes the FFT border effect post units-fix.
#
#   bash scripts/hpc/submit.sh sr/r1a_all.sh STAGE=tune [SEED=n]
#   bash scripts/hpc/submit.sh sr/r1a_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1a_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=8

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
