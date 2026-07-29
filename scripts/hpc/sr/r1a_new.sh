#!/bin/bash
# R1a — FROZEN SEN2SR preprocessing WITH reflect-padding (8 px), ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> test).
# lr_sr is auto-skipped (frozen SR). Stage 1 of the staged SEN2SR protocol:
# r7a_new.sh warm-starts its UNet from THIS arm's final ckpt.
# vs r1b: padding on/off re-establishes the FFT border effect.
#
#   bash scripts/hpc/submit.sh sr/r1a_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/r1a_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/r1a_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1a_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=8

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
