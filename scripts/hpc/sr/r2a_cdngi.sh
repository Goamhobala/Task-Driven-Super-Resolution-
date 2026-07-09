#!/bin/bash
# R2a — joint task-driven SEN2SR fine-tuning WITH reflect-padding (8 px),
# CDNGI labels. The tune stage searches the joint LR pair (lr, lr_sr).
# vs r2b: padding on/off separates "fine-tuning fixes the border" from
# genuine task-driven adaptation.
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r2a_cdngi.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r2a_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2a_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
