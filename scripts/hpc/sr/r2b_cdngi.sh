#!/bin/bash
# R2b — joint task-driven SEN2SR fine-tuning WITHOUT padding, CDNGI labels.
# The tune stage searches the joint LR pair (lr, lr_sr).
# vs r2a: padding on/off separates "fine-tuning fixes the border" from
# genuine task-driven adaptation.
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r2b_cdngi.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r2b_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2b_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
