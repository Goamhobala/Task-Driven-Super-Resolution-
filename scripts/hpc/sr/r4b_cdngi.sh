#!/bin/bash
# R4b — joint task-driven fine-tuning of SR4RS (WGAN-GP-trained generator,
# PyTorch port) WITHOUT padding, CDNGI labels. See r4a_cdngi.sh.
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r4b_cdngi.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r4b_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r4b_cdngi"
LABELS="cdngi"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-$HOME/InstaRoad/InstaRoadPrototype/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-2 4 8}"

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
