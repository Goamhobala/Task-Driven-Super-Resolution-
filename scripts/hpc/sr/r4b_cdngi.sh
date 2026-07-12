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
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"  # 8 OOMs on 44GB: SR4RS runs 256-ch convs
                                     # (incl. a 9x9) at the full 512px grid

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
