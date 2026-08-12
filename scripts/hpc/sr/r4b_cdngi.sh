#!/bin/bash
# R4b — joint task-driven fine-tuning of SR4RS (WGAN-GP-trained generator,
# PyTorch port) WITHOUT padding, CDNGI labels. See r4a_cdngi.sh.
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r4b_cdngi.sh STAGE=tune [SEED=n] [LOSS_ARM=arm]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r4b_cdngi.sh STAGE=fit [SEED=n] [LOSS_ARM=arm]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r4b_cdngi"
LABELS="cdngi"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0
# Any unet.losses.build_loss arm (see r4a_cdngi.sh); empty = legacy loss.
LOSS_ARM="${LOSS_ARM:-}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-4}"      # PINNED, not searched (2026-08-12) -- see
                                     # _stages.sh. 4 is the SR-series constant and
                                     # the largest that fits: 8 OOMs on 44GB (SR4RS
                                     # runs 256-ch convs, incl. a 9x9, at 512px).

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
