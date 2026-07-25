#!/bin/bash
# R4b — joint task-driven fine-tuning of SR4RS (WGAN-GP-trained generator,
# PyTorch port) WITHOUT padding, CDNGI labels. See r4a_cdngi.sh.
#
#   bash scripts/LightningStudio/run.sh sr/r4b_cdngi.sh STAGE=tune [SEED=n] [LOSS_ARM=arm]
#   bash scripts/LightningStudio/run.sh sr/r4b_cdngi.sh STAGE=fit [SEED=n] [LOSS_ARM=arm]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r4b_cdngi"
LABELS="cdngi"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0
# Any unet.losses.build_loss arm (see r4a_cdngi.sh); empty = legacy loss.
LOSS_ARM="${LOSS_ARM:-}"
SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"  # 8 OOMs on 44GB: SR4RS runs 256-ch convs
                                     # (incl. a 9x9) at the full 512px grid

source "$LS_DIR/sr/_stages.sh"
