#!/bin/bash
# UNet baseline, CDNGI labels (the split CSVs' own masks_raster).
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=unet/cdngi.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=unet/cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="cdngi"
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
MASK_DIRNAME=""   # CSVs' masks_raster = CDNGI

source "$REPO_DIR/scripts/hpc/unet/_stages.sh"
