#!/bin/bash
# R1b — frozen SEN2SR preprocessing WITHOUT padding, CDNGI labels.
# vs r1a: same frozen SR, padding on/off isolates the FFT border artifact.
# Frozen SR: lr_sr is not searched.
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r1b_cdngi.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r1b_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1b_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
