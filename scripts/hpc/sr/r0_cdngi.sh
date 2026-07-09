#!/bin/bash
# R0 — bicubic x4 upsampling (deterministic baseline), CDNGI labels.
# No SR params: lr_sr is not searched; SR_PAD is irrelevant (no FFT constraint).
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r0_cdngi.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r0_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r0_cdngi"
LABELS="cdngi"
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
