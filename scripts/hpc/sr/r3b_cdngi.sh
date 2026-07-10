#!/bin/bash
# R3b — joint task-driven fine-tuning of the FULL (Mamba) SEN2SR WITHOUT
# padding, CDNGI labels. R2's protocol with the heavier SR net: the tune
# stage searches the joint LR pair (lr, lr_sr).
# vs r3a: padding on/off isolates the FFT border artifact, as for R1/R2.
#
# Prerequisites: the full SEN2SR mlstac dir on scratch (override SEN2SR_DIR if
# yours is named differently) and `uv pip install mamba-ssm` in the venv.
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r3b_cdngi.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r3b_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r3b_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr_full"
FREEZE_SR="false"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SR_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"   # MambaSR is much heavier than Lite at 512px

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
