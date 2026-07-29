#!/bin/bash
# R7b — STAGED joint task-driven SEN2SR fine-tuning WITHOUT padding, ROSA_all.
# UNet warm-started from R1b's fitted ckpt. vs r7a: padding on/off under the
# staged protocol. pos_weight/batch/encoder pinned to R1b's best; the UNet lr
# is searched over a fine-tuning band [best/100, best] alongside lr_sr
# (PIN_LR=1 restores the exact pin — see _warm.sh).
#
# Order (same SEED, same LOSS_ARM throughout):
#   bash scripts/hpc/submit.sh sr/r1b_all.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/r7b_all.sh STAGE=tune ; ... STAGE=fit
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r7b_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=0

N_TRIALS="${N_TRIALS:-60}"   # 2-D search (lr fine-tune band x lr_sr; rest pinned)

STAGE1_TAG="r1b_all"
source "$REPO_DIR/scripts/hpc/sr/_warm.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
