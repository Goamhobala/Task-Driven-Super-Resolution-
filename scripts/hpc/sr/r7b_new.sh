#!/bin/bash
# R7b — STAGED joint task-driven SEN2SR fine-tuning WITHOUT padding, ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> test).
# UNet warm-started from R1b's FINAL ckpt. vs r7a: padding on/off under the
# staged protocol. pos_weight/batch/encoder pinned to R1b's best; the UNet lr
# is searched over a fine-tuning band [best/100, best] alongside lr_sr
# (PIN_LR=1 restores the exact pin — see _warm_tv.sh).
#
# Order (same SEED, same LOSS_ARM, same REG throughout):
#   bash scripts/hpc/submit.sh sr/r1b_new.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/r7b_new.sh STAGE=tune ; ... STAGE=fit
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r7b_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=0

N_TRIALS="${N_TRIALS:-60}"   # 2-D search (lr fine-tune band x lr_sr; rest pinned)

STAGE1_TAG="r1b_new"
source "$REPO_DIR/scripts/hpc/sr/_warm_tv.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
