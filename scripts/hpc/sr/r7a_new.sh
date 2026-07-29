#!/bin/bash
# R7a — STAGED joint task-driven SEN2SR fine-tuning WITH padding (8 px),
# ROSA_New. FINAL protocol (tune on train/val -> refit on train+val -> test).
# UNet warm-started from R1a's FINAL ckpt (frozen-SEN2SR, padded), then UNet +
# SEN2SR tuned together on the segmentation loss alone.
#   R7a - R1a: task-driven adaptation given a competent critic (clean, same
#   UNet lineage).   R7a - R2a: critic warm-start effect (staged vs cold).
#   (R7x - R1x) vs (R6 - R5): constrained-vs-unconstrained adaptation headroom
#   under the SAME staged protocol — the cross-model thesis comparison.
# pos_weight/batch/encoder pinned to R1a's best; the UNet lr is searched over
# a fine-tuning band [best/100, best] alongside lr_sr (PIN_LR=1 restores the
# exact pin — see _warm_tv.sh).
#
# Order (same SEED, same LOSS_ARM, same REG throughout):
#   bash scripts/hpc/submit.sh sr/r1a_new.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/r7a_new.sh STAGE=tune ; ... STAGE=fit
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r7a_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

N_TRIALS="${N_TRIALS:-60}"   # 2-D search (lr fine-tune band x lr_sr; rest pinned)

STAGE1_TAG="r1a_new"
source "$REPO_DIR/scripts/hpc/sr/_warm_tv.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
