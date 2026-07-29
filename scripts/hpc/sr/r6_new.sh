#!/bin/bash
# R6 — STAGED joint task-driven SR4RS fine-tuning, ROSA_New. FINAL protocol
# (tune on train/val -> refit on train+val -> test). The UNet is warm-started
# from R5's FINAL ckpt (frozen-SR4RS stage 1), then UNet + SR4RS generator are
# tuned together on the segmentation loss alone. Warm start = the task-critic is
# competent before its gradients sculpt the pretrained generator.
#   R5 - R0: value of frozen pretrained SR4RS.   R6 - R5: value of task-driven
#   adaptation given a competent critic (the thesis contrast).
# pos_weight/batch/encoder are PINNED to R5's best (same loss = same critic);
# the UNet lr is searched over a fine-tuning band anchored to R5's best
# ([best/100, best] — stage 2 fine-tunes a CONVERGED UNet, so the from-scratch
# lr may be destructively high; PIN_LR=1 restores the exact pin) alongside
# lr_sr. No pad variant (no FFT constraint).
#
# Order (same SEED, same LOSS_ARM, same REG throughout):
#   bash scripts/hpc/submit.sh sr/r5_new.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/r6_new.sh STAGE=tune ; ... STAGE=fit
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r6_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"  # pre-pin fallback; _warm_tv.sh pins to R5's best

N_TRIALS="${N_TRIALS:-60}"   # 2-D search (lr fine-tune band x lr_sr; rest pinned)

STAGE1_TAG="r5_new"
source "$REPO_DIR/scripts/hpc/sr/_warm_tv.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
