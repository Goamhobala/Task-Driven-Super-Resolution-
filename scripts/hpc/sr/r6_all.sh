#!/bin/bash
# R6 — STAGED joint task-driven SR4RS fine-tuning, ROSA_all. The UNet is
# warm-started from R5's fitted ckpt (frozen-SR4RS stage 1), then UNet + SR4RS
# generator are tuned together on the segmentation loss alone. Warm start =
# the task-critic is competent before its gradients sculpt the pretrained
# generator (Grigoryev et al. 2022 analogue of pretraining the discriminator).
#   R5 - R0: value of frozen pretrained SR4RS.   R6 - R5: value of task-driven
#   adaptation given a competent critic (the thesis contrast).
# lr/pos_weight/batch/encoder are PINNED to R5's best (same loss = same
# critic); the search covers lr_sr only. No pad variant (no FFT constraint).
#
# Order (same SEED, same LOSS_ARM throughout):
#   bash scripts/hpc/submit.sh sr/r5_all.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/r6_all.sh STAGE=tune ; ... STAGE=fit
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r6_all"
LABELS="all"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"  # pre-pin fallback; _warm.sh pins to R5's best

N_TRIALS="${N_TRIALS:-40}"   # 1-D search (lr_sr only; rest pinned) needs far fewer trials

STAGE1_TAG="r5_all"
source "$REPO_DIR/scripts/hpc/sr/_warm.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
