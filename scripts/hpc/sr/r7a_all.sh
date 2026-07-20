#!/bin/bash
# R7a — STAGED joint task-driven SEN2SR fine-tuning WITH padding (8 px),
# ROSA_all. UNet warm-started from R1a's fitted ckpt (frozen-SEN2SR, padded),
# then UNet + SEN2SR tuned together on the segmentation loss alone.
#   R7a - R1a: task-driven adaptation given a competent critic (clean, same
#   UNet lineage).   R7a - R2a: critic warm-start effect (staged vs cold).
#   (R7x - R1x) vs (R6 - R5): constrained-vs-unconstrained adaptation headroom
#   under the SAME staged protocol — the cross-model thesis comparison.
# lr/pos_weight/batch/encoder pinned to R1a's best; search covers lr_sr only.
#
# Order (same SEED, same LOSS_ARM throughout):
#   bash scripts/hpc/submit.sh sr/r1a_all.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/r7a_all.sh STAGE=tune ; ... STAGE=fit
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r7a_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

N_TRIALS="${N_TRIALS:-40}"   # 1-D search (lr_sr only; rest pinned) needs far fewer trials

STAGE1_TAG="r1a_all"
source "$REPO_DIR/scripts/hpc/sr/_warm.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
