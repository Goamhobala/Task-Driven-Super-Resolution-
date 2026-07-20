#!/bin/bash
# R2a — COLD joint task-driven SEN2SR fine-tuning WITH padding (8 px), ROSA_all.
# UNet starts from ImageNet; the tune stage searches the joint LR pair.
# Cold-vs-warm contrast: r7a_all.sh is the same treatment with a warm-started
# UNet (competent critic) — R7a minus R2a isolates the critic-init effect.
#
#   bash scripts/hpc/submit.sh sr/r2a_all.sh STAGE=tune [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2a_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2a_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
