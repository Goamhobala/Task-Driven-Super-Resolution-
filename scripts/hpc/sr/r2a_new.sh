#!/bin/bash
# R2a — COLD joint task-driven SEN2SR fine-tuning WITH padding (8 px), ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> test).
# UNet starts from ImageNet; the tune stage searches the joint LR pair.
# Cold-vs-warm contrast: r7a_new.sh is the same treatment with a warm-started
# UNet (competent critic) — R7a minus R2a isolates the critic-init effect.
#
#   bash scripts/hpc/submit.sh sr/r2a_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2a_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2a_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2a_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

# Recipe-v2 regularisation toggle (consumed by _stages_tv.sh). Exposed here so
# the unregularised ablation is a one-flag submit: REG=false.
REG="${REG:-true}"

# Save the fine-tuned SR generator every 2 epochs during the refit (weights
# only). Optional: override SR_SNAPSHOT_EVERY=0 to switch off.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
