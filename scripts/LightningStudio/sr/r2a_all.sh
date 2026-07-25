#!/bin/bash
# R2a — COLD joint task-driven SEN2SR fine-tuning WITH padding (8 px), ROSA_all.
# UNet starts from ImageNet; the tune stage searches the joint LR pair.
# Cold-vs-warm contrast: r7a_all.sh is the same treatment with a warm-started
# UNet (competent critic) — R7a minus R2a isolates the critic-init effect.
#
#   bash scripts/LightningStudio/run.sh sr/r2a_all.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r2a_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2a_all"
LABELS="all"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

# Recipe-v2 regularisation toggle (consumed by _stages.sh). Exposed here so the
# unregularised-GAN ablation is a one-flag submit: REG=false.
REG="${REG:-true}"

# Save the fine-tuned SR4RS generator every 2 epochs during the refit (weights
# only, ~45 MB/frame). Optional: override SR_SNAPSHOT_EVERY=0 to switch off.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

source "$LS_DIR/sr/_stages.sh"
