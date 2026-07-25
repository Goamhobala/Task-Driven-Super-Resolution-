#!/bin/bash
# R3a — joint task-driven fine-tuning of the FULL (Mamba) SEN2SR WITH
# reflect-padding (8 px), CDNGI labels. R2's protocol with the heavier SR net:
# the tune stage searches the joint LR pair (lr, lr_sr).
# vs r3b: padding on/off isolates the FFT border artifact, as for R1/R2.
#
# Prerequisites: the full SEN2SR mlstac dir on scratch (override SEN2SR_DIR if
# yours is named differently) and `uv pip install mamba-ssm` in the venv.
#
#   bash scripts/LightningStudio/run.sh sr/r3a_cdngi.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r3a_cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r3a_cdngi"
LABELS="cdngi"
UPSAMPLER="sen2sr_full"
FREEZE_SR="false"
SR_PAD=8
SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SEN2SR_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"   # MambaSR is much heavier than Lite at 512px

source "$LS_DIR/sr/_stages.sh"
