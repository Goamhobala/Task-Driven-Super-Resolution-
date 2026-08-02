#!/bin/bash
# R0 — bicubic x4 upsampling (deterministic baseline), ROSA_New. FINAL
# protocol: tune on train/val, refit on train+val, report on test. Lightning
# Studio twin of scripts/hpc/sr/r0_new.sh.
# No SR params: lr_sr is not searched; SR_PAD is irrelevant (no SR net).
# The 2.5 m anchor for the _new series — R0 does not transfer across datasets
# OR across protocols, so this arm must be rerun even though r0_all exists.
#
# NB the LOSS PILOT does not use this script directly — it runs the same
# config through loss/_pilot_new.sh (TRAIN_SPLITS=train holdout mode, E=50,
# bench on val). This script is the FINAL-protocol R0.
#
#   bash scripts/LightningStudio/run.sh sr/r0_new.sh STAGE=tune  [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r0_new.sh STAGE=fit   [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r0_new.sh STAGE=bench [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r0_new"
LABELS="new"
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

source "$LS_DIR/sr/_stages_tv.sh"
