#!/bin/bash
# R0 — bicubic x4 upsampling (deterministic baseline), ROSA_New. FINAL protocol:
# tune on train/val, refit on train+val, report on test.
# No SR params: lr_sr is not searched; SR_PAD is irrelevant (no SR net).
# The 2.5 m anchor for the _new series — R0 does not transfer across datasets
# OR across protocols, so this arm must be rerun even though r0_all exists.
#
#   bash scripts/hpc/submit.sh sr/r0_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/r0_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/r0_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r0_new"
LABELS="new"
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
