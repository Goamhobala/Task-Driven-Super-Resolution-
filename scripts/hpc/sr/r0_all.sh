#!/bin/bash
# R0 — bicubic x4 upsampling (deterministic baseline), ROSA_all labels.
# No SR params: lr_sr is not searched; SR_PAD is irrelevant (no SR net).
# The 2.5 m anchor for the _all series (the CDNGI-series R0 does not transfer
# across datasets — every series needs its own R0).
#
#   bash scripts/hpc/submit.sh sr/r0_all.sh STAGE=tune [SEED=n]
#   bash scripts/hpc/submit.sh sr/r0_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r0_all"
LABELS="all"
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
