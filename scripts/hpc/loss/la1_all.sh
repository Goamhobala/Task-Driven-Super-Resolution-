#!/bin/bash
# Arm A1 — focal_tversky. (1 − Tversky)^0.75 (Abraham & Khan 2019). A region-slot
# baseline seen in the road-seg literature. ROSA_all dataset.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/la1_all.sh
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/la1_all.sh TVERSKY_ALPHA=0.7
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="la1_focal_tversky"
ARM="focal_tversky"
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
