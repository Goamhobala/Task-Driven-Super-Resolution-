#!/bin/bash
# Arm 1 — bce. Plain BCE, no region/skeleton slot. Distribution-family anchor
# (expected floor) and the H1 reference. ROSA_all dataset.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l1_all.sh          # fit->bench
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=loss/l1_all.sh STAGE=fit [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=loss/l1_all.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l1_bce"
ARM="bce"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
