#!/bin/bash
# Arm A0 — bce_dice. 0.5·BCE + 0.5·Dice. The default baseline / Phase B anchor;
# its seed replicates define the seed-noise band Decision A/B compare against.
# ROSA_all dataset. Run several seeds to get the band.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/la0_all.sh          # seed 0
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/la0_all.sh SEED=1
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/la0_all.sh SEED=2
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="la0_bce_dice"
ARM="bce_dice"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
