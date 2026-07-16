#!/bin/bash
# Arm 5 — pstar_dice. Phase B region slot: 0.5·P* + 0.5·Dice, where P* is the
# Phase A pixel-slot winner. ROSA_all dataset. Protocol: 2 seeds.
#
# Set PSTAR to the Phase A winner (default bce until Phase A is decided). If P*
# is gap_ce/tl_ce, also pass its hyperparameter so P* is reproduced exactly.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l5_all.sh PSTAR=gap_ce GAP_R=5
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l5_all.sh PSTAR=gap_ce GAP_R=5 SEED=1
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l5_pstar_dice"
ARM="pstar_dice"
PSTAR="${PSTAR:-bce}"   # set to the Phase A winner P*

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
