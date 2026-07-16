#!/bin/bash
# Arm 6 — pstar_tversky. Phase B region slot: 0.5·P* + 0.5·Tversky(α=0.7),
# where P* is the Phase A pixel-slot winner. α>0.5 favours recall (label
# incompleteness + thin roads). ROSA_all dataset. Protocol: 2 seeds.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l6_all.sh PSTAR=gap_ce GAP_R=5
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l6_all.sh PSTAR=gap_ce GAP_R=5 TVERSKY_ALPHA=0.7 SEED=1
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l6_pstar_tversky"
ARM="pstar_tversky"
PSTAR="${PSTAR:-bce}"                 # set to the Phase A winner P*
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
