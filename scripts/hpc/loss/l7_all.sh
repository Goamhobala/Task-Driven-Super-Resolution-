#!/bin/bash
# Arm 7 — bstar_cldice. Phase C skeleton slot: (1−α)·B* + α·clDice (α=0.3, k=5),
# where B* is the Phase B winner. Differentiable topology. ROSA_all. 2 seeds.
#
# BSTAR selects the base (default bce_dice = the A0 anchor). If B* is a pstar
# compound, set BSTAR + PSTAR (+ its hp) so B* is reproduced exactly. The
# skeleton weight ramps in over epochs WARMUP_START→+WARMUP_RAMP (default 30→40).
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l7_all.sh                    # bce_dice+clDice
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l7_all.sh BSTAR=pstar_dice PSTAR=gap_ce GAP_R=5
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

BSTAR="${BSTAR:-bce_dice}"   # Phase B winner B*
EXP_TAG="l7_${BSTAR}_cldice"
ARM="${BSTAR}+cldice"
CL_ALPHA="${CL_ALPHA:-0.3}"
CL_ITERS="${CL_ITERS:-5}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
