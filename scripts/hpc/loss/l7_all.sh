#!/bin/bash
# Arm 7 — bstar_cldice. Phase C skeleton slot: (1−α)·B* + α·clDice (α=0.3, k=5),
# where B* is the Phase B winner. Differentiable topology. ROSA_all. 2 seeds.
#
# BSTAR selects the base (default bce_dice = the A0 anchor). If B* is a pstar
# compound, set BSTAR + PSTAR (+ its hp) AND pass B*'s TUNED ratio explicitly
# (MIX_W=<value from the Phase B run's best_loss_params.yaml>) — Phase C runs
# have their own run dirs, so the tuned file is NOT auto-found here. The
# skeleton weight ramps in over epochs WARMUP_START→+WARMUP_RAMP (default 30→40).
#
# NO Optuna for Phase C (the warmup means short trials never see the skeleton
# term — tune is a no-op here): α is the pre-registered GRID {0.2, 0.3, 0.5},
# one submit per point, each landing as a distinct model_name (_ca<α>).
#
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/l7_all.sh --B=loss/l7_all.sh \
#       A.CL_ALPHA=0.2 B.CL_ALPHA=0.5 BSTAR=pstar_dice PSTAR=bce MIX_W=0.63    # grid ends
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l7_all.sh BSTAR=pstar_dice PSTAR=bce MIX_W=0.63  # centre 0.3
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

BSTAR="${BSTAR:-bce}"   # Phase B winner B*
EXP_TAG="l7_${BSTAR}_cldice"
ARM="${BSTAR}+cldice"
CL_ALPHA="${CL_ALPHA:-0.3}"
CL_ITERS="${CL_ITERS:-5}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
