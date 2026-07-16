#!/bin/bash
# Arm 8 — bstar_skelrec. Phase C skeleton slot: B* + w·SkelRecall (w=1, tube
# r=1), where B* is the Phase B winner. Near-free topology (GT-side skeletons
# only). ROSA_all. 2 seeds. NB additive mix, so the effective anchor weight
# differs from arm 7's convex mix (flagged in the protocol review).
#
# BSTAR selects the base (default bce_dice). If B* is a pstar compound, set
# BSTAR + PSTAR (+ its hp). Skeleton weight ramps over WARMUP_START→+WARMUP_RAMP.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l8_all.sh                    # bce_dice + SkelRecall
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l8_all.sh BSTAR=pstar_dice PSTAR=gap_ce GAP_R=5
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

BSTAR="${BSTAR:-bce_dice}"   # Phase B winner B*
EXP_TAG="l8_${BSTAR}_skelrec"
ARM="${BSTAR}+skelrec"
SR_W="${SR_W:-1.0}"
SR_RADIUS="${SR_RADIUS:-1}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
