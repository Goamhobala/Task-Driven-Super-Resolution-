#!/bin/bash
# Arm 8 — bstar_skelrec. Phase C skeleton slot: B* + w·SkelRecall (w=1, tube
# r=1), where B* is the Phase B winner. Near-free topology (GT-side skeletons
# only). ROSA_all. 2 seeds. NB additive mix, so the effective anchor weight
# differs from arm 7's convex mix (flagged in the protocol review).
#
# BSTAR selects the base (default bce_dice). If B* is a pstar compound, set
# BSTAR + PSTAR (+ its hp) AND B*'s tuned ratio explicitly (MIX_W=<value from
# the Phase B best_loss_params.yaml> — not auto-found across run dirs).
# Skeleton weight ramps over WARMUP_START→+WARMUP_RAMP.
#
# NO Optuna for Phase C (warmup; tune is a no-op here): w is the pre-registered
# GRID {0.5, 1, 2}, one submit per point (distinct model_name _sw<w>).
#
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/l8_all.sh --B=loss/l8_all.sh \
#       A.SR_W=0.5 B.SR_W=2 BSTAR=pstar_dice PSTAR=bce MIX_W=0.63             # grid ends
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l8_all.sh BSTAR=pstar_dice PSTAR=bce MIX_W=0.63  # centre w=1
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

BSTAR="${BSTAR:-bce}"   # Phase B winner B*
EXP_TAG="l8_${BSTAR}_skelrec"
ARM="${BSTAR}+skelrec"
SR_W="${SR_W:-1.0}"
SR_RADIUS="${SR_RADIUS:-1}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
