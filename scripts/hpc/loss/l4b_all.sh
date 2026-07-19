#!/bin/bash
# Arm 4b — t4_ce. T4-weighted CE (tight-curvature variant, Giannini et al.
# 2026: TL's 4 line filters + 4 semicircle filters, base 8 -> 1), exploratory;
# replaces tl_ce only if it beats it. ROSA_all dataset.
#
# TL_THETA=0.375 reproduces the paper's binarization (default 0.5 matches the
# already-trained l3_tl_ce run; protocol grid {0.375, 0.5}).
#
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/l4a_all.sh --B=loss/l4b_all.sh
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l4b_all.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l4b_t4_ce"
ARM="t4_ce"
TL_ELL="${TL_ELL:-5}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
