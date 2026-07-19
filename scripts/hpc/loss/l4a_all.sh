#!/bin/bash
# Arm 4a — t2_ce. T2-weighted CE (wide-curvature variant, Giannini et al. 2026:
# TL's 4 line filters + 4 quarter-circle filters, base 8 -> 1), exploratory;
# replaces tl_ce only if it beats it. ROSA_all dataset.
#
# Default TL_THETA=0.375 = the paper's binarization (Appendix-B centre;
# grid {0.375, 0.5}). NB the legacy pre-knob l3_tl_ce run trained at 0.5.
#
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/l4a_all.sh --B=loss/l4b_all.sh
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l4a_all.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l4a_t2_ce"
ARM="t2_ce"
TL_ELL="${TL_ELL:-5}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
