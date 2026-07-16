#!/bin/bash
# Arm 2 — gap_ce. GapLoss-weighted CE (endpoint-buffer weighting, Yuan & Xu
# 2022). Pixel-slot candidate. ROSA_all dataset.
#
# Buffer radius grid r ∈ {3,5,9} (default 5 = Appendix B centre): submit once
# per r; each lands as a distinct model_name (l2_gap_ce_r3/_r5/_r9).
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l2_all.sh              # r=5
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l2_all.sh GAP_R=3
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l2_all.sh GAP_R=9
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l2_gap_ce"
ARM="gap_ce"
GAP_R="${GAP_R:-5}"   # grid {3,5,9}

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
