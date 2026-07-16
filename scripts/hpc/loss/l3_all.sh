#!/bin/bash
# Arm 3 — tl_ce. Topological-Loss-weighted CE (Nanni et al. 2024, faithful to
# paper + Giannini base reset). Pixel-slot candidate. ROSA_all dataset.
#
# Filter-length grid ℓ ∈ {3,5,7} (default 5 = paper centre; at 10 m GSD ℓ is a
# physical lookahead, ℓ px = 10·ℓ m): submit once per ℓ; each lands as a
# distinct model_name (l3_tl_ce_l3/_l5/_l7).
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l3_all.sh             # ℓ=5
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l3_all.sh TL_ELL=3
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l3_all.sh TL_ELL=7
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l3_tl_ce"
ARM="tl_ce"
TL_ELL="${TL_ELL:-5}"   # grid {3,5,7}

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
