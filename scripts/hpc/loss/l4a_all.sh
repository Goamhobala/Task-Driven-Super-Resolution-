#!/bin/bash
# Arm 4a — t2_ce. T2-weighted CE (wide-curvature variant, Giannini et al. 2026),
# exploratory; replaces tl_ce only if it beats it. ROSA_all dataset.
#
# PENDING: the T2 curvature kernels are not yet specified, so build_loss raises
# NotImplementedError. Fill in tl_weight_map(extra_kernels=...) in
# src/unet/losses.py first; this script is the ready-to-go launcher.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l4a_all.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l4a_t2_ce"
ARM="t2_ce"
TL_ELL="${TL_ELL:-5}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
