#!/bin/bash
# Arm 4b — t4_ce. T4-weighted CE (tight-curvature variant, Giannini et al.
# 2026), exploratory; replaces tl_ce only if it beats it. ROSA_all dataset.
#
# PENDING: the T4 curvature kernels are not yet specified, so build_loss raises
# NotImplementedError. Fill in tl_weight_map(extra_kernels=...) in
# src/unet/losses.py first; this script is the ready-to-go launcher.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l4b_all.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l4b_t4_ce"
ARM="t4_ce"
TL_ELL="${TL_ELL:-5}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
