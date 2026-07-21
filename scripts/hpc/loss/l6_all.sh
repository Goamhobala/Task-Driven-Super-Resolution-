#!/bin/bash
# Arm 6 — pstar_tversky. Phase B region slot: (1−mw)·P* + mw·Tversky(α), with
# mw SEARCHED via STAGE=tune (amendment 2026-07-21; see l5_all.sh). α defaults
# to 0.7 (recall side; Xu et al.); SEARCH_TVERSKY=true searches (mw, α)
# jointly — the 2D case is where TPE actually beats a grid. Protocol: 2 seeds.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l6_all.sh PSTAR=bce
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/l6_all.sh PSTAR=bce SEARCH_TVERSKY=true
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l6_pstar_tversky"
ARM="pstar_tversky"
PSTAR="${PSTAR:-bce}"                 # set to the Phase A winner P*
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"

source "$REPO_DIR/scripts/hpc/loss/_stages.sh"
