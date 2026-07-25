#!/bin/bash
# Arm 2 — gap_ce. GapLoss-weighted CE (endpoint-buffer weighting, Yuan & Xu
# 2022). Pixel-slot candidate. ROSA_all dataset.
#
# Buffer radius grid r ∈ {3,5,9} (default 5 = Appendix B centre): submit once
# per r; each lands as a distinct model_name (l2_gap_ce_r3/_r5/_r9).
#
#   bash scripts/LightningStudio/run_both.sh loss/l2_all.sh              # r=5
#   bash scripts/LightningStudio/run_both.sh loss/l2_all.sh GAP_R=3
#   bash scripts/LightningStudio/run_both.sh loss/l2_all.sh GAP_R=9
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l2_gap_ce"
ARM="gap_ce"
GAP_R="${GAP_R:-5}"   # grid {3,5,9}

source "$LS_DIR/loss/_stages.sh"
