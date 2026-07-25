#!/bin/bash
# Arm 9 — gap_tl_ce. GL+TL blended weighted CE: one pixel-slot arm whose
# attention map sums the mean-normalized GapLoss (endpoint buffers) and TL
# (directional corridors) maps — exactly 0.5·gap_ce + 0.5·tl_ce under the
# §4.4 normalization. Post-hoc Phase-A amendment (dated 2026-07-19) motivated
# by Nanni et al. 2024 Table 2, where GL+TL / GL+TL+DI are the best compounds
# on 3 of 4 datasets; NOT a slot violation (see make_gap_tl_ce docstring).
# The paper's GL+TL+DI = Phase B pstar_dice with PSTAR=gap_tl_ce.
#
#   bash scripts/LightningStudio/run_pair.sh --A=loss/l9_all.sh --B=loss/l3_all.sh B.TL_THETA=0.375
#   bash scripts/LightningStudio/run_both.sh loss/l9_all.sh
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l9_gap_tl_ce"
ARM="gap_tl_ce"
GAP_R="${GAP_R:-5}"       # GL side (Appendix B centre)
TL_ELL="${TL_ELL:-5}"     # TL side
TL_THETA="${TL_THETA:-0.375}"

source "$LS_DIR/loss/_stages.sh"
