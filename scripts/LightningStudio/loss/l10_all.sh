#!/bin/bash
# Arm 10 — wbce. Pos-weighted CE as its OWN pixel-slot candidate (amendment
# 2026-07-20): a static class-level reweighting, same slot as GapLoss/TL's
# spatially adaptive ones. The bce_dice anchor stays PLAIN (Giannini Eq. 3 /
# the literature default) — "BCE+Dice with tunable pos_weight" is Phase B's
# pstar_dice with PSTAR=wbce, not a modified anchor.
#
# POS_WEIGHT grid: 5 (legacy default) and ~40 (inverse road frequency at
# ~2-3% road pixels). NB much of pos_weight's effect is an operating-point
# shift the per-arm θ* sweep already grants every arm — the residual under
# test is the gradient balance during training.
#
#   bash scripts/LightningStudio/run_pair.sh --A=loss/l10_all.sh --B=loss/l10_all.sh B.POS_WEIGHT=40
#   bash scripts/LightningStudio/run_pair.sh --A=loss/l10_all.sh --B=loss/l5_all.sh B.PSTAR=wbce
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l10_wbce"
ARM="wbce"
POS_WEIGHT="${POS_WEIGHT:-5}"   # grid {5, 40}

source "$LS_DIR/loss/_stages.sh"
