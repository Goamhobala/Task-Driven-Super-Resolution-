#!/bin/bash
# Arm A0 — bce_dice. 0.5·BCE + 0.5·Dice. The default baseline / Phase B anchor;
# its seed replicates define the seed-noise band Decision A/B compare against.
# ROSA_all dataset. Run several seeds to get the band.
#
#   bash scripts/LightningStudio/run_both.sh loss/la0_all.sh          # seed 0
#   bash scripts/LightningStudio/run_both.sh loss/la0_all.sh SEED=1
#   bash scripts/LightningStudio/run_both.sh loss/la0_all.sh SEED=2
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="la0_bce_dice"
ARM="bce_dice"

source "$LS_DIR/loss/_stages.sh"
