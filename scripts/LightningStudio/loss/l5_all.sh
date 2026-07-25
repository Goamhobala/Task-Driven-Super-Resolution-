#!/bin/bash
# Arm 5 — pstar_dice. Phase B region slot: (1−mw)·P* + mw·Dice, where P* is
# the Phase A winner and mw is SEARCHED (amendment 2026-07-21): STAGE=tune runs
# unet.tune_loss (N_TRIALS x TUNE_EPOCHS, fixed LR + train seed), STAGE=fit
# refits the tuned mw at the full budget. MIX_W=0.5 pins the legacy frozen
# ratio (and puts it in the model_name). Protocol: 2 seeds.
#
#   bash scripts/LightningStudio/run_both.sh loss/l5_all.sh PSTAR=bce      # tune->fit->bench
#   bash scripts/LightningStudio/run_both.sh loss/l5_all.sh PSTAR=bce SEED=1
#   bash scripts/LightningStudio/run_pair.sh --A=loss/l5_all.sh --B=loss/l6_all.sh MIX_W=0.5  # frozen centres
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l5_pstar_dice"
ARM="pstar_dice"
PSTAR="${PSTAR:-bce}"   # set to the Phase A winner P*

source "$LS_DIR/loss/_stages.sh"
