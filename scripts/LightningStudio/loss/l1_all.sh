#!/bin/bash
# Arm 1 — bce. Plain BCE, no region/skeleton slot. Distribution-family anchor
# (expected floor) and the H1 reference. ROSA_all dataset.
#
#   bash scripts/LightningStudio/run_both.sh loss/l1_all.sh          # fit->bench
#   bash scripts/LightningStudio/run.sh loss/l1_all.sh STAGE=fit [SEED=n]
#   bash scripts/LightningStudio/run.sh loss/l1_all.sh STAGE=bench [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="l1_bce"
ARM="bce"

source "$LS_DIR/loss/_stages.sh"
