#!/bin/bash
# Arm A1 — focal_tversky. (1 − Tversky)^0.75 (Abraham & Khan 2019). A region-slot
# baseline seen in the road-seg literature. ROSA_all dataset.
#
#   bash scripts/LightningStudio/run_both.sh loss/la1_all.sh
#   bash scripts/LightningStudio/run_both.sh loss/la1_all.sh TVERSKY_ALPHA=0.7
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="la1_focal_tversky"
ARM="focal_tversky"
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"

source "$LS_DIR/loss/_stages.sh"
