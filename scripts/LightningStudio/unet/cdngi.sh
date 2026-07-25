#!/bin/bash
# UNet baseline, CDNGI labels (the split CSVs' own masks_raster).
#
#   bash scripts/LightningStudio/run.sh unet/cdngi.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh unet/cdngi.sh STAGE=fit [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="cdngi"
DATASET_DIR="${DATASET_DIR:-${INSTAROAD_ROOT}/ROSA_Dense_CDNGI}"
MASK_DIRNAME=""   # CSVs' masks_raster = CDNGI

source "$LS_DIR/unet/_stages.sh"
