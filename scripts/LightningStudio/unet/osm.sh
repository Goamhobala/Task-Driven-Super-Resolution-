#!/bin/bash
# UNet baseline, OSM labels (mask_osm_10 beside each split's imagery;
# same CDNGI-built imagery/splits, so the metric delta vs unet/cdngi.sh is
# attributable to the label source alone).
#
# Prerequisite (once): sort the 10 m OSM masks into the dataset —
#   OpenStreetMapTest/sort_osm_masks.py --dest-dirname mask_osm_10
#   (or dataset_hr_masks.py --scale 1)
#
#   bash scripts/LightningStudio/run.sh unet/osm.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh unet/osm.sh STAGE=fit [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="osm"
DATASET_DIR="${DATASET_DIR:-${INSTAROAD_ROOT}/ROSA_Dense_CDNGI}"
MASK_DIRNAME="mask_osm_10"   # actual on-disk dir name (see `ls <split>/`)

source "$LS_DIR/unet/_stages.sh"
