#!/bin/bash
# UNet baseline, OSM labels (masks_osm_10m beside each split's imagery;
# same CDNGI-built imagery/splits, so the metric delta vs unet/cdngi.sh is
# attributable to the label source alone).
#
# Prerequisite (once): sort the 10 m OSM masks into the dataset —
#   OpenStreetMapTest/sort_osm_masks.py --dest-dirname masks_osm_10m
#   (or dataset_hr_masks.py --scale 1)
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=unet/osm.sh STAGE=tune [SEED=n]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=unet/osm.sh STAGE=fit [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="osm"
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
MASK_DIRNAME="mask_osm_10m"

source "$REPO_DIR/scripts/hpc/unet/_stages.sh"
