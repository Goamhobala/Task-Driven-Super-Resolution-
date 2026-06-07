#!/bin/bash
#
# Run the full S2-ROSA pipeline (mask -> classify -> visualize) for every
# satellite image found in <dataset-dir>/imagery/.
#
# Usage:
#   run_pipeline.bash <dataset-dir> <road-parquet>
#
# Example:
#   run_pipeline.bash /Volumes/FILES/S2ROSA \
#     /Volumes/FILES/RoadVectorData/OvertureSARoadData/south_africa_overture_roads.parquet

set -u

# 1. Validate arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset-dir> <road-parquet>"
    echo "Example: $0 /Volumes/FILES/S2ROSA /Volumes/FILES/.../south_africa_overture_roads.parquet"
    exit 1
fi

DATASET_DIR="$1"
ROAD_PARQUET="$2"

# 2. Validate inputs exist
IMAGERY_DIR="$DATASET_DIR/imagery"

if [ ! -d "$IMAGERY_DIR" ]; then
    echo "Error: imagery directory '$IMAGERY_DIR' does not exist."
    exit 1
fi

if [ ! -f "$ROAD_PARQUET" ]; then
    echo "Error: road parquet '$ROAD_PARQUET' does not exist."
    exit 1
fi

# 3. Locate cli.py relative to this script (scripts/ is one level below it)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$SCRIPT_DIR/../cli.py"

echo "Dataset:      $DATASET_DIR"
echo "Road parquet: $ROAD_PARQUET"
echo "----------------------------------------"

# 4. Iterate over every satellite image in imagery/
#    Skip macOS resource forks (._*) and any stray mask files.
shopt -s nullglob
FOUND=0
for SAT_PATH in "$IMAGERY_DIR"/*.tif; do
    SAT_IMG="$(basename "$SAT_PATH")"

    case "$SAT_IMG" in
        ._*|*_mask.tif) continue ;;
    esac

    FOUND=$((FOUND + 1))
    STEM="${SAT_IMG%.tif}"
    MASK_IMG="${STEM}_mask.tif"

    echo "Processing: $SAT_IMG"

    echo " -> mask"
    uv run python "$CLI" mask \
        --dataset-dir "$DATASET_DIR" \
        --sat-img "$SAT_IMG" \
        --parquet "$ROAD_PARQUET" || { echo " -> mask FAILED for $SAT_IMG"; continue; }

    echo " -> classify"
    uv run python "$CLI" classify \
        --dataset-dir "$DATASET_DIR" \
        --mask-img "$MASK_IMG" || { echo " -> classify FAILED for $SAT_IMG"; continue; }

    echo " -> visualize"
    uv run python "$CLI" visualize \
        --dataset-dir "$DATASET_DIR" \
        --sat-img "$SAT_IMG" || { echo " -> visualize FAILED for $SAT_IMG"; continue; }

    echo " -> done: $SAT_IMG"
    echo ""
done

echo "----------------------------------------"
if [ "$FOUND" -eq 0 ]; then
    echo "No satellite images found in $IMAGERY_DIR"
    exit 1
fi
echo "Finished. Processed $FOUND image(s)."
