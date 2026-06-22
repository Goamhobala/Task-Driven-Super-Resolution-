#!/bin/bash
#
# Plot patch classifications for every satellite image in <dataset-dir>/imagery/.
# Two PNGs per zone (satellite + binary-mask backdrop) are written to
# <dataset-dir>/classification_plots/.
#
# Usage:
#   run_visualize.bash <dataset-dir>
#
# Example:
#   run_visualize.bash /Volumes/FILES/S2ROSA

set -u

# 1. Validate arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset-dir>"
    exit 1
fi

DATASET_DIR="$1"

IMAGERY_DIR="$DATASET_DIR/imagery"
if [ ! -d "$IMAGERY_DIR" ]; then
    echo "Error: imagery directory '$IMAGERY_DIR' does not exist."
    exit 1
fi

# 2. cli uses absolute `sentinel2data.*` imports, so run it as a module from src/.
#    scripts/ -> sentinel2data/ -> src/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 3. Ensure the output directory exists (savefig does not create it).
mkdir -p "$DATASET_DIR/classification_plots"

# Headless plotting (no GUI display needed for savefig).
export MPLBACKEND=Agg

echo "Dataset: $DATASET_DIR"
echo "----------------------------------------"

# 4. Iterate every satellite image; skip macOS forks (._*) and mask files.
cd "$SRC_DIR" || exit 1
shopt -s nullglob
FOUND=0
for SAT_PATH in "$IMAGERY_DIR"/*.tif "$IMAGERY_DIR"/*.tiff; do
    SAT_IMG="$(basename "$SAT_PATH")"
    case "$SAT_IMG" in
        ._*|*_mask.tif) continue ;;
    esac

    FOUND=$((FOUND + 1))
    STEM="${SAT_IMG%.*}"

    for BACKDROP in satellite mask; do
        echo "Visualizing: $STEM ($BACKDROP)"
        uv run --extra sentinel2 python -m sentinel2data.cli visualize \
            --dataset-dir "$DATASET_DIR" \
            --zone-name "$STEM" \
            --backdrop "$BACKDROP" || echo " -> FAILED for $STEM ($BACKDROP)"
    done
done

echo "----------------------------------------"
if [ "$FOUND" -eq 0 ]; then
    echo "No satellite images found in $IMAGERY_DIR"
    exit 1
fi
echo "Finished. Plotted $FOUND image(s) to $DATASET_DIR/classification_plots/"
