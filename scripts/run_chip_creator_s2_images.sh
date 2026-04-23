#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_chip_creator_s2_images.sh <csv_path> <output_dir> [chunk_size] [--resume-chunk N]
# Example: ./scripts/run_chip_creator_s2_images.sh \
#   /mnt/hhd/home/Projects/InstaRoadPrototype/dataset/sentinel2/sa_observations.csv \
#   /mnt/hhd/home/Projects/S2Image10mV2 \
#   100 --resume-chunk 99

CSV_PATH="${1:?Usage: $0 <csv_path> <output_dir> [chunk_size] [--resume-chunk N]}"
OUTPUT_DIR="${2:?Usage: $0 <csv_path> <output_dir> [chunk_size] [--resume-chunk N]}"
CHUNK_SIZE="${3:-100}"
RESUME_CHUNK=1

shift 3 2>/dev/null || shift $# 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume-chunk) RESUME_CHUNK="${2:?--resume-chunk requires a value}"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

S2_IMAGES_DIR="$OUTPUT_DIR/s2_images"
EXTRAS_DIR="$OUTPUT_DIR/extras"
DUPLICATES_DIR="$OUTPUT_DIR/duplicates"

mkdir -p "$S2_IMAGES_DIR" "$EXTRAS_DIR" "$DUPLICATES_DIR" \
         "$OUTPUT_DIR/chips" "$OUTPUT_DIR/seg_maps"

CHUNK_DIR=$(mktemp -d /tmp/s2_chunks_XXXX)
trap 'rm -rf "$CHUNK_DIR"' EXIT

HEADER=$(head -1 "$CSV_PATH")
TOTAL_ROWS=$(tail -n +2 "$CSV_PATH" | wc -l)
TOTAL_CHUNKS=$(( (TOTAL_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Total rows:      $TOTAL_ROWS"
echo "Chunk size:      $CHUNK_SIZE rows -> $TOTAL_CHUNKS chunks"
echo "Output dir:      $OUTPUT_DIR"
echo "Resuming from:   chunk $RESUME_CHUNK"
echo ""

tail -n +2 "$CSV_PATH" | split -l "$CHUNK_SIZE" - "$CHUNK_DIR/chunk_"

for f in "$CHUNK_DIR"/chunk_*; do
    { echo "$HEADER"; cat "$f"; } > "${f}.csv"
    rm "$f"
done

CHUNKS=("$CHUNK_DIR"/*.csv)
FAILED=0

collect_chunk_output() {
    local CHUNK_NUM="$1"
    local CHUNK_EXTRAS="$EXTRAS_DIR/chunk${CHUNK_NUM}_extras"
    local CHUNK_DUPS="$DUPLICATES_DIR/chunk${CHUNK_NUM}_duplicates"
    mkdir -p "$CHUNK_EXTRAS"

    # Move chips, checking for duplicates
    for chip in "$OUTPUT_DIR/chips"/chip_*.tif; do
        [ -f "$chip" ] || continue
        NAME=$(basename "$chip")
        if [ -f "$S2_IMAGES_DIR/$NAME" ]; then
            mkdir -p "$CHUNK_DUPS"
            mv "$chip" "$CHUNK_DUPS/$NAME"
            echo "  [DUPLICATE] $NAME -> duplicates/chunk${CHUNK_NUM}_duplicates/"
        else
            mv "$chip" "$S2_IMAGES_DIR/$NAME"
        fi
    done

    # Move everything else to extras
    for item in \
        "$OUTPUT_DIR/s2_dataset.json" \
        "$OUTPUT_DIR/filtered_obsv_records.gpkg" \
        "$OUTPUT_DIR/dask-report.html" \
        "$OUTPUT_DIR/hls_raster_dataset.csv"; do
        [ -f "$item" ] && mv "$item" "$CHUNK_EXTRAS/$(basename "$item")"
    done
    [ -d "$OUTPUT_DIR/seg_maps" ] && [ "$(ls -A "$OUTPUT_DIR/seg_maps")" ] && \
        mv "$OUTPUT_DIR/seg_maps" "$CHUNK_EXTRAS/seg_maps" && \
        mkdir -p "$OUTPUT_DIR/seg_maps"
}

for i in "${!CHUNKS[@]}"; do
    f="${CHUNKS[$i]}"
    CHUNK_NUM=$(( i + 1 ))

    if [[ "$CHUNK_NUM" -lt "$RESUME_CHUNK" ]]; then
        echo "[$(date +%H:%M:%S)] Skipping chunk $CHUNK_NUM/$TOTAL_CHUNKS (resume from $RESUME_CHUNK)"
        continue
    fi

    echo "=========================================="
    echo "[$(date +%H:%M:%S)] Chunk $CHUNK_NUM/$TOTAL_CHUNKS: $(basename "$f")"
    echo "=========================================="

    if uv run instageo/data/chip_creator.py \
        --dataframe_path="$f" \
        --output_directory="$OUTPUT_DIR" \
        --data_source=S2 \
        --data_format=csv \
        --processing_method=cog \
        --chip_size=256 \
        --cloud_coverage=0 \
        --temporal_tolerance=365 \
        --num_steps=1 \
        --nois_time_series_task \
        --noshift_to_month_start \
        --min_count=1 \
        --spatial_resolution=0.00008983; then
        echo "[$(date +%H:%M:%S)] Chunk $CHUNK_NUM/$TOTAL_CHUNKS finished. Collecting output..."
        collect_chunk_output "$CHUNK_NUM"
    else
        echo "[$(date +%H:%M:%S)] Chunk $CHUNK_NUM/$TOTAL_CHUNKS FAILED."
        collect_chunk_output "$CHUNK_NUM"
        FAILED=$(( FAILED + 1 ))
    fi
done

echo ""
echo "Done. $TOTAL_CHUNKS chunks processed, $FAILED failed."
echo "  Images:     $S2_IMAGES_DIR"
echo "  Extras:     $EXTRAS_DIR"
echo "  Duplicates: $DUPLICATES_DIR"
if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
