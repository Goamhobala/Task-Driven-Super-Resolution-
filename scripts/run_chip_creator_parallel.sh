#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_chip_creator_parallel.sh <csv_path> <output_dir> [num_parallel]
# Example: ./scripts/run_chip_creator_parallel.sh \
#   /mnt/hhd/home/Projects/InstaRoadPrototype/dataset/sentinel2/sa_observations.csv \
#   /mnt/hhd/home/Projects/S2Image10m \
#   4

CSV_PATH="${1:?Usage: $0 <csv_path> <output_dir> [num_parallel]}"
OUTPUT_DIR="${2:?Usage: $0 <csv_path> <output_dir> [num_parallel]}"
NUM_PARALLEL="${3:-4}"

CHUNK_DIR=$(mktemp -d /tmp/s2_chunks_XXXX)
trap 'rm -rf "$CHUNK_DIR"' EXIT

echo "Splitting $CSV_PATH into chunks of ~$(( $(tail -n +2 "$CSV_PATH" | wc -l) / NUM_PARALLEL )) rows..."

HEADER=$(head -1 "$CSV_PATH")
TOTAL_ROWS=$(tail -n +2 "$CSV_PATH" | wc -l)
CHUNK_SIZE=$(( (TOTAL_ROWS + NUM_PARALLEL - 1) / NUM_PARALLEL ))

tail -n +2 "$CSV_PATH" | split -l "$CHUNK_SIZE" - "$CHUNK_DIR/chunk_"

for f in "$CHUNK_DIR"/chunk_*; do
    { echo "$HEADER"; cat "$f"; } > "${f}.csv"
    rm "$f"
done

echo "Created $(ls "$CHUNK_DIR"/*.csv | wc -l) chunks of ~$CHUNK_SIZE rows each."
echo "Starting $NUM_PARALLEL parallel processes, logs in $CHUNK_DIR/"
echo ""

PIDS=()
for f in "$CHUNK_DIR"/*.csv; do
    LOG="$CHUNK_DIR/log_$(basename "$f" .csv).txt"
    uv run instageo/data/chip_creator.py \
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
        --spatial_resolution=0.00008983 > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  Started PID $! -> $LOG"
done

echo ""
echo "Waiting for all processes to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "  PID $PID finished successfully."
    else
        echo "  PID $PID FAILED. Check logs in $CHUNK_DIR/"
        FAILED=$(( FAILED + 1 ))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "All chunks completed successfully."
else
    echo "$FAILED chunk(s) failed. Check logs in $CHUNK_DIR/ for details."
    exit 1
fi
