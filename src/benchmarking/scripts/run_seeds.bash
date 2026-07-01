#!/usr/bin/env bash
# Train one UNet config across N seeds (default 0 1 2), evaluate each trained
# checkpoint to the benchmark store over non-overlapping chips, then report the
# cross-seed mean +/- std (training instability).
#
# For a pairwise comparison, run this twice with different MODEL_NAME + UNET_CONFIG
# into the SAME STORE_DIR, then:
#   python -m benchmarking.cli --config "$BENCH_CONFIG" compare \
#       --store-dir "$STORE_DIR" --model-a unet_a --model-b unet_b
#
# Env overrides: DATA, UNET_CONFIG, BENCH_CONFIG, MODEL_NAME, STORE_DIR, SEEDS.
#
#   MODEL_NAME=unet_enhanced SEEDS="0 1 2" bash src/benchmarking/scripts/run_seeds.bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

DATA="${DATA:-/Volumes/MacOSFiles/ROSA_CDNGI_dense}"
UNET_CONFIG="${UNET_CONFIG:-src/unet/configs/unet.yaml}"
BENCH_CONFIG="${BENCH_CONFIG:-src/benchmarking/configs/benchmark.yaml}"
MODEL_NAME="${MODEL_NAME:-unet}"
STORE_DIR="${STORE_DIR:-benchmarks}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"

UNET="python -m unet.cli"
BENCH="python -m benchmarking.cli"

for s in "${SEEDS[@]}"; do
  ckdir="checkpoints/${MODEL_NAME}_seed${s}"
  echo "=================== ${MODEL_NAME} seed ${s}: TRAIN ==================="
  # Fresh checkpoints/ per seed: the config's ModelCheckpoint writes there, then
  # we move the whole dir so the next seed does not overwrite it.
  rm -rf checkpoints
  $UNET fit --config "$UNET_CONFIG" --seed_everything "$s" --data.dataset_dir "$DATA"
  rm -rf "$ckdir" && mv checkpoints "$ckdir"

  echo "=================== ${MODEL_NAME} seed ${s}: EVAL -> store ==================="
  $BENCH --config "$BENCH_CONFIG" eval \
    --dataset-dir "$DATA" \
    --checkpoint "$ckdir/last.ckpt" \
    --model-name "$MODEL_NAME" \
    --seed "$s" \
    --store-dir "$STORE_DIR"
done

echo "=================== cross-seed variance (${MODEL_NAME}) ==================="
$BENCH --config "$BENCH_CONFIG" variance --store-dir "$STORE_DIR" --model-name "$MODEL_NAME"
