#!/bin/bash
# =============================================================================
# Run a full experiment end-to-end on Lightning Studio: STAGE=tune, then fit,
# then bench, for one experiment, back-to-back. Replaces train_both.sbatch.
#
#   bash scripts/LightningStudio/run_both.sh <script under LightningStudio/> [KEY=VALUE ...]
#
# Examples:
#   bash scripts/LightningStudio/run_both.sh unet/cdngi.sh
#   bash scripts/LightningStudio/run_both.sh sr/r2a_cdngi.sh SEED=1
#   bash scripts/LightningStudio/run_both.sh loss/la0_all.sh SEED=2
#
# Notes:
#   * STAGE is driven internally; passing STAGE=... is ignored (a warning prints).
#   * Stages run in sequence; the first failure skips the rest and exits non-zero.
#   * bench writes per-chip metrics into the SHARED store (STORE_DIR, default
#     INSTAROAD_ROOT/benchmarks) for compare / variance / report.
#   * On the free single-GPU tier this is a long run — a tune search plus a full
#     refit plus the test/bench. Watch your GPU-hour budget, or split the stages
#     across separate run.sh calls / days.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

SCRIPT=""
KV=()
while [ $# -gt 0 ]; do
  case "$1" in
    --SCRIPT=*|--script=*) SCRIPT="${1#*=}" ;;
    STAGE=*|stage=*)       echo "WARN: STAGE is driven by run_both.sh; ignoring '$1'." >&2 ;;
    *=*)                   KV+=("$1") ;;
    *) if [ -z "$SCRIPT" ]; then SCRIPT="$1"; else
         echo "ERROR: unexpected argument '$1' (expected KEY=VALUE)" >&2; exit 2
       fi ;;
  esac
  shift
done

if [ -z "$SCRIPT" ]; then
  echo "usage: bash scripts/LightningStudio/run_both.sh <script> [KEY=VALUE ...]" >&2
  exit 2
fi
if [ -f "$LS_DIR/$SCRIPT" ]; then
  SCRIPT_PATH="$LS_DIR/$SCRIPT"
elif [ -f "$SCRIPT" ]; then
  SCRIPT_PATH="$SCRIPT"
else
  echo "ERROR: inner script not found: $SCRIPT (looked in $LS_DIR/)" >&2
  exit 2
fi

for kv in ${KV[@]+"${KV[@]}"}; do export "$kv"; done

echo "### run_both.sh: $SCRIPT_PATH  [${KV[*]:-}] ###"

echo "### ==> STAGE=tune ###"
STAGE=tune bash "$SCRIPT_PATH"

echo "### ==> STAGE=fit ###"
STAGE=fit bash "$SCRIPT_PATH"

echo "### ==> STAGE=bench ###"
STAGE=bench bash "$SCRIPT_PATH"

echo "### run_both.sh: DONE (tune + fit + bench) ###"
