#!/bin/bash
# =============================================================================
# Run ONE experiment stage on Lightning Studio. This replaces the cluster's
# submit.sh + train.sbatch: no SLURM, no sbatch headers — it just resolves the
# inner experiment script, exports your KEY=VALUE config, and runs it here in
# the current machine.
#
#   bash scripts/LightningStudio/run.sh <script under LightningStudio/> [KEY=VALUE ...]
#
# Examples:
#   bash scripts/LightningStudio/run.sh unet/cdngi.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh unet/cdngi.sh STAGE=fit  SEED=1
#   bash scripts/LightningStudio/run.sh sr/r2a_all.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l2_all.sh STAGE=fit GAP_R=9
#
# Notes:
#   * STAGE selects the stage the engine runs: tune | fit | bench
#     (loss arms default to fit; unet/sr default to tune). Use run_both.sh to
#     chain tune -> fit -> bench in one go.
#   * Every run tees its own log into the run dir (INSTAROAD_ROOT/runs/...),
#     the same as on the cluster.
#   * DRY_RUN=1 prints what would run without running it.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

SCRIPT=""
KV=()
while [ $# -gt 0 ]; do
  case "$1" in
    --SCRIPT=*|--script=*) SCRIPT="${1#*=}" ;;
    *=*)                   KV+=("$1") ;;
    *) if [ -z "$SCRIPT" ]; then SCRIPT="$1"; else
         echo "ERROR: unexpected argument '$1' (expected KEY=VALUE)" >&2; exit 2
       fi ;;
  esac
  shift
done

if [ -z "$SCRIPT" ]; then
  echo "usage: bash scripts/LightningStudio/run.sh <script> [KEY=VALUE ...]" >&2
  echo "  e.g. bash scripts/LightningStudio/run.sh sr/r5_all.sh STAGE=tune SEED=1" >&2
  exit 2
fi

# Resolve the inner script: a name under LightningStudio/ (incl. nested, e.g.
# sr/r5_all.sh), or an existing literal/absolute path used as given.
if [ -f "$LS_DIR/$SCRIPT" ]; then
  SCRIPT_PATH="$LS_DIR/$SCRIPT"
elif [ -f "$SCRIPT" ]; then
  SCRIPT_PATH="$SCRIPT"
else
  echo "ERROR: inner script not found: $SCRIPT (looked in $LS_DIR/)" >&2
  exit 2
fi

# KEY=VALUE tokens become environment overrides for the inner script.
for kv in ${KV[@]+"${KV[@]}"}; do export "$kv"; done

echo "### run.sh: $SCRIPT_PATH  [${KV[*]:-}] ###"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "(DRY_RUN=1: not executed)"
  exit 0
fi
exec bash "$SCRIPT_PATH"
