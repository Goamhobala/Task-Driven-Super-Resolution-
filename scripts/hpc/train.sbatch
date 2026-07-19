#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="unet-job"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out

# Generic batch dispatcher: one set of SBATCH headers for every stage. Pick the
# experiment script with --SCRIPT= (resolved under scripts/hpc/) and pass config
# as KEY=VALUE tokens (exported as env vars). Each experiment script is fully
# self-contained; the only knobs meant to vary are SEED and STAGE (tune | fit).
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=unet/cdngi.sh STAGE=tune
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=unet/cdngi.sh STAGE=fit
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r2a_cdngi.sh STAGE=tune SEED=1
#
# The headers above are defaults; override any of them on the command line, e.g.
# `sbatch --gres=gpu:1 --time=06:00:00 scripts/hpc/train.sbatch ...` for the refit.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

SCRIPT=""
PASS_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --SCRIPT=*|--script=*) SCRIPT="${arg#*=}" ;;
    *=*)                   export "$arg" ;;          # KEY=VALUE -> env override
    *)                     PASS_ARGS+=("$arg") ;;    # forwarded positional arg
  esac
done

if [ -z "$SCRIPT" ]; then
  echo "usage: sbatch scripts/train.sbatch --SCRIPT=<script.sh> [KEY=VALUE ...] [args...]" >&2
  exit 2
fi

# Resolve the inner script: names (incl. nested, e.g. unet/cdngi.sh) resolve
# under scripts/hpc/; an existing literal/absolute path is used as given.
if [ -f "$REPO_DIR/scripts/hpc/$SCRIPT" ]; then
  SCRIPT_PATH="$REPO_DIR/scripts/hpc/$SCRIPT"
elif [ -f "$SCRIPT" ]; then
  SCRIPT_PATH="$SCRIPT"
else
  echo "ERROR: inner script not found: $SCRIPT (looked in $REPO_DIR/scripts/hpc/)" >&2
  exit 2
fi

echo "### train.sbatch: running $SCRIPT_PATH  args=[${PASS_ARGS[*]:-}] ###"
# Expand the array only if non-empty (avoids a stray empty arg under set -u).
exec bash "$SCRIPT_PATH" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
