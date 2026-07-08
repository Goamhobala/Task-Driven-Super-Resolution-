#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="unet-job"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out
#SBATCH --mem-per-cpu=8G

# Generic batch dispatcher: one set of SBATCH headers for every stage. Pick the
# inner script with --SCRIPT= and pass config as KEY=VALUE tokens (exported as
# env vars, which the inner scripts read). Anything else is forwarded as args.
#
#   sbatch scripts/train.sbatch --SCRIPT=tune_only.sh
#   sbatch scripts/train.sbatch --SCRIPT=tune_only.sh ENCODER_WEIGHTS=none STUDY_TAG=random
#   sbatch --gres=gpu:1 scripts/train.sbatch --SCRIPT=fit_only.sh STUDY_TAG=random
#
# The headers above are defaults; override any of them on the command line, e.g.
# `sbatch --gres=gpu:1 --time=06:00:00 scripts/train.sbatch ...` for the refit.
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

# Resolve the inner script: bare name -> scripts/, otherwise use the given path.
case "$SCRIPT" in
  */*) SCRIPT_PATH="$SCRIPT" ;;
  *)   SCRIPT_PATH="$REPO_DIR/scripts/hpc/$SCRIPT" ;;
esac
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "ERROR: inner script not found: $SCRIPT_PATH" >&2
  exit 2
fi

echo "### train.sbatch: running $SCRIPT_PATH  args=[${PASS_ARGS[*]:-}] ###"
# Expand the array only if non-empty (avoids a stray empty arg under set -u).
exec bash "$SCRIPT_PATH" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
