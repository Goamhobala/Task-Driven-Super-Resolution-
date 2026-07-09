#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="train-both"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out
#SBATCH --mem-per-cpu=8G

# End-to-end dispatcher: run STAGE=tune then STAGE=fit for one experiment in a
# SINGLE allocation. Same interface as train.sbatch (--SCRIPT= + KEY=VALUE), but
# you don't hand-submit the fit once the search finishes — this does both.
#
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=unet/cdngi.sh
#   sbatch scripts/hpc/train_both.sbatch --SCRIPT=sr/r2a_cdngi.sh SEED=1
#
# Notes vs train.sbatch:
#   * gpu:2 so the tune stage fans out across both GPUs; the fit stage uses one
#     GPU (REFIT_GPUS=1) and leaves the other idle — the price of one job.
#   * --time is bumped (search + full refit + test back-to-back). Override on the
#     command line if your experiment needs more/less.
#   * STAGE is driven internally; passing STAGE=... is ignored (a warning prints).
#   * If tune fails, fit is skipped and the job exits non-zero.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

SCRIPT=""
PASS_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --SCRIPT=*|--script=*) SCRIPT="${arg#*=}" ;;
    STAGE=*|stage=*)       echo "WARN: STAGE is driven by train_both.sbatch; ignoring '$arg'." >&2 ;;
    *=*)                   export "$arg" ;;          # KEY=VALUE -> env override
    *)                     PASS_ARGS+=("$arg") ;;    # forwarded positional arg
  esac
done

if [ -z "$SCRIPT" ]; then
  echo "usage: sbatch scripts/hpc/train_both.sbatch --SCRIPT=<script.sh> [KEY=VALUE ...] [args...]" >&2
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

# STAGE is set per-run as an inline assignment so it wins over anything the
# environment carries and never leaks between the two invocations.
echo "### train_both.sbatch: $SCRIPT_PATH  args=[${PASS_ARGS[*]:-}] ###"

echo "### ==> STAGE=tune ###"
STAGE=tune bash "$SCRIPT_PATH" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

echo "### ==> STAGE=fit ###"
STAGE=fit bash "$SCRIPT_PATH" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

echo "### train_both.sbatch: DONE (tune + fit) ###"
