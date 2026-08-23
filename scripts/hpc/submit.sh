#!/bin/bash
# Submit wrapper with DYNAMIC job names — thin front-end for train.sbatch /
# train_both.sbatch (which keep static #SBATCH names; SLURM headers can't
# interpolate, so the name is injected here via `sbatch -J`).
#
#   bash scripts/hpc/submit.sh [--both] <script under scripts/hpc/> \
#        [KEY=VALUE ...] [-- <extra sbatch flags>]
#
#   bash scripts/hpc/submit.sh sr/r5_all.sh STAGE=tune SEED=1
#       -> sbatch -J r5_all_tune_s1 ... train.sbatch --SCRIPT=sr/r5_all.sh ...
#   bash scripts/hpc/submit.sh sr/r6_all.sh STAGE=fit
#       -> sbatch -J r6_all_fit_s0 --gres=gpu:1 ...      (gpu:1 auto for fit/bench)
#   bash scripts/hpc/submit.sh --both sr/r7a_all.sh LOSS_ARM=bce_dice+cldice
#       -> sbatch -J r7a_all_both_s0_bce_dice-cldice ... train_both.sbatch ...
#   bash scripts/hpc/submit.sh sr/r2a_all.sh STAGE=tune -- --time=16:00:00
#
# Logs: slurm-%x-%j.txt (name + jobid), so `ls slurm-r6_all*` finds a run.
# DRY_RUN=1 prints the sbatch command without submitting.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$HERE/../.." && pwd)}"

BOTH=0
SCRIPT=""
KV=()
EXTRA=()
STAGE="tune"
SEED="0"
LOSS_ARM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --both) BOTH=1 ;;
    --)     shift; EXTRA=("$@"); break ;;
    *=*)    KV+=("$1")
            case "$1" in
              STAGE=*)    STAGE="${1#*=}" ;;
              SEED=*)     SEED="${1#*=}" ;;
              LOSS_ARM=*) LOSS_ARM="${1#*=}" ;;
            esac ;;
    *)      if [ -z "$SCRIPT" ]; then SCRIPT="$1"; else
              echo "ERROR: unexpected argument '$1' (KEY=VALUE, --both, or -- <sbatch flags>)" >&2; exit 2
            fi ;;
  esac
  shift
done

if [ -z "$SCRIPT" ]; then
  echo "usage: bash scripts/hpc/submit.sh [--both] <script> [KEY=VALUE ...] [-- <sbatch flags>]" >&2
  exit 2
fi
if [ ! -f "$REPO_DIR/scripts/hpc/$SCRIPT" ] && [ ! -f "$SCRIPT" ]; then
  echo "ERROR: inner script not found: $SCRIPT (looked in $REPO_DIR/scripts/hpc/)" >&2
  exit 2
fi

TAG="$(basename "$SCRIPT" .sh)"
LOSS_TAG=""
[ -n "$LOSS_ARM" ] && LOSS_TAG="_$(echo "$LOSS_ARM" | tr '+' '-')"

if [ "$BOTH" -eq 1 ]; then
  SBATCH_FILE="$REPO_DIR/scripts/hpc/train_both.sbatch"
  JOB="${TAG}_both_s${SEED}${LOSS_TAG}"
else
  SBATCH_FILE="$REPO_DIR/scripts/hpc/train.sbatch"
  JOB="${TAG}_${STAGE}_s${SEED}${LOSS_TAG}"
fi

# Convention: tune fans out over the header's gpu:2; fit/bench run on one GPU.
GRES_ARGS=()
if [ "$BOTH" -eq 0 ] && { [ "$STAGE" = "fit" ] || [ "$STAGE" = "bench" ]; }; then
  case " ${EXTRA[*]:-} " in
    *" --gres"*|*"--gres="*) : ;;               # user override wins
    *) GRES_ARGS=(--gres=gpu:1) ;;
  esac
fi

CMD=(sbatch -J "$JOB" -o "slurm-%x-%j.txt"
     ${GRES_ARGS[@]+"${GRES_ARGS[@]}"}
     ${EXTRA[@]+"${EXTRA[@]}"}
     "$SBATCH_FILE" "--SCRIPT=$SCRIPT"
     ${KV[@]+"${KV[@]}"})

echo "submit.sh -> ${CMD[*]}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "(DRY_RUN=1: not submitted)"
  exit 0
fi
exec "${CMD[@]}"
