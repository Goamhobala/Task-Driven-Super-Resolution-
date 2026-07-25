#!/bin/bash
# =============================================================================
# Launch any run.sh / run_both.sh / run_pair.sh invocation as a DETACHED
# background job, so it keeps running after you close the browser tab / lose the
# terminal. Output is logged and the PID recorded so you can follow or stop it.
#
#   bash scripts/LightningStudio/job.sh <run|run_both|run_pair> <args...>
#
# Examples:
#   bash scripts/LightningStudio/job.sh run_both sr/r2a_all.sh SEED=0
#   bash scripts/LightningStudio/job.sh run unet/cdngi.sh STAGE=tune
#   bash scripts/LightningStudio/job.sh run_pair --A=loss/l1_all.sh --B=loss/la0_all.sh
#
# Manage running jobs:
#   tail -f <the .log path printed below>     # follow progress
#   kill $(cat <the .pid path printed below>) # stop it
#
# This detaches WITHIN the current Studio machine (the machine must stay on).
# To run on a SEPARATE machine that spins up and tears down on its own, use the
# Lightning SDK helper instead:  python scripts/LightningStudio/submit_job.py --help
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

[ $# -ge 2 ] || { echo "usage: bash scripts/LightningStudio/job.sh <run|run_both|run_pair> <args...>" >&2; exit 2; }

DISP="$1"; shift
case "$DISP" in
  run|run_both|run_pair) : ;;
  *) echo "ERROR: first arg must be run | run_both | run_pair (got '$DISP')" >&2; exit 2 ;;
esac
DISP_PATH="$LS_DIR/${DISP}.sh"
[ -f "$DISP_PATH" ] || { echo "ERROR: dispatcher not found: $DISP_PATH" >&2; exit 2; }

JOBS_DIR="${JOBS_DIR:-$INSTAROAD_ROOT/jobs}"
mkdir -p "$JOBS_DIR"
# Name the job after the first script-looking arg (strip dirs/extension/flags).
TAG="job"
for a in "$@"; do case "$a" in *.sh|--A=*.sh|--a=*.sh) TAG="$(basename "${a#*=}" .sh)"; break ;; esac; done
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$JOBS_DIR/${DISP}_${TAG}_${STAMP}.log"
PIDF="$JOBS_DIR/${DISP}_${TAG}_${STAMP}.pid"

echo "launching detached: $DISP $*"
# setsid detaches from the terminal; nohup ignores SIGHUP; output -> LOG.
setsid nohup bash "$DISP_PATH" "$@" > "$LOG" 2>&1 &
echo $! > "$PIDF"

echo "  pid : $(cat "$PIDF")   (stop with:  kill \$(cat $PIDF) )"
echo "  log : $LOG"
echo "  follow:  tail -f $LOG"
