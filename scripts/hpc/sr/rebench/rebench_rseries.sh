#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=rseries-rebench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# RE-BENCH THE R SERIES ON THE RELABELLED TEST SPLIT.
#
#   sbatch scripts/hpc/sr/rebench/rebench_rseries.sh
#
# All the work is in rebench_rseries.py; this only picks the interpreter and
# forwards the paths. The python side calls benchmarking.runner.evaluate
# DIRECTLY -- benchmarking.cli imports typer, which is not installed in the
# cluster environment.
#
# THE VENV IS NOT OPTIONAL. Without `source $VENV_DIR/bin/activate` the job
# lands on the system miniconda (python 3.9, no torch, no typer) and every run
# dies at import. _stages_tv.sh activates the same venv for the same reason.
#
# WHY THIS EXISTS
# ---------------
# The ROSA_New TEST split was manually relabelled and replaced in place. The new
# labels are not a subset of the old: 7 tiles dropped (181 -> 174) and 92 of the
# 174 survivors changed. Every existing test row is therefore a DIFFERENT
# QUANTITY from anything scored now, and no filtering reconciles them. Train and
# val were NOT touched, so the fits, the tunes and theta* all stand -- this is a
# scoring-only redo.
#
# THETA* IS STILL VALID. It is selected on VAL, which did not change, so the
# staged sweep.json holds exactly the theta the protocol would pick today.
# Re-sweeping would only reproduce it at full cost, so there is no resweep path.
#
# INPUT: a flat staged dir of <RUN_TAG>_seed<N>.{ckpt,sweep.json,
# best_params.yaml}. The NAMES ARE THE INTERFACE -- model_name, seed and exp_tag
# are parsed out of them. Do not rename the files.
#
#   RUNS_DIR / STORE_DIR / DATASET_DIR / VENV_DIR   paths
#   ARMS="r4b r3a"    only these arms (substring match)
#   DRY_RUN=1         print the plan and exit
#   BATCH_SIZE=8      lower on OOM; does NOT change the scores
set -euo pipefail

USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
ROOT="${ROOT:-/scratch/${USER_NAME}/InstaRoad}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
RUNS_DIR="${RUNS_DIR:-${ROOT}/runs/rseries_rebench}"
STORE_DIR="${STORE_DIR:-${ROOT}/benchmarks_newdata}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/ROSA_New}"
MODELS_ROOT="${MODELS_ROOT:-${ROOT}/models}"

[ -d "$RUNS_DIR" ]  || { echo "ERROR: no staged dir at $RUNS_DIR" >&2; exit 1; }
[ -x "$VENV_DIR/bin/python" ] || {
  echo "ERROR: no venv at $VENV_DIR (expected the one _stages_tv.sh uses)." >&2
  echo "  Without it this runs on the system python and every job dies at import." >&2
  exit 1; }

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

exec "$VENV_DIR/bin/python" "$REPO_DIR/scripts/hpc/sr/rebench/rebench_rseries.py" \
  --runs-dir "$RUNS_DIR" \
  --store-dir "$STORE_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --models-root "$MODELS_ROOT" \
  --split "${SPLIT:-test}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --buffer-px "${BUFFER_PX:-1,2,3,4,5}" \
  --ap-bins "${AP_BINS:-101}" \
  --tile-metrics "${TILE_METRICS:-apls,cldice}" \
  ${ARMS:+--arms $ARMS} \
  ${DRY_RUN:+--dry-run}
