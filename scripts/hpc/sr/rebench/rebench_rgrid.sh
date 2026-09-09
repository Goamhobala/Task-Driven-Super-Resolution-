#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=rgrid-rebench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# RE-BENCH THE r2grid RAILS ARMS ON THE RELABELLED TEST SPLIT.
#
#   sbatch scripts/hpc/sr/rebench/rebench_rgrid.sh
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
# best_params.yaml} plus a manifest.json. The NAMES ARE THE INTERFACE -- do not
# rename the files -- and manifest.json is authoritative for model_name / seed /
# exp_tag, having been built from the arms' existing store rows so the new rows
# group with the old ones.
#
# COVERS 8 RUNS: the r2grid rails arms, {on,off} x ls1e-{4,5,6,7}, all seed 0.
# The R series is a SEPARATE staged folder and a separate job
# (rebench_rseries.sh) -- it is already benched, so nothing here re-uploads or
# re-scores it.
#
# SAME STORE AS THE R SERIES, deliberately: these rows carry different
# model_names, so they cannot collide, and one store of new-label rows is what
# `report` needs to put the grid and the series on one axis.
#
#   RUNS_DIR / STORE_DIR / DATASET_DIR / VENV_DIR   paths
#   ARMS="on_ls1e-4"  only these arms (substring match), e.g. ARMS=on / ARMS=off
#                     to split the constraint lanes
#   DRY_RUN=1         print the plan and exit
#   BATCH_SIZE=8      lower on OOM; does NOT change the scores
set -euo pipefail

USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
ROOT="${ROOT:-/scratch/${USER_NAME}/InstaRoad}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
RUNS_DIR="${RUNS_DIR:-${ROOT}/runs/rgrid_rebench}"
STORE_DIR="${STORE_DIR:-${ROOT}/benchmarks_corrected}"
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

# Same runner as the R series: the staged folder + its manifest are the only
# things that differ.
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
