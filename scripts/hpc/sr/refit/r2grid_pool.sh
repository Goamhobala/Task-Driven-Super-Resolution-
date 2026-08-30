#!/bin/bash
# POOL RUNNER — several lr_sr-grid cells x their replicate seeds, ONE allocation.
#
#   cd scripts/hpc
#   sbatch --job-name=r2grid-pool --time=24:00:00 \
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=sr/refit/r2grid_pool.sh
#
#   CELLS="r2grid_off_ls1e-4 r2grid_on_ls1e-4" SEEDS="1 2" sbatch ... \
#          train.sbatch --SCRIPT=sr/refit/r2grid_pool.sh
#
# CELLS are the generated per-cell script names under this directory, without
# `.sh` — they ARE the cell names (r2grid_<lane>_ls<rate>), so what ran is
# readable from the job's environment. Regenerate them from the synced run dirs
# with `python scripts/local/make_grid_refit_scripts.py`.
#
# WHICH CELLS TO LIST
# -------------------
# §2 and §10: NOT all eight. Seed 0 is the grid; replicate only the cells that
# end up carrying a quantitative sentence — typically the pair at the decade
# where the two lanes diverge. The default below is the four extremes, which is
# where that pair usually lives, but it is a starting point rather than a
# recommendation: eight refits is a day of GPU, and §4 forbids leaning on
# between-cell metric differences anyway. The mechanism claim rests on the
# trajectories and on the dose-response ACROSS lr_sr, which is already four
# independent runs per lane.
#
# WALLTIME
# --------
# Each (cell, seed) is a 100-epoch refit at ~2-3 h on an L40S, plus a test
# bench. The default 4 cells x 2 seeds is 8 refits, i.e. 16-24 h — size --time
# for what you actually list. A job killed at the wall clock is not lost work:
# every cell script is resumable (run_seeds skips a completed seed, RESUMEs a
# partial one from last.ckpt, and skips a bench already in the store), so
# resubmitting the same pool picks up where it stopped.
#
# A failing cell does NOT abort the pool: the remaining cells still run and the
# failures are listed at the end, with a non-zero exit so the job is not
# reported green. One bad cell should not cost the allocation.
set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
HERE="$REPO_DIR/scripts/hpc/sr/refit"

CELLS="${CELLS:-r2grid_off_ls1e-4 r2grid_on_ls1e-4 r2grid_off_ls1e-5 r2grid_on_ls1e-5}"
export SEEDS="${SEEDS:-1 2}"

# Fail before the first GPU second if a name is wrong, rather than 6 h in.
missing=()
for cell in $CELLS; do
  [ -f "${HERE}/${cell}.sh" ] || missing+=("$cell")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: no such cell script(s): ${missing[*]}" >&2
  echo "  Available:" >&2
  find "$HERE" -maxdepth 1 -name 'r2grid_*_ls*.sh' -exec basename {} .sh \; |
    sort | sed 's/^/    /' >&2
  echo "  (generate them: python scripts/local/make_grid_refit_scripts.py)" >&2
  exit 2
fi

echo "=== r2grid pool: cells=[${CELLS}]  seeds=[${SEEDS}] ==="
t_pool=$(date +%s)
failed=()
for cell in $CELLS; do
  echo ""
  echo "##################################################################"
  echo "########## POOL: ${cell}  seeds=${SEEDS}"
  echo "##################################################################"
  t0=$(date +%s)
  # Subprocess, like every other stage runner here: the engine `exit`s on
  # several paths, so a sourced chain would end at the first cell.
  if bash "${HERE}/${cell}.sh"; then
    echo "########## POOL: ${cell} done in $((($(date +%s) - t0) / 60)) min"
  else
    rc=$?
    echo "########## POOL: ${cell} FAILED (rc=${rc}) after $((($(date +%s) - t0) / 60)) min" >&2
    echo "##########   continuing with the remaining cells" >&2
    failed+=("$cell")
  fi
done

echo ""
echo "=== r2grid pool done in $((($(date +%s) - t_pool) / 60)) min ==="
if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED cells: ${failed[*]}" >&2
  echo "  Re-submit the pool with CELLS=\"${failed[*]}\" — completed seeds are" >&2
  echo "  skipped and partial fits resume, so nothing is retrained." >&2
  exit 1
fi
echo "All cells complete. Report: python -m benchmarking.cli report --store-dir <store>"
