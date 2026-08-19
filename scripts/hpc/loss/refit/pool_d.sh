#!/bin/bash
# Refit pool d — the SIX arms still at n=2 seeds, on 1 GPU, in ONE SLURM job.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-pool-d --time=24:00:00 --qos=l40sfree \
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=loss/refit/pool_d.sh
#
# WHICH SEED IS ACTUALLY MISSING, AND WHY THIS POOL ADDS SEED 3
# ------------------------------------------------------------
# Seeds 1 and 2 are COMPLETE for all 22 arms, on both splits — pools a/b/c did
# their job. What these six lack is the THIRD seed, and for the other sixteen
# that third seed is seed 0: the original pilot fit, trained on Lightning/Modal
# and benched from its local checkpoint. These six never got a benched seed-0
# row (checked 2026-08-19 against benchmarks_seeds/val):
#
#     gap_ce  wbce  wbce_dice  gap_tl_ce_lcdice  gap_tl_ce_sdice  gap_tl_dice
#
# Five of them still HAVE a usable seed-0 checkpoint, but it lives on the Mac
# under runslightning/, not on scratch — so the cluster cannot reach it and
# re-fitting seed 0 here would NOT reproduce it anyway (different platform, and
# the sixth, gap_tl_ce_sdice, has no checkpoint left at all). Seeds are
# exchangeable draws, so a fresh SEED=3 is a statistically equivalent third
# member of the mean +/- std; it is simply labelled 3 rather than 0. `report`
# groups on model_name, which carries no seed, so the row merges automatically.
#
# If you would rather these six read {0,1,2} like the other sixteen, bench the
# five local seed-0 checkpoints on MPS instead (~2 h, no cluster time) and skip
# this pool for all but gap_tl_ce_sdice:  ARMS=gap_tl_ce_sdice bash pool_d.sh
#
# WHY RUN_TEST DEFAULTS TO 0 HERE (it is 1 everywhere else)
# --------------------------------------------------------
# On test every arm is currently at exactly n=2 {1,2} — that table is BALANCED.
# Benching seed 3 on test would put these six at n=3 while the other sixteen
# stayed at n=2, i.e. it would fix val's imbalance by creating test's. The store
# is append-only, so adding those rows is the hard direction to undo, whereas
# re-running this pool later with RUN_TEST=1 is cheap (the fits are skipped).
# So: val by default, and opt in when the other sixteen also get a third seed.
#
#   RUN_TEST=1 sbatch ... train.sbatch --SCRIPT=loss/refit/pool_d.sh
#
# Budget: 6 fits (~1.5-2 h each on an L40S) + 6 val benches (~15-20 min each,
# theta* sweep included) ~= 11-14 h serial on one GPU. 24 h leaves headroom;
# a job that dies at the wall clock resumes on re-submit rather than restarting.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARMS="gap_ce wbce wbce_dice gap_tl_ce_lcdice gap_tl_ce_sdice gap_tl_dice"
export NGPU=1
export SEEDS="${SEEDS:-3}"
export RUN_TEST="${RUN_TEST:-0}"   # see header — test is already balanced at n=2

bash "$REPO_DIR/scripts/hpc/loss/refit/run_pool.sh"
