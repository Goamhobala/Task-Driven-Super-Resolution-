#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=rl4_full-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# RL4-FULL, ALL THREE STAGES IN ONE ALLOCATION: the 30x10 search over
# (lr, lr_sr), then a 100-epoch refit at seeds 444, 666 and 888, each benched.
#
#   sbatch scripts/hpc/sr/rl/full/pool_rl4_full.sh
#
# THIS SCRIPT IS IDEMPOTENT, AND THAT IS THE POINT. It does not assume it can
# finish. Every stage is guarded by what it would produce:
#
#   tune    skipped once best_params.yaml exists
#   fit     fit_state decides: complete -> skip, partial -> RESUME from
#           last.ckpt (no epoch is retrained), absent -> start
#   bench   skipped if the (model, seed, test) row is already in the store
#
# So re-running it is always safe and finished work costs seconds. An r4-class
# joint fit is ~2 days against a 48 h wall clock, so ONE JOB WILL NOT FINISH
# THE ARM -- that is expected, not a failure.
#
# CHAINING IT THROUGH A MAINTENANCE WINDOW
# ----------------------------------------
# Queued jobs keep running and keep starting while the scheduler is closed to
# NEW submissions, so submit everything BEFORE the window and let the links
# hand off to each other.
#
# THE TUNE IS SERIAL, THE THREE REFITS ARE NOT. One search produces one
# overlay; the three seeds then replicate it and are completely independent, so
# they should run side by side on three GPUs rather than queueing behind each
# other. That is four chains: one for the tune, one per seed.
#
#   REPO=$HOME/InstaRoad/InstaRoadPrototype
#   P=$REPO/scripts/hpc/sr/rl/full/pool_rl4_full.sh
#
#   # 1. the search: two links, in case 30x10 does not fit one allocation
#   T=$(STAGES=tune sbatch --parsable "$P")
#   T=$(STAGES=tune sbatch --parsable --dependency=afterany:$T "$P")
#
#   # 2. three INDEPENDENT chains, each hanging off the last tune link
#   for s in 444 666 888; do
#     PREV=$T
#     for _ in $(seq 3); do
#       PREV=$(STAGES=refit SEEDS=$s sbatch --parsable \
#                --dependency=afterany:$PREV -J rl4_full_s$s "$P")
#     done
#   done
#
# sbatch propagates the submitting environment (--export=ALL is the default),
# which is how STAGES and SEEDS reach the script.
#
# WHY afterany, NOT afterok. A link that hits the wall clock mid-fit exits
# non-zero, and that is the NORMAL case here -- afterok would hold the rest of
# the chain exactly when you cannot intervene. The guards make the retry safe:
# a link with nothing left to do exits in seconds.
#
# WHY DEPENDING ON THE *LAST* TUNE LINK IS CORRECT. If the search finished in
# the first link, the second finds best_params.yaml and exits almost
# immediately, so the refits start straight away. If it did not, the second
# link resumes it -- the Optuna study is sqlite:///<run dir>/study.db, so
# completed trials survive a wall-clock kill. Either way the refits begin only
# once an overlay exists. A refit link that somehow starts without one exits 3
# with a clear message rather than fitting an untuned model.
#
# HOW MANY LINKS PER CHAIN. A link covers 48 h:
#   tune    30 trials x 10 epochs, minus whatever MedianPruner kills from
#           epoch 2 on                                              ~1-2 links
#   refit   100 epochs of joint SR4RS. For scale, FROZEN SR4RS (r3b) measured
#           8m18s/epoch = ~14 h for 100; joint adds the SR backward pass, and
#           r4b seed 66 needed three submissions to finish its fit  ~1-3 links
# Hence 2 tune links and 3 per seed. Over-provisioning is nearly free; running
# out mid-fit costs you the window.
#
# KEEP_LAST defaults to 1 on this path, which is what makes the chain work:
# last.ckpt is the resume point, and deleting it would cost days.
#
#   STAGES=tune / STAGES=refit     restrict what a link does
#   SEEDS="444"                    one seed at a time
#   TUNE_SEED=0                    the seed the search runs at
#   FORCE_FIT=1                    discard partial fits and retrain (rarely)
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARM=rl4_full
export SEEDS="${SEEDS:-444 666 888}"

bash "$REPO_DIR/scripts/hpc/sr/rl/full/_rl_full_pool.sh"
