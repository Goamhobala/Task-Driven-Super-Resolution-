#!/bin/bash
# Refit pool b — 7 arm(s) on 1 GPU(s), all in ONE SLURM job.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-pool-b --time=24:00:00 --qos=l40sfree \
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=loss/refit/pool_b.sh
#
# --ntasks=1 --cpus-per-task=8 requests 8 cores while staying inside an
# 8-core-per-task cap. The batch script still runs ONCE (train.sbatch execs it
# rather than srun-ing it); the extra tasks exist to widen the CPU allocation.
# run_pool.sh reads SLURM_CPUS_ON_NODE — the whole allocation, not the per-task
# slice — and divides it across the lanes, so dataloader workers are not
# oversubscribed. It prints what it saw; if SLURM_CPUS_ON_NODE comes back as 8
# rather than 8, the site binds the batch step to one task's cores and the
# lanes should be launched with `srun --exclusive -n1 -c8` instead.
#
# One arm per GPU at a time (NOT DDP — see run_pool.sh for why batch-8 makes
# that a protocol requirement, not a preference). Every per-arm script skips
# finished fits and already-benched rows, so re-submitting after a timeout
# resumes where it stopped instead of retraining.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARMS="gap_ce gap_t2_ce_lcdice gap_t4_ce gap_tl_ce_sdice gapt4_pstar_lcdice sdice tl_ce"
export NGPU=1
export SEEDS="${SEEDS:-1 2}"

bash "$REPO_DIR/scripts/hpc/loss/refit/run_pool.sh"
