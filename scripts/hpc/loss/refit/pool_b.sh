#!/bin/bash
# Refit pool b — 6 arm(s) on 1 GPU(s), all in ONE SLURM job.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-pool-b --time=48:00:00 --qos=l40sfree\
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=loss/refit/pool_b.sh
#
# One arm per GPU at a time (NOT DDP — see run_pool.sh for why batch-8 makes
# that a protocol requirement, not a preference). Every per-arm script skips
# finished fits and already-benched rows, so re-submitting after a timeout
# resumes where it stopped instead of retraining.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARMS="gap_t2_ce gap_t2t4_ce gap_tl_ce_lcdice gapt4_pstar_dice lcdice t4_ce"
export NGPU=1
export SEEDS="${SEEDS:-1 2}"

bash "$REPO_DIR/scripts/hpc/loss/refit/run_pool.sh"
