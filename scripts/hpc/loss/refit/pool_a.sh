#!/bin/bash
# Refit pool a — 14 arm(s) on 2 GPU(s), all in ONE SLURM job.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-pool-a --time=48:00:00 --qos=l40sfree\
#          --gres=gpu:2 --cpus-per-task=16 \
#          train.sbatch --SCRIPT=loss/refit/pool_a.sh
#
# One arm per GPU at a time (NOT DDP — see run_pool.sh for why batch-8 makes
# that a protocol requirement, not a preference). Every per-arm script skips
# finished fits and already-benched rows, so re-submitting after a timeout
# resumes where it stopped instead of retraining.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARMS="gap_ce gap_t2_ce_dice gap_t2_ce_lcdice gap_t2_ce_sdice gap_t4_ce gap_tl_ce gap_tl_ce_sdice gap_tl_dice gapt4_pstar_lcdice gapt4_pstar_sdice sdice t2_ce tl_ce wbce"
export NGPU=2
export SEEDS="${SEEDS:-1 2}"

bash "$REPO_DIR/scripts/hpc/loss/refit/run_pool.sh"
