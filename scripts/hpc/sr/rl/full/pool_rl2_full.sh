#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=rl2_full-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# RL2-FULL END TO END: a 30x10 search over (lr, lr_sr), then a 100-epoch refit at
# seeds 444, 666 and 888. The linear probe given the R-series budget, so the
# ladder's weak scores can be attributed to the READ-OUT rather than to the
# budget it never got.
#
#   sbatch scripts/hpc/sr/rl/full/pool_rl2_full.sh
#
# WHY rl2 CHAINS AND rl4 DOES NOT. SEN2SR-Lite is 187 K trainable parameters
# against SR4RS's 11.3 M, so this arm's search and three refits plausibly fit in
# one 48 h allocation. rl4_full does not, and has no pool at all -- its stages
# are submitted one job at a time (see rl4_full.sh).
#
# WALL CLOCK, HONESTLY. SEN2SR-Lite is 187 K trainable parameters and the
# Studio measured ~2m45s/epoch on an L4, so a 100-epoch refit is a few hours
# and the 30x10 search (pruned) is comparable.
# 48 h will very likely NOT cover the search plus three refits. That is
# expected: SUBMIT IT AGAIN. The tune is skipped once best_params.yaml exists,
# a finished seed costs seconds, and a half-trained one resumes from last.ckpt.
# Splitting is tidier if the queue is generous:
#
#   STAGES=tune                    the search alone
#   STAGES=refit SEEDS="444"       one seed at a time
#
#   TUNE_SEED=0     the seed the search runs at (its own fit is the tune's)
#   SEEDS="..."     override the three refit seeds
#   REFIT_EPOCHS=   shorter refits
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARM=rl2_full
export SEEDS="${SEEDS:-444 666 888}"

bash "$REPO_DIR/scripts/hpc/sr/rl/full/_rl_full_pool.sh"
