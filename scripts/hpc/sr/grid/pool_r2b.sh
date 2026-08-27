#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=r2bgrid-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# THE HC-OFF LANE of the lr_sr mechanism grid: r2b's configuration (the bare
# SEN2SR generator — no positivity clamp, no frequency splice, SR_PAD=0) at all
# four pinned lr_sr values. docs/lrsr_grid_ablation_plan.md — the r2a lane is
# pool_r2a.sh.
#
#   sbatch scripts/hpc/sr/grid/pool_r2b.sh
#
# No flags: the headers above are the ones you would otherwise type. Note that
# they only apply when this file is submitted DIRECTLY — sbatch reads #SBATCH
# from the file it is given, so routing it through train.sbatch instead
# (`sbatch --job-name=... --time=48:00:00 --gres=gpu:1 --cpus-per-task=8 \
#   scripts/hpc/train.sbatch --SCRIPT=sr/grid/pool_r2b.sh`) ignores them and you
# are back to passing the flags yourself. Everything train.sbatch does for an
# arm — export the config, exec the script — this pool does for four cells.
#
# --output=slurm-%x-%j.txt, not train.sbatch's slurm-%j.txt: the whole point of
# two lanes is telling them apart, and %x is the job name. Still .txt, so
# Nextcloud renders it.
#
# WHAT IT RUNS
# ------------
# Four cells x (tune -> fit -> bench), sequentially on one GPU, extremes first
# per §8: 1e-4 and 1e-5 carry the interaction and the probable collapses; the
# 1e-6/1e-7 cells interpolate toward the formal r2b arm and are the least
# informative, so they run last and are the ones a timeout drops.
#
# 48 h is a budget, not a promise. Per cell is roughly a 15x10 lr tune (pruned,
# cheap) plus a 2-3 h refit, so the lane usually fits — but if it does not, just
# SUBMIT IT AGAIN: every stage is guarded by what it would produce, so finished
# cells cost seconds and a half-trained refit resumes from last.ckpt.
#
#   LRSRS="1e-4"           only that cell
#   STAGES="tune"          all four tunes first, fits later
#   SEED=1                 a replicate seed for a cell that carries a sentence
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export HC=off
export LRSRS="${LRSRS:-1e-4 1e-5 1e-6 1e-7}"

bash "$REPO_DIR/scripts/hpc/sr/grid/run_pool.sh"
