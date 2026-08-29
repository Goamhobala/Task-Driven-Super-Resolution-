#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=r1a-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# R1a — FROZEN SEN2SR-Lite behind the full HC bundle (clamp -> FFT splice ->
# pad 8 with output crop). The `a` cell of the frozen row.
# Everything for this arm in ONE job: the tuned seed's tune -> fit -> bench,
# then the refit seeds through scripts/hpc/sr/refit/r1a_new_gap_ce.sh.
#
#   sbatch scripts/hpc/sr/refit/pool_r1a.sh
#
# No flags: the headers above are the ones you would otherwise type. They apply
# only when this file is submitted DIRECTLY — sbatch reads #SBATCH from the file
# it is given, so routing it through train.sbatch instead ignores them.
#
# --output=slurm-%x-%j.txt, not train.sbatch's slurm-%j.txt: %x is the job name,
# so r1a's and r1b's logs never collide. Still .txt, so Nextcloud renders it.
#
# This arm is ALREADY TUNED, so phase 1 normally starts at the fit. Set
# TUNED_SEED to wherever that tune lives (66 is the series convention, as for
# r0/r2a/r2b) — the pool skips the tune when it finds best_params.yaml there.
#
# 48 h is a budget, not a promise. A frozen-SR refit has no SR backward pass, so
# it is cheaper than the r2 arms' joint fine-tuning but dearer than r0's
# bicubic. If it does not fit, SUBMIT IT AGAIN: every stage is guarded by what
# it would produce, so finished work costs seconds and a half-trained refit
# resumes from last.ckpt.
#
#   TUNED_SEED=0            the tune lives at a seed other than 66
#   STAGES="fit bench"      the tuned seed only, no refit seeds
#   STAGES="refit"          the refit seeds only
#   SEEDS="3 4"             override the refit script's own seed default
#   LR=<value>              skip reading it off the tuned overlay
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARM=r1a_new
export REFIT=r1a_new_gap_ce
export LOSS_ARM="${LOSS_ARM:-gap_ce}"

bash "$REPO_DIR/scripts/hpc/sr/refit/run_arm_pool.sh"
