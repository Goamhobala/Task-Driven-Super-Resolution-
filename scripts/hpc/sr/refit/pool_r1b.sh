#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=r1b-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# R1b — FROZEN SEN2SR-Lite with the HARD CONSTRAINT OFF (bare generator: no
# clamp, no splice, no pad). The `b` cell of the frozen row, and the frozen
# twin of r2b: r2b - r1b is the exact counterpart of r2a - r1a.
# Everything for this arm in ONE job: the tuned seed's tune -> fit -> bench,
# then the refit seeds through scripts/hpc/sr/refit/r1b_new_nohc_gap_ce.sh.
#
#   sbatch scripts/hpc/sr/refit/pool_r1b.sh
#
# No flags: the headers above are the ones you would otherwise type. They apply
# only when this file is submitted DIRECTLY — sbatch reads #SBATCH from the file
# it is given, so routing it through train.sbatch instead ignores them.
#
# --output=slurm-%x-%j.txt, not train.sbatch's slurm-%j.txt: %x is the job name,
# so r1a's and r1b's logs never collide. Still .txt, so Nextcloud renders it.
#
# THIS ARM WAS REDEFINED (2026-08-28) and has NO TUNE YET. The old r1b overlay
# is a different operator and is not even on the path (_nohc retags the run
# dir), so phase 1 starts at the tune — budget for it on top of the refits.
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

export ARM=r1b_new
export REFIT=r1b_new_nohc_gap_ce
export LOSS_ARM="${LOSS_ARM:-gap_ce}"

bash "$REPO_DIR/scripts/hpc/sr/refit/run_arm_pool.sh"
