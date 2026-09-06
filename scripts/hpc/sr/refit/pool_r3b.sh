#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=r3b-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# R3b — FROZEN SR4RS, BARE (no HC, no clamp, no splice, pad 0). The frozen
# SR4RS cell of the constraint x generator grid, and the SR4RS twin of r1b:
# r3b - r1b isolates the GENERATOR with the read-out held frozen, exactly as
# r2b - r1b isolates joint fine-tuning with the generator held fixed.
# Everything for this arm in ONE job: the tuned seed's tune -> fit -> bench,
# then the refit seeds through scripts/hpc/sr/refit/r3b_new_nohc_gap_ce.sh.
#
#   sbatch scripts/hpc/sr/refit/pool_r3b.sh
#
# No flags: the headers above are the ones you would otherwise type. They apply
# only when this file is submitted DIRECTLY — sbatch reads #SBATCH from the file
# it is given, so routing it through train.sbatch instead ignores them.
#
# --output=slurm-%x-%j.txt, not train.sbatch's slurm-%j.txt: %x is the job name,
# so r3a's and r3b's logs never collide. Still .txt, so Nextcloud renders it.
#
# THE r3 TAG WAS REDEFINED: it used to mean the full (Mamba) SEN2SR. The
# retired r3b_cdngi.sh is a DIFFERENT series (cdngi labels, cdngi store) and
# stays on disk unchanged — do not read the two as the same treatment. The tune
# this pool discovers is the NEW one, at seed 66. Queue behind a running tune
# with:
#
#   sbatch --dependency=afterok:<tune jobid> scripts/hpc/sr/refit/pool_r3b.sh
#
# 48 h is a budget, not a promise. A frozen-SR refit has no SR backward pass, so
# it is cheaper than the r2/r4 arms' joint fine-tuning but dearer than r0's
# bicubic. SR4RS is heavier per forward than SEN2SR-Lite, so expect this to sit
# above r1b even though both are frozen. If it does not fit, SUBMIT IT AGAIN: every stage is guarded by what
# it would produce, so finished work costs seconds and a half-trained refit
# resumes from last.ckpt.
#
# THE TUNED SEED IS DISCOVERED, NOT ASSUMED
# -----------------------------------------
# The pool globs this arm's run dirs for the best_params.yaml `sr.tune` wrote
# and uses whichever seed has it — so it does not need to be told, and it is
# safe to queue with `--dependency=afterok:<tune jobid>` BEFORE the tune has
# finished: the glob runs at job start, by which time the overlay exists.
# Two tuned seeds is an error, not a coin flip (two searches = two different
# hyperparameter sets); pass TUNED_SEED to settle it.
#
# Use `afterok`, not `afterany`: if the tune dies, afterok holds this job,
# whereas afterany would start it, find no overlay, and launch a NEW tune.
#
#   TUNED_SEED=42           override the discovery
#   NEW_TUNE_SEED=42        seed to CREATE a tune at, if none exists at all
#   STAGES="fit bench"      the tuned seed only, no refit seeds
#   STAGES="refit"          the refit seeds only
#   SEEDS="3 4"             override the refit script's own seed default
#                           (must not contain the tuned seed — it is flagged)
#   LR=<value>              skip reading it off the tuned overlay
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARM=r3b_new
export REFIT=r3b_new_nohc_gap_ce
export LOSS_ARM="${LOSS_ARM:-gap_ce}"

bash "$REPO_DIR/scripts/hpc/sr/refit/run_arm_pool.sh"
