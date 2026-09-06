#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=r4a-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# R4a — JOINT SR4RS WITH the hard-constraint bundle (FFT splice on, pad 8).
# It is the last cell of the constraint x generator grid, and the controlled
# partner of r4b: same generator, same loss, same protocol, the CONSTRAINT the
# only difference. That contrast is the whole point of the arm -- joint
# fine-tuning walks bare SR4RS off the reflectance scale (r4b measured a median
# output shift of about -1.17), and the FFT constraint pins the band means, so
# r4a - r4b measures whether the constraint rescues joint SR4RS.
#
# SR4RS SHIPS NO MASK OF ITS OWN. r4a_new.sh defaults HC_MASK_PATH to
# SEN2SRLite_RGBN/hard_constraint.safetensor on scratch, and the engine hard-
# errors without it (_stages_tv.sh:169). If this pool dies immediately with an
# SR_HC/HC_MASK_PATH error, that scratch file is what is missing -- not a config
# bug here.
#
# Everything for this arm in ONE job: the tuned seed's tune -> fit -> bench,
# then the refit seeds through scripts/hpc/sr/refit/r4a_new_hc_gap_ce.sh.
#
#   sbatch scripts/hpc/sr/refit/pool_r4a.sh
#
# No flags: the headers above are the ones you would otherwise type. They apply
# only when this file is submitted DIRECTLY -- sbatch reads #SBATCH from the file
# it is given, so routing it through train.sbatch instead ignores them.
#
# --output=slurm-%x-%j.txt, not train.sbatch's slurm-%j.txt: %x is the job name,
# so r4a's and r4b's logs never collide.
#
# 48 h is a budget, not a promise. THE r4 ARMS ARE THE DEAREST IN THE GRID: a
# joint refit puts an SR backward pass through SR4RS, which is heavier per
# forward than SEN2SR-Lite, so budget above r2 and well above the frozen r1/r3
# arms. Two seeds plus their benches is the realistic ceiling. If it does not
# fit, SUBMIT IT AGAIN: every stage is guarded by what it would produce, so
# finished work costs seconds and a half-trained refit resumes from last.ckpt.
#
# THE TUNED SEED IS DISCOVERED, NOT ASSUMED
# -----------------------------------------
# The pool globs this arm's run dirs for the best_params.yaml `sr.tune` wrote
# and uses whichever seed has it -- so it does not need to be told, and it is
# safe to queue with `--dependency=afterok:<tune jobid>` BEFORE the tune has
# finished: the glob runs at job start, by which time the overlay exists.
# Two tuned seeds is an error, not a coin flip (two searches = two different
# hyperparameter sets); pass TUNED_SEED to settle it.
#
# The pool also reads lr off that overlay and checks it against the value baked
# into the refit script. THIS ARM'S TUNE IS ALREADY DONE and its overlay is
# baked in verbatim, so a mismatch warning here means the overlay on scratch is
# not the one that was pasted -- stop and reconcile before letting seeds run.
#
# Use `afterok`, not `afterany`: if the tune dies, afterok holds this job,
# whereas afterany would start it, find no overlay, and launch a NEW tune.
#
#   TUNED_SEED=42           override the discovery
#   NEW_TUNE_SEED=42        seed to CREATE a tune at, if none exists at all
#   STAGES="fit bench"      the tuned seed only, no refit seeds
#   STAGES="refit"          the refit seeds only
#   SEEDS="3 4"             override the refit script's own seed default
#                           (must not contain the tuned seed -- it is flagged)
#   LR=<value>              skip reading it off the tuned overlay
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export ARM=r4a_new
export REFIT=r4a_new_hc_gap_ce
export LOSS_ARM="${LOSS_ARM:-gap_ce}"

bash "$REPO_DIR/scripts/hpc/sr/refit/run_arm_pool.sh"
