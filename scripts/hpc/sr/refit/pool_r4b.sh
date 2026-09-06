#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=r4b-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# R4b — JOINT SR4RS, BARE (no HC, no splice, pad 0).
# It is the joint SR4RS cell of the constraint x generator grid, the SR4RS twin
# of r2b, and the bare partner of r4a.
#
# MIND THE TAG: r4b_new.sh sets no SR_HC, so the arm runs at the engine default
# (native) and its run dirs carry NO HC tag -- sr_r4b_new_gap_ce_anorm_recalpost,
# with no _nohc, unlike r3b/r2b which force SR_HC=off. The refit script matches
# it deliberately. Do not "fix" one without the other, or the refit seeds stop
# grouping with the tuned seed.
#
# EXPECT GENERATOR DRIFT ON THIS ARM: joint fine-tuning walks bare SR4RS off the
# reflectance scale (a median output shift of about -1.17 was measured here).
# That is the finding, not a fault -- r4a is the constrained comparison. Check
# the fit log's `[joint_sr] ... SR input mean` against the sane 0.05-0.35 band.
#
# Everything for this arm in ONE job: the tuned seed's tune -> fit -> bench,
# then the refit seeds through scripts/hpc/sr/refit/r4b_new_gap_ce.sh.
#
#   sbatch scripts/hpc/sr/refit/pool_r4b.sh
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

export ARM=r4b_new
export REFIT=r4b_new_gap_ce
export LOSS_ARM="${LOSS_ARM:-gap_ce}"

bash "$REPO_DIR/scripts/hpc/sr/refit/run_arm_pool.sh"
