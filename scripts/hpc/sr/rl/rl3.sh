#!/bin/bash
# RL3 — FROZEN SR4RS x4 under a LINEAR PROBE read-out, BARE. HPC (l40s).
# Cluster twin of scripts/LightningStudio/sr/rl/rl3.sh: same campaign constants
# (_rl_common.sh), same engine (generated from this one), same store tags —
# EXCEPT that this arm early-stops (see below), which its _es5 tag records.
#
# The write-up ladder's r3 is the run-tag r5 (tags never change; the store is
# append-only and the thesis carries one mapping table). rl3 is its probe twin:
# rl3 - rl0 = the separability a frozen WGAN-GP generator adds.
#
# COST WARNING. "Frozen" does not mean cheap here. The probe is 5 parameters,
# but the arm still runs an 11.3 M-param SR4RS forward at 512 px on every batch,
# so rl3 costs roughly what r5 costs — replacing the decoder saves the U-Net's
# forward+backward, not the SR front-end, and the front-end dominates. It is
# also I/O-bound (five trainable parameters), so ask for CPUs: 8+.
#
# SR4RS ships no FFT hard constraint, so SR_HC=off is its native behaviour;
# it is forced explicitly anyway so that all five arms carry the same _nohc tag
# and no arm's constraint state is implicit.
#
# EARLY STOPPING — ON, patience 5 on val_ap (the campaign's own monitor).
# The Studio arms run the fixed 30-epoch budget; this one stops when the probe
# plateaus, which the campaign PRE-REGISTERED as happening before epoch 10
# ("frozen val-AP must plateau before epoch 10", plan §6.1). It is affordable
# here for the same reason it is safe: a frozen generator cannot drift, so
# nothing this arm measures is still changing after the plateau — the only
# thing the remaining epochs buy is queue time on the most expensive frozen arm
# in the series.
#
# Three consequences, all of them stated rather than discovered:
#   1. the budget is a CEILING for this arm, not the between-arm constant it is
#      everywhere else. `rl4 - rl3` therefore compares a 30-epoch joint run
#      against a stopped-at-plateau frozen one. Defensible only while the
#      plateau claim holds — CHECK IT on the logged val_ap curve, and if the
#      arm was still climbing when it stopped, rerun it with FIT_EARLY_STOP=0
#      rather than reporting the contrast.
#   2. the cosine does not complete: the run ends part-way down the schedule at
#      a non-zero lr, instead of at lr 0 like every other arm.
#   3. rows land under ..._es5_holdout_..., which is what keeps them out of the
#      Studio's fixed-budget rl3 rows. Deliberate: they are not the same
#      protocol and must not be averaged.
# FIT_EARLY_STOP=0 at submit time reverts all three and reproduces the Studio
# arm exactly (its rows then merge with the Studio's, which is the point).
#
# Prerequisite: gen_*.{safetensors,json,npz} in SEN2SR_DIR. Parity-verify the
# port locally first (`python -m sr.sr4rs_torch`) — there is no TF on the
# cluster.
#
#   S=sr/rl/rl3.sh
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 \
#          scripts/hpc/train.sbatch --SCRIPT=$S STAGE=tune    # 1x1: pin + timing
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=24:00:00 \
#          scripts/hpc/train.sbatch --SCRIPT=$S STAGE=fit
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=04:00:00 \
#          scripts/hpc/train.sbatch --SCRIPT=$S STAGE=bench
#
# Read h/epoch and peak VRAM off STAGE=tune (one trial, one epoch — the plan's
# §4 gate-1 measurement) before committing the fit's walltime.
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"

EXP_TAG="${EXP_TAG:-rl3_new}"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

# The one place this arm departs from its Studio twin (see EARLY STOPPING).
# Set for EVERY stage: ES_TAG is in the run dir, so a tune tagged one way and a
# fit the other would look for best_params.yaml in a directory that does not
# exist.
FIT_EARLY_STOP="${FIT_EARLY_STOP:-1}"
ES_PATIENCE="${ES_PATIENCE:-5}"
ES_MONITOR="${ES_MONITOR:-$MONITOR}"   # val_ap — never val_iou@0.5 (probe doc §5.1)
ES_MODE="${ES_MODE:-max}"

source "$RL_DIR/../_stages_tv.sh"
