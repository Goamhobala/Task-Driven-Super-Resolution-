#!/bin/bash
# SEED REFIT — sr_r0_new_gap_ce_anorm_recalpost
#
# R0 — bicubic x4 upsampling (deterministic baseline). No SR net, so no lr_sr,
# no snapshots, and SR_PAD is irrelevant.
#
# Re-runs this arm's FINAL-protocol refit at further seeds so the reported
# number can be a cross-seed mean +/- std rather than a single draw. Nothing
# else changes: same 100-epoch budget, same train+val merge, same θ* sweep on
# val at the end, same store. `model_name` carries no seed, so these rows group
# with the tuned seed's row automatically under `report`.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-r0_new_gap_ce --time=24:00:00 \
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=sr/refit/r0_new_gap_ce.sh
#
# CHANGE THE SEEDS AND NOTHING ELSE:
#   sbatch ... train.sbatch --SCRIPT=sr/refit/r0_new_gap_ce.sh   # SEEDS defaults below
#   SEEDS="3 4" sbatch ... train.sbatch --SCRIPT=sr/refit/r0_new_gap_ce.sh
#
# 24 h because a 100-epoch joint refit is ~6-9 h on an L40S and a chained job
# that dies at the wall clock loses every seed after the first. Two seeds plus
# their benches fit; three do not — submit those as a second job.
#
# Hyperparameters are seed 66's tune (best val_ap=0.5806), copied verbatim from
#   sr_r0_new_gap_ce_anorm_recalpost_seed66
# and baked in, so the cluster needs nothing from SRruns/ and the exact config a
# refit used is readable in the file that ran it.
#
# THE EXPORTS BELOW ARE NOT REDUNDANT WITH THE YAML. `_stages_tv.sh` appends
# --model.pstar / --model.gap_r / --model.warmup_start / ... AFTER the --config
# layers, so those flags OVERRIDE best_params.yaml with the engine's ENV
# defaults. Exporting them pins the belt to the same values the overlay carries.
# (tl_theta / gap_theta / pos_weight / mix_w / lr / lr_sr are deliberately kept
# out of the belt by the engine, so the overlay alone is authoritative there.)
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
source "$REPO_DIR/scripts/hpc/sr/refit/_refit_lib.sh"

EXP_TAG="r0_new"
LOSS_ARM="gap_ce"
SEEDS="${SEEDS:-42 888}"          # <-- the only thing you normally change

# --- this arm's SR treatment -------------------------------------------------
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0
SR_HC="native"
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"

# RUN_TAG must reproduce _stages_tv.sh's RUN_DIR exactly, or the seed lands in a
# directory that does not group with the tuned seed:
#   sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${PROTO_TAG}
RUN_TAG="sr_r0_new_gap_ce_anorm_recalpost"
export MODEL_NAME="sr_r0_new_gap_ce_anorm_recalpost_ap"   # matches the tuned seed so the store groups them

# --- fit-belt pins (see header) ---------------------------------------------
export PSTAR="gap_ce"
export GAP_R="4"
export GAP_K="60.0"
export TL_ELL="5"
export TVERSKY_ALPHA="0.7"
export CL_ALPHA="0.3"
export CL_ITERS="5"
export SKEL_W="1.0"
export SKEL_RADIUS="1"
export WARMUP_START="30"
export WARMUP_RAMP="10"

read -r -d '' BEST_PARAMS <<'YAML' || true
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  upsampler: bicubic
  freeze_sr: false
  sr_pad: 0
  lr: 0.00033552787012427956
  loss_arm: gap_ce
  pstar: gap_ce
  gap_r: 4
  gap_k: 60.0
  tl_ell: 5
  tl_theta: 0.375
  gap_theta: 0.55836
  tversky_alpha: 0.7
  cl_alpha: 0.3
  cl_iters: 5
  sr_w: 1.0
  sr_radius: 1
  warmup_start: 30
  warmup_ramp: 10
  mix_w: 0.6075946831862098
  pos_weight: 4.77222
  lr_schedule: cosine
  sr_warmup_epochs: 1.0
  l2sp_lambda: 0.0
  adaptive_norm: true
  adaptive_norm_momentum: 0.01
  norm_recalibrate: post
data:
  batch_size: 4
  mask_source: raster
  mask_dirname: mask_new_2pt5
trainer:
  precision: bf16-mixed
YAML

run_seeds
