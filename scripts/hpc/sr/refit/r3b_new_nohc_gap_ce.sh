#!/bin/bash
# SEED REFIT — sr_r3b_new_nohc_gap_ce_anorm_recalpost
#
# R3b — FROZEN SR4RS (WGAN-GP generator, PyTorch port), BARE: no positivity
# clamp, no FFT splice, no pad. SR_HC=off puts _nohc into the run dir, the study
# and the bench model_name. It fills the frozen SR4RS cell of the
# constraint x generator grid:
#
#                        b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   frozen  SEN2SR-Lite  r1b_new                  r1a_new
#   frozen  SR4RS        r3b_new  <- THIS ARM     r3a_new
#   joint   SEN2SR-Lite  r2b_new                  r2a_new
#
# Same loss arm (gap_ce) and same tuned seed (66) as r0/r1b/r2b, so the column
# is controlled: identical task loss, identical protocol, ONLY the generator and
# its constraint differ.
#
# Re-runs this arm's FINAL-protocol refit at further seeds so the reported number
# can be a cross-seed mean +/- std rather than a single draw. Nothing else
# changes: same 100-epoch budget, same train+val merge, same theta* sweep on val
# at the end, same store. `model_name` carries no seed, so these rows group with
# the tuned seed's row automatically under `report`.
#
#   cd scripts/hpc
  # sbatch --job-name=refit-r3b_new_nohc_gap_ce --time=48:00:00 \
  #        --gres=gpu:1 --cpus-per-task=8 \
  #        train.sbatch --SCRIPT=sr/refit/r3b_new_nohc_gap_ce.sh
#
# CHANGE THE SEEDS AND NOTHING ELSE:
#   sbatch ... train.sbatch --SCRIPT=sr/refit/r3b_new_nohc_gap_ce.sh  # SEEDS below
#   SEEDS="3 4" sbatch ... train.sbatch --SCRIPT=sr/refit/r3b_new_nohc_gap_ce.sh
#
# WALL CLOCK: A FROZEN refit has NO SR BACKWARD PASS, so this is cheaper than
# the joint r2/r4 arms -- closer to r1b than to r2b. 24 h comfortably holds two
# seeds plus their benches. Re-submit rather than guess: every stage is guarded
# by what it would produce, so finished work costs seconds and a half-trained
# refit resumes from last.ckpt.
#
# Hyperparameters are seed 66's tune (best val_ap=0.5482), copied verbatim from
#   sr_r3b_new_nohc_gap_ce_anorm_recalpost_seed66
# and baked in, so the cluster needs nothing from SRruns/ and the exact config a
# refit used is readable in the file that ran it.
#
# THE EXPORTS BELOW ARE NOT REDUNDANT WITH THE YAML. `_stages_tv.sh` appends
# --model.pstar / --model.gap_r / --model.warmup_start / ... AFTER the --config
# layers, so those flags OVERRIDE best_params.yaml with the engine's ENV
# defaults. Exporting them pins the belt to the same values the overlay carries.
# (tl_theta / gap_theta / pos_weight / mix_w / lr are deliberately kept out of
# the belt by the engine, so the overlay alone is authoritative there.)
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
source "$REPO_DIR/scripts/hpc/sr/refit/_refit_lib.sh"

EXP_TAG="r3b_new"
LOSS_ARM="gap_ce"
SEEDS="${SEEDS:-222 444}"          # <-- the only thing you normally change

# --- this arm's SR treatment -------------------------------------------------
# FROZEN, unlike r2b: the generator is a fixed feature extractor and only the
# UNet trains. SR4RS ships no FFT hard constraint, so SR_HC=off is its NATIVE
# behaviour -- forced explicitly anyway so every `b` cell carries the identical
# _nohc tag and no arm's constraint state is implicit.
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SR_PAD=0
SR_HC="off"
# SR snapshots are pointless with a frozen generator: every epoch's weights are
# byte-identical to the last. 0 = off. Override only if you have a reason.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"

USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
# SR4RS weights live in their OWN model dir. Without this the engine falls back
# to _stages_tv.sh's default (SEN2SRLite_RGBN), which holds no gen_weights.
# safetensors and the fit dies at the pre-flight check. Matches sr/{r3b,r4a,r4b}_new.sh.
export SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

# RUN_TAG must reproduce _stages_tv.sh's RUN_DIR exactly, or the seed lands in a
# directory that does not group with the tuned seed:
#   sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${PROTO_TAG}
RUN_TAG="sr_r3b_new_nohc_gap_ce_anorm_recalpost"
export MODEL_NAME="sr_r3b_new_nohc_gap_ce_anorm_recalpost_ap"   # matches the tuned seed

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

# NOTE: no lr_sr. The engine auto-skips it when FREEZE_SR=true, and the tuned
# overlay carries none -- there is no SR parameter group to give a rate to.
read -r -d '' BEST_PARAMS <<'YAML' || true
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  upsampler: sr4rs
  freeze_sr: true
  sr_pad: 0
  lr: 0.00019182654424406862
  sr_hc: 'off'
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
  sr_hold_epochs: 0.0
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
