#!/bin/bash
# SEED REFIT — sr_r4b_new_gap_ce_anorm_recalpost
#
# R4b — COLD JOINT fine-tuning of SR4RS (WGAN-GP generator, PyTorch port), BARE:
# no FFT splice, no pad. It fills the joint SR4RS cell of the constraint x
# generator grid:
#
#                        b: bare (pad 0)          a: HC bundle (pad 8)
#   frozen  SEN2SR-Lite  r1b_new                  r1a_new
#   frozen  SR4RS        r3b_new                  r3a_new
#   joint   SEN2SR-Lite  r2b_new                  r2a_new
#   joint   SR4RS        r4b_new  <- THIS ARM     r4a_new
#
# NOTE THE TAG: r4b_new.sh sets NO SR_HC, so the engine's default (native)
# applies and HC_TAG is EMPTY -- this arm's run dir is
# sr_r4b_new_gap_ce_anorm_recalpost, with no _nohc. That is deliberate and it
# differs from r3b/r2b, which force SR_HC=off and therefore carry _nohc. Setting
# SR_HC=off here would retag the run dir, the study AND the bench model_name,
# and the seeds would no longer group with the tuned seed's row.
#
# *** KNOWN HAZARD FOR THIS ARM: GENERATOR DRIFT. ***
# Joint fine-tuning walks SR4RS off the reflectance scale -- a median output
# shift of about -1.17 was measured on this arm, with the range stretched ~33x.
# SEN2SR's FFT hard constraint pins the band means and prevents exactly this,
# which is why r2b does not show it and why r4a (the same generator WITH the
# constraint mounted) is the controlled comparison. Nothing here corrects the
# drift; it is a property of the arm being measured. Check the fit log's
# `[joint_sr] ... SR input mean` line against the sane 0.05-0.35 reflectance
# band before trusting a seed, and expect adaptive_norm to be absorbing a lot.
#
# Re-runs this arm's FINAL-protocol refit at further seeds so the reported number
# can be a cross-seed mean +/- std rather than a single draw. Nothing else
# changes: same 100-epoch budget, same train+val merge, same theta* sweep on val
# at the end, same store. `model_name` carries no seed, so these rows group with
# the tuned seed's row automatically under `report`.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-r4b_new_gap_ce --time=48:00:00 \
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=sr/refit/r4b_new_gap_ce.sh
#
# CHANGE THE SEEDS AND NOTHING ELSE:
#   SEEDS="3 4" sbatch ... train.sbatch --SCRIPT=sr/refit/r4b_new_gap_ce.sh
#
# WALL CLOCK: joint SR4RS is the DEAREST cell of the grid -- an SR backward pass
# through the heavier generator. Budget above r2b, well above the frozen r1/r3
# arms. Re-submit rather than guess: every stage is guarded by what it would
# produce, so finished work costs seconds and a half-trained refit resumes from
# last.ckpt.
#
# Hyperparameters are the tuned seed's search (best val_ap=0.5946, trial #34,
# alpha = lr_sr/lr = 1.50e-01), copied verbatim and baked in, so the cluster
# needs nothing from SRruns/ and the exact config a refit used is readable in
# the file that ran it.
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

EXP_TAG="r4b_new"
LOSS_ARM="gap_ce"
SEEDS="${SEEDS:-222 444}"          # <-- the only thing you normally change

# --- this arm's SR treatment -------------------------------------------------
# JOINT (freeze_sr=false): the generator trains with the UNet. SR_HC is left at
# the engine default (native) to match r4b_new.sh -- see the tag note above.
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0
SR_HC="native"
# Snapshots matter HERE, unlike the frozen arms: the generator is moving, and
# these are the only record of how far it drifted. Keep them.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
# SR4RS weights live in their OWN model dir. Without this the engine falls back
# to _stages_tv.sh's default (SEN2SRLite_RGBN), which holds no gen_weights.
# safetensors and the fit dies at the pre-flight check. Matches sr/{r3b,r4a,r4b}_new.sh.
export SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

# RUN_TAG must reproduce _stages_tv.sh's RUN_DIR exactly, or the seed lands in a
# directory that does not group with the tuned seed:
#   sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${PROTO_TAG}
# HC_TAG is empty for SR_HC=native, hence no _nohc here.
RUN_TAG="sr_r4b_new_gap_ce_anorm_recalpost"
export MODEL_NAME="sr_r4b_new_gap_ce_anorm_recalpost_ap"   # matches the tuned seed

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
  upsampler: sr4rs
  freeze_sr: false
  sr_pad: 0
  lr: 0.0001905599397870636
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
  lr_sr: 2.854252979457172e-05
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
