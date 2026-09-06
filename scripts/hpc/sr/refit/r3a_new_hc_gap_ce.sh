#!/bin/bash
# SEED REFIT — sr_r3a_new_hc_gap_ce_anorm_recalpost
#
# R3a — FROZEN SR4RS (WGAN-GP generator, PyTorch port) WITH the hard-constraint
# bundle mounted: FFT splice on, pad 8. It fills the constrained frozen cell of
# the constraint x generator grid:
#
#                        b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   frozen  SEN2SR-Lite  r1b_new                  r1a_new
#   frozen  SR4RS        r3b_new                  r3a_new  <- THIS ARM
#   joint   SEN2SR-Lite  r2b_new                  r2a_new
#   joint   SR4RS        r4b_new                  r4a_new
#
# *** THIS ARM BORROWS SEN2SR-LITE'S MASK. ***
# SR4RS ships NO hard constraint of its own, so `hc_mask_path` below points at
# SEN2SRLite_RGBN/hard_constraint.safetensor -- the engine refuses to start
# otherwise (_stages_tv.sh:169: SR_HC=on + UPSAMPLER=sr4rs + empty HC_MASK_PATH
# is a hard error). The path is a CLUSTER path; it is correct for sbatch and
# wrong everywhere else, so override HC_MASK_PATH if you ever run this off the
# HPC. The exported HC_MASK_PATH is what actually reaches the fit -- the engine
# appends --model.hc_mask_path AFTER the --config layers, so it wins over the
# key in the overlay. Both are set, to the same path, so neither route surprises.
# The mask was optimised for SEN2SR-Lite, NOT for SR4RS, so any gain here is a
# LOWER BOUND on what a tuned cutoff could give this generator.
#
# WHAT THIS ARM BUYS: r3a - r3b is the constraint's own contribution to a FROZEN
# generator -- no optimisation in the path to absorb or exploit it. r4a - r3a is
# task-driven adaptation on the constrained SR4RS lane, the twin of r2a - r1a on
# the SEN2SR lane.
#
# Re-runs this arm's FINAL-protocol refit at further seeds so the reported number
# can be a cross-seed mean +/- std rather than a single draw. Nothing else
# changes: same 100-epoch budget, same train+val merge, same theta* sweep on val
# at the end, same store. `model_name` carries no seed, so these rows group with
# the tuned seed's row automatically under `report`.
#
#   cd scripts/hpc
  # sbatch --job-name=refit-r3a_new_hc_gap_ce --time=48:00:00 \
  #        --gres=gpu:1 --cpus-per-task=8 \
  #        train.sbatch --SCRIPT=sr/refit/r3a_new_hc_gap_ce.sh
#
# CHANGE THE SEEDS AND NOTHING ELSE:
#   SEEDS="3 4" sbatch ... train.sbatch --SCRIPT=sr/refit/r3a_new_hc_gap_ce.sh
#
# WALL CLOCK: FROZEN, so there is NO SR BACKWARD PASS -- cheaper than the joint
# r4a. But it is NOT as cheap as r3b: the FFT splice runs on every forward and
# pad 8 grows the grid to 576 px, so allow ~1.3x r3b's per-epoch time. Re-submit
# rather than guess: every stage is guarded by what it would produce, so finished
# work costs seconds and a half-trained refit resumes from last.ckpt.
#
# Hyperparameters are the tuned seed's search (best val_ap=0.5590, trial #21),
# copied verbatim and baked in, so the cluster needs nothing from SRruns/ and the
# exact config a refit used is readable in the file that ran it.
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

EXP_TAG="r3a_new"
LOSS_ARM="gap_ce"
SEEDS="${SEEDS:-222 444}"          # <-- the only thing you normally change

# --- this arm's SR treatment -------------------------------------------------
# FROZEN (freeze_sr=true) + the HC bundle. Pad travels with the constraint: the
# FFT splice rings at the patch border, and pad 8 is the margin that gets cropped
# away so the ringing never reaches a scored pixel.
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SR_PAD=8
SR_HC="on"
# SR4RS has no mask of its own -- borrow SEN2SR-Lite's. See the header.
USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
export HC_MASK_PATH="${HC_MASK_PATH:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"
# SR4RS weights live in their OWN model dir. Without this the engine falls back
# to _stages_tv.sh's default (SEN2SRLite_RGBN), which holds no gen_weights.
# safetensors and the fit dies at the pre-flight check. Matches sr/r3a_new.sh.
export SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
# SR snapshots are pointless with a frozen generator: every epoch's weights are
# byte-identical to the last. 0 = off. Override only if you have a reason.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"

# RUN_TAG must reproduce _stages_tv.sh's RUN_DIR exactly, or the seed lands in a
# directory that does not group with the tuned seed:
#   sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${PROTO_TAG}
# SR_HC=on => HC_TAG=_hc.
RUN_TAG="sr_r3a_new_hc_gap_ce_anorm_recalpost"
export MODEL_NAME="sr_r3a_new_hc_gap_ce_anorm_recalpost_ap"   # matches the tuned seed

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
  sr_pad: 8
  lr: 0.00029793627058285826
  sr_hc: 'on'
  hc_mask_path: /scratch/yhxjin001/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor
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
