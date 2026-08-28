#!/bin/bash
# SEED REFIT — sr_r1b_new_nohc_gap_ce_anorm_recalpost
#
# R1b — FROZEN SEN2SR-Lite with the FFT HARD CONSTRAINT OFF (bare generator:
# no positivity clamp, no frequency splice, no pad). The SR net runs forward
# only; no lr_sr, no gradients into the generator, no snapshots. SR_HC=off puts
# _nohc into the run dir, the study and the bench model_name.
#
# *** THIS ARM WAS REDEFINED (2026-08-28) *** — r1b_new used to mean "frozen
# SEN2SR + HC, pad 0". Legacy sr_r1b_new_* rows WITHOUT _nohc are the old arm;
# the tag keeps them apart in the append-only store, but they must still be
# flagged as retired in the analysis config, exactly as r2b's were.
#
# Same loss arm and protocol as r0/r1a/r2a/r2b, so the four SEN2SR cells form
# one 2x2 — identical task loss, identical protocol, only the operator and
# whether it is trained differ:
#
#                 b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   frozen        r1b_new  <- THIS ARM     r1a_new
#   cold joint    r2b_new                  r2a_new
#
#   r2b - r1b  what adaptation writes into the BARE generator — the exact twin
#              of r2a - r1a, so the two differences are comparable rather than
#              measured against two different frozen baselines.
#   r1a - r1b  what the constraint does with NO optimisation in the path.
#
# Re-runs this arm's FINAL-protocol refit at further seeds so the reported
# number can be a cross-seed mean +/- std rather than a single draw. Nothing
# else changes: same 100-epoch budget, same train+val merge, same θ* sweep on
# val at the end, same store. `model_name` carries no seed, so these rows group
# with the tuned seed's row automatically under `report`.
#
#   cd scripts/hpc
#   sbatch --job-name=refit-r1b_new_nohc_gap_ce --time=24:00:00 \
#          --gres=gpu:1 --cpus-per-task=8 \
#          train.sbatch --SCRIPT=sr/refit/r1b_new_nohc_gap_ce.sh
#
# CHANGE THE SEEDS AND NOTHING ELSE:
#   sbatch ... train.sbatch --SCRIPT=sr/refit/r1b_new_nohc_gap_ce.sh   # SEEDS defaults below
#   SEEDS="3 4" sbatch ... train.sbatch --SCRIPT=sr/refit/r1b_new_nohc_gap_ce.sh
#
# 24 h: a frozen-SR refit is cheaper than r2b's joint fine-tuning (no SR
# backward pass) but dearer than r0's bicubic, since SEN2SR still runs forward
# every batch. Two seeds plus their benches fit; a chained job that dies at the
# wall clock loses every seed after the first, so submit a third as its own job.
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

EXP_TAG="r1b_new"
LOSS_ARM="gap_ce"
SEEDS="${SEEDS:-1 2}"          # <-- the only thing you normally change

# --- this arm's SR treatment -------------------------------------------------
UPSAMPLER="sen2sr"
FREEZE_SR="true"               # forward only: no lr_sr in the overlay below
# The bare lane: no FFT splice means no Gibbs ringing at the patch border, so
# there is no artifact for the pad to mitigate. Pad travels with the constraint.
SR_PAD=0
SR_HC="off"                    # <-- the treatment; also what tags the run _nohc
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"   # frozen weights never change

# RUN_TAG must reproduce _stages_tv.sh's RUN_DIR exactly, or the seed lands in a
# directory that does not group with the tuned seed:
#   sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${PROTO_TAG}
# Verified against the engine itself:
#   LOSS_ARM=gap_ce SEED=1 PRINT_RUN_DIR=1 bash scripts/hpc/sr/r1b_new.sh
RUN_TAG="sr_r1b_new_nohc_gap_ce_anorm_recalpost"
export MODEL_NAME="sr_r1b_new_nohc_gap_ce_anorm_recalpost_ap"   # matches the tuned seed

# norm_recalibrate stays `post`, NOT the `pre` that sr.tune's help calls "the
# complete fix for frozen-SR arms". The refit must run what the tune ran: the
# tuned seed used the engine default (post), `pre` would retag the run dir
# `_recalpre`, and this arm's rows would stop grouping with it. Change it only
# as a deliberate new arm, never here.

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

# --- THE TUNED lr IS NOT BAKED IN YET ----------------------------------------
# This arm was REDEFINED on 2026-08-28, so there is no tune for it yet at all —
# the old r1b's overlay belongs to a different operator and must not be reused.
# Run STAGE=tune on r1b_new.sh first, then read its lr off the tuned seed:
#   grep '^  lr:' /scratch/$USER/InstaRoad/runs/sr_r1b_new_nohc_gap_ce_anorm_recalpost_seed<TUNED>/best_params.yaml
#
# Then either pass it once —  LR=<value> sbatch ... --SCRIPT=sr/refit/r1b_new_nohc_gap_ce.sh
# or, better, paste it into the YAML in place of __LR__, delete this block, and
# note the tuned seed and its best val_ap in the header the way r2b does. The
# whole point of baking it in is that the config a refit used is readable in the
# file that ran it, with no lookup and no drift.
LR="${LR:-}"
if [ -z "$LR" ]; then
  echo "ERROR: r1b_new_nohc_gap_ce.sh has no tuned lr baked in yet." >&2
  echo "  This arm was redefined (bare generator); the OLD r1b tune is a" >&2
  echo "  different treatment and must not be reused. Tune it first:" >&2
  echo "    sbatch ... train.sbatch --SCRIPT=sr/r1b_new.sh STAGE=tune SEED=<n> LOSS_ARM=gap_ce" >&2
  echo "  then:" >&2
  echo "    grep '^  lr:' \${RUNS_ROOT:-/scratch/\$USER/InstaRoad/runs}/${RUN_TAG}_seed<TUNED>/best_params.yaml" >&2
  echo "  and re-run with LR=<value>, or paste it into the YAML (preferred)." >&2
  exit 2
fi

# Unlike r0/r2a/r2b this heredoc is QUOTED and patched below, so the placeholder
# cannot be accidentally expanded away by a stray $ in a future edit.
read -r -d '' BEST_PARAMS <<'YAML' || true
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  upsampler: sen2sr
  freeze_sr: true
  sr_pad: 0
  lr: __LR__
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
BEST_PARAMS="${BEST_PARAMS/__LR__/$LR}"
echo "### ${RUN_TAG}: lr=${LR}  seeds='${SEEDS}'  sr_hc=off sr_pad=0 (frozen bare generator)"

run_seeds
