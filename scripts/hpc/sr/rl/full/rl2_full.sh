#!/bin/bash
# RL2-FULL — the LINEAR-PROBE arm given the R-SERIES BUDGET.
#
# THE QUESTION THIS ANSWERS. The rl ladder pins lr_sr per rung, runs 30 epochs,
# uses wBCE, and never searches anything -- by design, because it measures a
# DOSE-RESPONSE, not a best achievable score. So it cannot answer "is the linear
# probe weak because a linear read-out is weak, or because it was never given
# the budget the U-Net arms got?" This arm removes that confound: same head,
# same generator, same bare lane as the rl2 ladder,
# but the FULL r2/r4 treatment -- gap_ce, a real 30x10 search over (lr, lr_sr),
# a 100-epoch refit, three seeds.
#
#   STAGE=tune  then  STAGE=fit  then  STAGE=bench
#   Easier: sbatch scripts/hpc/sr/rl/full/pool_rl2_full.sh   (all three, chained)
#
# DISJOINT BY CONSTRUCTION. EXP_TAG is rl2_full, not rl2_new<rung>, so run dirs,
# Optuna studies and store rows can never collide with the ladder's. The two
# are answering different questions and must not be averaged together.
#
# WHAT IS DELIBERATELY *NOT* COPIED FROM THE LADDER
# -------------------------------------------------
#   lr_sr        the ladder PINS it per rung; here it is SEARCHED, which is the
#                whole point -- the budget includes choosing the rate.
#   lr           the ladder pins it at the head lr (3e-3); here it is searched
#                over the engine's default decade range.
#   loss         wbce -> gap_ce, so this arm sits on the same loss surface as
#                every R-series arm it will be read against.
#   epochs       30 -> 100 refit, 10 per tune trial.
#   protocol     train-only -> train+val (FINAL), so PROTO_TAG is empty and the
#                row is comparable with r2/r4 rather than with a holdout run.
#
# WHAT IS KEPT: HEAD=linear (that IS the arm), the bare lane (SR_HC=off,
# SR_PAD=0), adaptive norm with post recalibration, and the loosened rails --
# a linear probe over a jointly-tuned generator is exactly the setting where
# the production band would fire, and killing the run would delete the
# measurement.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"

EXP_TAG="${EXP_TAG:-rl2_full}"
LABELS="new"

# --- the arm -----------------------------------------------------------------
HEAD="${HEAD:-linear}"          # the treatment: a 1x1 conv read-out, no U-Net
UPSAMPLER="sen2sr"
FREEZE_SR="false"                # joint: the generator trains with the head
SR_HC="${SR_HC:-off}"
SR_PAD="${SR_PAD:-0}"

# The generator moves and this arm exists to watch what that does to it.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-1}"

# --- the budget: identical to r2/r4 -----------------------------------------
N_TRIALS="${N_TRIALS:-20}"
TUNE_EPOCHS="${TUNE_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
BATCH_SIZES="${BATCH_SIZES:-4}"   # between-arm constant of the SR series
SEARCH_GPUS="${SEARCH_GPUS:-1}"
REFIT_GPUS="${REFIT_GPUS:-1}"

# --- the search space, narrowed and SHIFTED UP -------------------------------
# lr     [1e-4, 1e-2]  (engine default [1e-5, 1e-2]) -- drops the bottom decade.
# lr_sr  [1e-5, 1e-2]  (engine default [1e-7, 1e-4]) -- two decades instead of
#                      three, but moved UP by two.
#
# THIS IS NOT THE R-SERIES lr_sr SPACE, and that is deliberate. r2/r4 search
# [1e-7, 1e-4] and land near its floor (r2b 4.46e-7, r4b 2.85e-5): with a U-Net
# doing the reading, the generator barely has to move. A linear read-out cannot
# compensate for a generator that stays put, so the useful rates live where the
# rl ladder found them -- its rungs are 1e-3 to 1e-6 -- and searching the
# R-series range would spend most of 30 trials below anything this arm can use.
#
# CONSEQUENCE FOR THE COMPARISON: the budget (30 x 10 trials, 100-epoch refit,
# gap_ce, FINAL protocol) matches r2/r4 exactly; the lr_sr SEARCH SPACE does
# not. So "the probe was given the same budget" is fair, while "the probe
# searched the same space" is not -- say the former in the write-up.
export LR_MIN="${LR_MIN:-1e-4}"
export LR_MAX="${LR_MAX:-1e-2}"
export LR_SR_MIN="${LR_SR_MIN:-1e-5}"
export LR_SR_MAX="${LR_SR_MAX:-1e-2}"

# --- the loss: the R-series frozen control, verbatim ------------------------
# Not re-searched per arm -- that is the R-series rule, and re-searching here
# would confound the head with a different loss.
LOSS_ARM="${LOSS_ARM:-gap_ce}"
export PSTAR="${PSTAR:-gap_ce}"
export GAP_R="${GAP_R:-4}"
export GAP_K="${GAP_K:-60.0}"
export TL_ELL="${TL_ELL:-5}"
export TL_THETA="${TL_THETA:-0.375}"
export GAP_THETA="${GAP_THETA:-0.55836}"
export MIX_W="${MIX_W:-0.6075946831862098}"
export TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
export CL_ALPHA="${CL_ALPHA:-0.3}"
export CL_ITERS="${CL_ITERS:-5}"
export SKEL_W="${SKEL_W:-1.0}"
export SKEL_RADIUS="${SKEL_RADIUS:-1}"
export WARMUP_START="${WARMUP_START:-30}"
export WARMUP_RAMP="${WARMUP_RAMP:-10}"
export SEARCH_THETAS="${SEARCH_THETAS:-false}"
export SEARCH_MIX_W="${SEARCH_MIX_W:-false}"
export POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-4.77222}"
export POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-4.77222}"

# --- protocol: the FINAL one, matching r2/r4 --------------------------------
export TRAIN_SPLITS="${TRAIN_SPLITS:-train val}"   # merge_val=1, no _holdout tag
export MONITOR="${MONITOR:-val_ap}"
export ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
export ADAPTIVE_NORM_M="${ADAPTIVE_NORM_M:-0.01}"
export NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"
export BENCH_SPLIT="${BENCH_SPLIT:-test}"
export SWEEP_CRITERION="${SWEEP_CRITERION:-iou}"
export AP_BINS="${AP_BINS:-101}"
export TILE_METRICS="${TILE_METRICS:-apls,cldice}"

# --- rails: loosened, not disabled ------------------------------------------
# Same reasoning as the lr_sr grid: let the run develop and read the envelope
# post hoc from the logged adapt_std_b* curves. Do NOT reach for
# adaptive_norm_check_every=0 -- that silences the warn stream too, i.e.
# deletes the measurement.
export STD_BAND_RAISE_LO="${STD_BAND_RAISE_LO:-0.01}"
export STD_BAND_RAISE_HI="${STD_BAND_RAISE_HI:-100}"
export STD_BAND_ACTION="${STD_BAND_ACTION:-warn}"

export WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_rl_fullbudget}"
export WANDB_NAME="${WANDB_NAME:-${EXP_TAG}_seed${SEED:-0}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-rl2_full}"

echo "=== RL2-FULL: linear probe, SEN2SR-Lite, joint, gap_ce, ${N_TRIALS}x${TUNE_EPOCHS} tune / ${REFIT_EPOCHS}-epoch refit ==="

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
