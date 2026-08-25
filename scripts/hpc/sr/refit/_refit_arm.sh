#!/bin/bash
# ONE stage of ONE seed of an R-SERIES seed refit — the inner runner every
# per-arm script in this directory invokes as a SUBPROCESS.
#
# Sibling of loss/refit/_refit_arm.sh, and a subprocess for the same reason:
# `sr/_stages_tv.sh` calls `exit` on several paths (the SKIP_TEST guard, the
# bench's completion, its error branches), so a sourced chain would end at the
# first stage and every later one would silently never run.
#
# WHAT THIS PINS THAT THE LOSS REFIT DOES NOT
# -------------------------------------------
# The loss pilot is HOLDOUT protocol (train only, 50 epochs, adaptive-norm off,
# bench on val). The R series is the FINAL protocol, and every one of those
# differences changes the model:
#
#   TRAIN_SPLITS="train val"  val is FOLDED IN (merge_val=1). This is what makes
#                             PROTO_TAG empty, i.e. the run dir has no _holdout.
#   REFIT_EPOCHS=100          twice the pilot budget.
#   ADAPTIVE_NORM=1           with NORM_RECALIBRATE=post -> _anorm_recalpost in
#                             the run dir, the study and the bench model_name.
#   BENCH_SPLIT=test          val is inside training now, so a val bench would
#                             be a training score (the engine refuses it).
#
# Getting any of these wrong yields a DIFFERENT run dir, so the seed would not
# group with the tuned seed in the store — a silent no-op rather than an error.
#
# The SR treatment (UPSAMPLER / SR_PAD / SR_HC / SR_SNAPSHOT_EVERY) differs per
# arm and arrives through the environment; `_stages_tv.sh` reads them as plain
# shell variables under `set -u`, so they are required here rather than
# defaulted to r0's values.
#
#   env EXP_TAG=r2a_new LOSS_ARM=gap_ce UPSAMPLER=sen2sr SR_PAD=8 \
#       SEED=1 STAGE=fit bash _refit_arm.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

: "${EXP_TAG:?_refit_arm.sh needs EXP_TAG (r0_new|r2a_new|r2b_new|...)}"
: "${LOSS_ARM:?_refit_arm.sh needs LOSS_ARM}"
: "${SEED:?_refit_arm.sh needs SEED}"
: "${STAGE:?_refit_arm.sh needs STAGE (fit|bench)}"
: "${UPSAMPLER:?_refit_arm.sh needs UPSAMPLER (bicubic|sen2sr|sr4rs)}"

LABELS="${LABELS:-new}"
FREEZE_SR="${FREEZE_SR:-false}"
SR_PAD="${SR_PAD:-0}"
SR_HC="${SR_HC:-native}"
REG="${REG:-true}"
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"

# --- FINAL protocol, pinned so no per-arm script can drift off it ------------
export TRAIN_SPLITS="${TRAIN_SPLITS:-train val}"   # merge_val=1, no _holdout tag
export REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
export MONITOR="${MONITOR:-val_ap}"                # -> MON_TAG=_ap in model_name
export SKIP_TEST="${SKIP_TEST:-0}"
export BENCH_SPLIT="${BENCH_SPLIT:-test}"
# θ* is the argmax of MACRO F1 — the mean of the per-chip F1, so a sparse rural
# chip counts as much as a dense urban one. (The engine's own default is `iou`,
# which is GLOBAL pooled counts and is therefore tuned for whichever chips hold
# the most road pixels. SELECT_ON, which earlier versions of this file
# exported, is read by nothing — `_stages_tv.sh` passes the criterion through
# SWEEP_CRITERION.) All four criteria are recorded at every θ regardless, so
# re-argmaxing on another one later costs no inference.
export SWEEP_CRITERION="${SWEEP_CRITERION:-f1_macro}"
# Left at the engine defaults so these rows are shape-identical to the tuned
# seed's rows in the same append-only store. Set them only if you re-bench the
# whole arm, not one seed of it.
export TILE_METRICS="${TILE_METRICS:-apls}"
export BUFFER_PX="${BUFFER_PX:-}"
export AP_BINS="${AP_BINS:-}"

# --- architecture: the R series runs the adaptive-norm adapter ---------------
# Unlike the loss pilot (which predates it), every _new R run carries
# _anorm_recalpost. Pinning these is what makes RUN_DIR match the tuned seed's.
export ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
export ADAPTIVE_NORM_M="${ADAPTIVE_NORM_M:-0.01}"
export NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"

# --- the R-series rule: never re-search the loss at refit time ---------------
# θ/pos_weight come from the baked overlay. Leaving the search on would
# reconfound the SR comparison this series exists to make.
export SEARCH_THETAS="${SEARCH_THETAS:-false}"
export SEARCH_MIX_W="${SEARCH_MIX_W:-false}"

# --- compute shape: one L40S, eight cores ------------------------------------
export REFIT_GPUS="${REFIT_GPUS:-1}"
export SEARCH_GPUS="${SEARCH_GPUS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-7}"
export PRECISION="${PRECISION:-bf16-mixed}"

# --- wandb: one run per (arm, seed) ------------------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_r_series_seeds}"
export WANDB_NAME="${WANDB_NAME:-${EXP_TAG}_${LOSS_ARM}_seed${SEED}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${EXP_TAG}_${LOSS_ARM}}"

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
