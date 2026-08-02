#!/bin/bash
# Loss-pilot harness (Protocol v2.1, 2026-07-30). Every loss/l*_new.sh arm is
# the SAME r0_new experiment — bicubic x4, ROSA_New, pre-rasterised
# mask_new_2pt5 labels via JointSRDataModule(mask_source=raster) — differing
# ONLY in LOSS_ARM (+ its hps). Sourced by the arm script AFTER it sets
# LOSS_ARM; this file pins the pilot constants and hands off to the shared
# sr/_stages_tv.sh engine.
#
# Pilot rules (deliberate differences from the final _new refit protocol):
#   * TRAIN_SPLITS=train   val stays a HOLDOUT. Every pilot decision reads the
#                          val bench; test is never benched by the pilot. The
#                          engine tags run dirs / model names with _holdout so
#                          pilot and final rows can never mix in a store.
#   * REFIT_EPOCHS=50      the pilot's fixed budget — a BETWEEN-ARM CONSTANT.
#   * WARMUP_START=15/RAMP=5   §4.5 scaled to E=50 (engine defaults assume 100).
#   * SKIP_TEST=1          fit does not run the test split (test stays unseen).
#   * BENCH_SPLIT=val, TILE_METRICS=apls,cldice, STORE_DIR=benchmarks_loss_pilot.
#   * Checkpoint = END of budget for every arm (engine behaviour): uniform
#     selection, immune to the pre-ramp capture that broke the 10 m Phase C.
#
# PER-ARM TUNING (amendment 2026-08-02, supersedes the shared-tune rule):
# every arm runs its own Optuna search under an IDENTICAL search space and
# budget (N_TRIALS x TUNE_EPOCHS at TUNE_LENGTH patches/epoch) — the fairness
# constant is the budget, not the config. Dimensions are gated on what the
# arm consumes (sr.tune): lr+batch always; pos_weight λ (log [1,40]) for
# wbce/gap/tl/gap_tl; tl_theta/gap_theta ([0.3,0.7]) for the map-building
# arms; nothing extra for bce (the λ=1 floor), balance_ce (adaptive λ by
# construction) or the region arms (dice/sdice/lcdice — lr matters there
# regardless: lcDice ≈ Dice²/2 near 0, so its gradient scale is its own).
#
#   bash scripts/LightningStudio/run.sh loss/l2_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l2_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l2_new.sh STAGE=bench
#
# PILOT_SHARED_TUNE=1 restores the old behaviour (copy l10's overlay instead
# of tuning) — for compounds that inherit the winner's config, not Phase A.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

: "${LOSS_ARM:?arm script must set LOSS_ARM before sourcing _pilot_new.sh}"

# --- The r0_new configuration (identical for every arm) ----------------------
EXP_TAG="${EXP_TAG:-r0_new}"
LABELS="${LABELS:-new}"
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

# --- Pilot constants ----------------------------------------------------------
TRAIN_SPLITS="${TRAIN_SPLITS:-train}"
REFIT_EPOCHS="${REFIT_EPOCHS:-50}"
WARMUP_START="${WARMUP_START:-15}"
WARMUP_RAMP="${WARMUP_RAMP:-5}"
SKIP_TEST="${SKIP_TEST:-1}"
BENCH_SPLIT="${BENCH_SPLIT:-val}"
TILE_METRICS="${TILE_METRICS:-apls,cldice}"
STORE_DIR="${STORE_DIR:-${INSTAROAD_ROOT}/benchmarks_loss_pilot}"

# --- Per-arm tune budget (identical across arms — a protocol constant) --------
N_TRIALS="${N_TRIALS:-30}"
TUNE_EPOCHS="${TUNE_EPOCHS:-8}"
TUNE_LENGTH="${TUNE_LENGTH:-2000}"      # patches/epoch during trials (~4.4x cheaper
                                        # than the full 8830; ranking, not fitting)
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-40}"  # log search up to ~inverse frequency (34)

if [ "${TRAIN_SPLITS}" != "train" ]; then
  echo "WARN: pilot arm running with TRAIN_SPLITS='${TRAIN_SPLITS}' — this is no" >&2
  echo "  longer the loss pilot (val would not be an honest decision split)." >&2
fi

# --- Optional shared config (opt-in since 2026-08-02): copy l10's overlay ----
if [ "${STAGE:-tune}" = "fit" ] && [ "${PILOT_SHARED_TUNE:-0}" = "1" ]; then
  _runs="${RUNS_ROOT:-${INSTAROAD_ROOT}/runs}"
  _tag="_$(echo "${LOSS_ARM}" | tr '+' '-')"
  _dst="${_runs}/sr_${EXP_TAG}${_tag}_holdout_seed${SEED:-0}"
  _src="${PILOT_TUNE_FROM:-${_runs}/sr_${EXP_TAG}_wbce_holdout_seed${SEED:-0}}"
  if [ ! -f "${_dst}/best_params.yaml" ]; then
    if [ -f "${_src}/best_params.yaml" ]; then
      mkdir -p "${_dst}"
      cp "${_src}/best_params.yaml" "${_dst}/best_params.yaml"
      echo "pilot: shared screening config copied from ${_src}"
    else
      echo "ERROR: shared screening config not found: ${_src}/best_params.yaml" >&2
      echo "  Run the one shared tune first:" >&2
      echo "    bash scripts/LightningStudio/run.sh loss/l10_new.sh STAGE=tune" >&2
      echo "  (or point PILOT_TUNE_FROM at the run dir that has it.)" >&2
      exit 1
    fi
  fi
fi

source "$LS_DIR/sr/_stages_tv.sh"
