#!/bin/bash
# =============================================================================
# RL-SERIES CAMPAIGN CONSTANTS — Lightning Studio.
# docs/rl_lightning_campaign_plan.md (rev 3, 2026-08-29) is the authority for
# everything in this file; docs/sr_linear_probe.md remains the authority on the
# head itself and the θ*/AP protocol, and the campaign plan overrides it where
# they conflict.
#
# Sourced FIRST by every rl arm script — before env.sh, deliberately: env.sh
# exports NUM_WORKERS behind a ${VAR:-default} guard, so a value set here wins,
# and a submit-time NUM_WORKERS=… still beats both.
#
# WHY THIS FILE EXISTS. The whole series rests on one rule: every arm runs the
# IDENTICAL loss, head, budget, batch size, normalisation policy and band-guard
# policy, and only the SR front-end and the pinned lr_sr differ. Five
# hand-maintained copies of those constants is a drift hazard whose failure mode
# is silent — a drift would not break a run, it would break the control. One
# sourced file makes the constancy structural instead of clerical.
#
# Everything is overridable at submit time so a single arm can be perturbed for
# a diagnostic, but note that perturbing ONE arm voids every between-arm
# contrast in the series.
# =============================================================================

# --- The read-out ------------------------------------------------------------
# 1x1 conv over the 4 post-SR bands: 4 weights + 1 bias. No decoder to
# compensate for a bad front-end, so every bit of structure in the prediction
# was put there by the upsampler.
HEAD="${HEAD:-linear}"

# val_ap, not val_iou@0.5. A 5-parameter logistic regression has no reason to be
# calibrated at 0.5, so selecting on IoU@0.5 would reward whichever arm's SR
# output happens to place the decision boundary near 0.5 — the nuisance variable
# moving with the treatment (probe doc §5.1).
MONITOR="${MONITOR:-val_ap}"

# --- NO TUNING ANYWHERE (plan §1) -------------------------------------------
# The head lr is PINNED and the lr_sr ladder is fixed, so there is no search
# space left. The "tune" stage is run once per arm with ONE trial of ONE epoch
# and does two jobs, neither of them tuning:
#
#   1. it writes best_params.yaml, which STAGE=fit requires. Pinning a band's
#      two ends together (LR_MIN == LR_MAX) makes Optuna's log-uniform suggest
#      the constant, which then lands in the overlay exactly like a searched
#      value — the pos_weight / r2grid pattern, and the only reason the fit
#      stage needs no special case (tests/test_sr_hold_ramp.py pins it);
#   2. it IS the plan's §4 gate-1 timing run: one epoch per arm class on the
#      real data, recording h/epoch and peak VRAM before any 30-epoch job is
#      launched. Read them off this stage's log and the nvidia-smi peak.
#
# NOT N_TRIALS=0. That is the RESCUE idiom — it re-emits best_params.yaml from
# an EXISTING study's completed trials (engine header: "rescue a killed
# search"). Every rl arm here is a fresh study, and on a fresh study
# `study.optimize(n_trials=0)` runs nothing, so sr.tune hits its own guard:
#   Study '...' has 0 COMPLETE trials (0 total) -- nothing to write.
# One completed trial has to exist before there is anything to write down.
#
# It IS the right flag afterwards: once an arm's 1x1 pass has run, re-run its
# tune stage with N_TRIALS=0 to regenerate the overlay (after editing a pinned
# constant, say) without spending another epoch.
N_TRIALS="${N_TRIALS:-1}"
TUNE_EPOCHS="${TUNE_EPOCHS:-1}"
PATIENCE="${PATIENCE:-5}"   # inert at 1 epoch; kept so the engine's line prints

# --- The pinned head lr (plan §1, §5) ---------------------------------------
# 3e-3, PINNED for all five arms. Set 2026-08-29.
#
# The licence for pinning rather than searching is the cluster rl3 study
# (sr_rl3_new_linear_pstar_dice_anorm_recalpost_ap_seed0): ≥30 trials over
# lr ∈ [~2.4e-3, ~7.9e-3] moved val_ap by only 0.043–0.049 — the objective is
# flat in lr across more than half a decade, so re-searching it per arm would
# spend GPU-hours resolving noise AND would let search quality vary by
# treatment, which is the defect the whole series exists to avoid.
#
# 3e-3 sits at the low end of that band. It is a CHOSEN CONSTANT, not that
# study's argmax — the study is EVIDENCE ONLY (wrong loss, pstar_dice/gap_t4_ce,
# and wrong platform to be an arm of this campaign), so quoting its best trial
# to four significant figures would imply a precision the flat objective does
# not support. What matters for every rl contrast is that the number is the
# SAME in all five arms and inside the evidenced-flat region; both hold.
#
# If you would rather pin that study's actual best trial, read it with
#   python scripts/LightningStudio/sr/rl/read_head_lr.py <rl3 run dir>/study.db
# and change the default HERE — once, for all five arms. A per-arm head lr
# voids every between-arm contrast in the series.
HEAD_LR="${HEAD_LR:-3e-3}"
# Guard the band, not the exact value: a head lr outside the region the rl3
# study actually explored is not covered by the flatness evidence, so pinning it
# would be a guess wearing the evidence's clothes.
if ! awk -v v="$HEAD_LR" 'BEGIN { exit !(v + 0 == v && v >= 1e-3 && v <= 1e-2) }' \
     </dev/null 2>/dev/null; then
  echo "ERROR: HEAD_LR='${HEAD_LR}' is outside [1e-3, 1e-2]." >&2
  echo "  The rl3 study sampled lr in roughly [2.4e-3, 7.9e-3] and found the" >&2
  echo "  objective flat there — that flatness is the ONLY reason this campaign" >&2
  echo "  is allowed to pin the head lr instead of searching it. Outside that" >&2
  echo "  region there is no evidence, and a pinned guess would be worse than a" >&2
  echo "  search. (RL_LOOSE_OK=1 to override, for ALL FIVE arms or none.)" >&2
  if [ "${RL_LOOSE_OK:-0}" != "1" ]; then exit 2; fi
fi
LR_MIN="${LR_MIN:-$HEAD_LR}"
LR_MAX="${LR_MAX:-$HEAD_LR}"

# --- The budget: ONE 30-epoch run per arm-seed (plan §2) --------------------
# Frozen arms (rl0/rl1/rl3): 30 epochs head-only.
# Joint arms  (rl2/rl4):     10 epochs with lr_sr held at EXACTLY 0, then 20
#                            joint epochs on the rung's lr_sr.
# Fair by construction: with lr_sr = 0 the joint arm's first 10 epochs ARE a
# frozen-arm run, so the branches diverge only at epoch 11 and the frozen arm's
# epochs 11–30 ARE the matched-budget control. 30 = 30, no inheritance, no
# warm_start_head, no stage pairing. (Formal amendment to probe doc §2, whose
# cold-init rejection assumed the 1-epoch SOFT ramp; the HARD hold enforces
# LP-FT within one run.)
REFIT_EPOCHS="${REFIT_EPOCHS:-30}"
# SR_HOLD_EPOCHS is deliberately NOT set here. The joint arms set it (to 10) in
# _rl_rung.sh, behind its own ${VAR:-10} guard, so a submit-time
# SR_HOLD_EPOCHS=… still reaches it; defaulting it to 0 here would consume that
# guard and silently ignore the override. The frozen arms take the engine's own
# default of 0, at which nothing is appended to any command line.
# Re-based to the hold boundary by the model, so the joint arms ramp over epoch
# 11 rather than over an epoch 1 that is inside the hold.
SR_WARMUP_EPOCHS="${SR_WARMUP_EPOCHS:-1.0}"

# PINNED, never searched, and a BETWEEN-ARM CONSTANT. `length` is fixed per
# epoch, so changing bs silently changes the step budget — a bs=2 arm would take
# twice the optimiser steps of a bs=4 arm inside the same epoch count. 4 is also
# the largest that fits the heaviest arms (rl3/rl4 run SR4RS's 256-channel
# convs, one of them 9x9, at 512 px).
BATCH_SIZES="${BATCH_SIZES:-4}"
PRECISION="${PRECISION:-bf16-mixed}"   # L4 has native bf16; never a T4 (plan §3)

# --- Loss: wbce, PINNED, uniform across all five arms (plan §1) -------------
# The gap-family losses' CPU cost was the old ~1 h/epoch problem; wbce is
# GPU-only. pos_weight is COPIED from an existing wbce overlay, never searched:
#   runs/sr_r0_new_wbce_holdout_seed0/best_params.yaml   (λ* = 2.4789710497080004,
#   searched at batch_size=4 under the same holdout protocol this campaign uses).
# min == max is how this engine expresses "a constant, not a band".
#
# Fallback if a joint arm collapses: wbce+dice — applied to ALL FIVE ARMS or not
# at all. Write-up cost of the loss choice: one more qualitative difference from
# the R-series (joining bs / monitor / capacity / budget). The rl series is
# internally controlled; rl-vs-r is qualitative, stated once.
LOSS_ARM="${LOSS_ARM:-wbce}"
PSTAR="${PSTAR:-bce}"                 # inert: `wbce` is a base arm, not a pstar_*
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-2.4789710497080004}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-2.4789710497080004}"
SEARCH_THETAS="${SEARCH_THETAS:-false}"
SEARCH_MIX_W="${SEARCH_MIX_W:-false}"

# --- Protocol: holdout, so the val curves exist (plan §2, §6) ---------------
# TRAIN_SPLITS=train, NOT the R-series' merged "train val". The campaign's
# primary evidence is within-run trajectories: "frozen val-AP must plateau
# before epoch 10" is the falsifiable check on the hold length, and "the joint
# arms' hold phases reproduce the frozen curves" is the harness self-check
# (§6.1). Both need a val loop every epoch, and the merged protocol sets
# limit_val_batches: 0 — there would be no curve to read. Nothing is lost by
# holding val out here, because nothing is tuned on it.
#
# Consequences, all of them wanted: the run dir / study / bench row gain a
# _holdout tag (so no rl row can ever land beside an R-series one); θ* is swept
# on val, which under this protocol is genuinely UNSEEN; and test stays the
# single held-out report split.
TRAIN_SPLITS="${TRAIN_SPLITS:-train}"
# ...and the fit must actually RUN the val loop. joint_sr_trainval.yaml is
# layered last at the fit stage and sets limit_val_batches: 0 unconditionally,
# so TRAIN_SPLITS=train alone gives a holdout SPLIT with no holdout CURVE.
# FIT_VAL_LOOP=1 restores the loop without restoring any selection: the
# checkpoint is still the end of a fixed 30-epoch budget (monitor:null,
# save_on_train_epoch_end, no EarlyStopping). The val loop only logs.
FIT_VAL_LOOP="${FIT_VAL_LOOP:-1}"
BENCH_SPLIT="${BENCH_SPLIT:-test}"
SWEEP_SPLIT="${SWEEP_SPLIT:-val}"
# Global pooled counts, not the macro criteria: a macro denominator moves with
# how badly an arm collapsed, which is the axis the top rung deliberately
# probes. All four criteria land in sweep.json either way.
SWEEP_CRITERION="${SWEEP_CRITERION:-iou}"
AP_BINS="${AP_BINS:-101}"             # AP leads every figure in this campaign
TILE_METRICS="${TILE_METRICS:-apls}"

# --- Recipe v2 dynamics ------------------------------------------------------
# CLIP is deliberately left at the engine's 1.0 and NOT restated here: under
# HEAD=linear the engine routes it to --model.clip_sr and switches the
# Trainer-level global clip off, so the 5-parameter head is treated identically
# in the frozen and joint arms (probe doc §6.2). Setting it here would also
# defeat the REG=false branch, which zeroes CLIP only if nothing claimed it.
REG="${REG:-true}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"  # the hold gate lives inside this cosine
L2SP_LAMBDA="${L2SP_LAMBDA:-0.0}"

# --- Adaptive post-SR normalisation, ON and TRACKING ------------------------
# Uniform across all five arms INCLUDING the frozen ones — ANORM_TAG is part of
# the run dir and the bench model_name, so an arm run without it could never
# share a row with one run with it.
ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
ADAPTIVE_NORM_M="${ADAPTIVE_NORM_M:-0.01}"
NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"

# --- Band guard: DEFANGED, NOT DELETED (plan §2, grid plan §3) --------------
# The top rung (1e-3) exists to probe the ceiling, and the guard exists to kill
# exactly the collapse that rung is asked to produce. Loosened rails mean the
# raise can never fire while the warn stream and the variance-floor diagnostic
# keep printing — the stability envelope is then read POST HOC from the logged
# adapt_std_b* curves, at the step each band crosses the NOMINAL 0.5x / 4.0x of
# its starting std. That is strictly more information than the raise gave: the
# trajectory after the crossing is retained.
#
# Do NOT reach for adaptive_norm_check_every=0 instead — that silences the warn
# stream too, i.e. deletes the measurement. And do NOT delete the exit path: it
# is what RAILS_TAG marks, and the tag is what keeps these runs visibly
# non-production in an append-only store.
#
# At the 1e-3 rung the failure mode may be NaN rather than a band exit. That
# endpoint is data: record the step, add no machinery.
STD_BAND_RAISE_LO="${STD_BAND_RAISE_LO:-0.01}"
STD_BAND_RAISE_HI="${STD_BAND_RAISE_HI:-100}"
STD_BAND_ACTION="${STD_BAND_ACTION:-warn}"

# --- Snapshots: EVERY epoch --------------------------------------------------
# The strips ARE the mechanism figure, and a collapse with no frames is an
# observation that cannot be shown. Each snapshot records sr_hold_epochs, so a
# dose-0 frame (epoch < hold — the ladder's free zero-dose point) is
# distinguishable from an adapting one without going back to the config.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

# --- Bare-only: no hard constraint anywhere (plan §1) -----------------------
# The joint arms measure the UNCONSTRAINED upper bound on how far task gradients
# repurpose the generator into a segmenter; the HC question belongs to the
# R-series 2x2 and the r2grid probes. rl1 therefore twins r1b, not r1a. Frozen
# splice contrasts come free from LDA on cached outputs.
# Forced `off` (not `native`) on every arm including bicubic, so all five run
# dirs carry the same _nohc tag and no arm's constraint state is implicit.
SR_HC="${SR_HC:-off}"
SR_PAD="${SR_PAD:-0}"

# --- Compute ----------------------------------------------------------------
# One L4 per job; the lanes are the parallelism, not the boxes (plan §3).
SEARCH_GPUS="${SEARCH_GPUS:-1}"
REFIT_GPUS="${REFIT_GPUS:-1}"
# env.sh's NUM_WORKERS=0 is a DDP-era GDAL fork guard. The pre-rasterised mask
# COGs open lazily inside __getitem__ and are fork-safe, and an L4 job box has
# 8 vCPU, so 0 workers just starves the GPU.
#
# The `:-` idiom cannot be used here: run.sh sources env.sh BEFORE it execs an
# arm script, so NUM_WORKERS always arrives already exported as 0 and a
# `${NUM_WORKERS:-4}` would resolve to 0 every time. Treat 0 as "env.sh's
# default, not a choice", let a non-zero submit-time value through, and use
# RL_NUM_WORKERS to ask for 0 (or anything else) deliberately.
if [ -n "${RL_NUM_WORKERS:-}" ]; then
  NUM_WORKERS="$RL_NUM_WORKERS"
elif [ -z "${NUM_WORKERS:-}" ] || [ "${NUM_WORKERS}" = "0" ]; then
  NUM_WORKERS=4
fi
export NUM_WORKERS

SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_rl_campaign}"

# --- Sanity: the controls must actually be constant -------------------------
if [ "$SEARCH_THETAS" != "false" ] || [ "$SEARCH_MIX_W" != "false" ] \
   || [ "$POS_WEIGHT_MIN" != "$POS_WEIGHT_MAX" ] || [ "$LR_MIN" != "$LR_MAX" ]; then
  echo "WARN: the rl-series controls are NOT frozen for this submit:" >&2
  echo "  lr=[${LR_MIN}, ${LR_MAX}]  pos_weight=[${POS_WEIGHT_MIN}, ${POS_WEIGHT_MAX}]" >&2
  echo "  SEARCH_THETAS=${SEARCH_THETAS} SEARCH_MIX_W=${SEARCH_MIX_W}" >&2
  echo "  Every rl contrast assumes the head, the loss and the budget are" >&2
  echo "  identical across arms. If this is deliberate it must be done to ALL" >&2
  echo "  FIVE arms, or the series is void. (RL_LOOSE_OK=1 to silence.)" >&2
  if [ "${RL_LOOSE_OK:-0}" != "1" ]; then exit 2; fi
fi
if [ "$TUNE_EPOCHS" != "1" ] || [ "$N_TRIALS" != "1" ]; then
  echo "NOTE: N_TRIALS=${N_TRIALS} TUNE_EPOCHS=${TUNE_EPOCHS} — the campaign pins" >&2
  echo "  1x1 (a config-writing + timing pass, not a search). Anything else is" >&2
  echo "  tuning, which plan §1 rules out for every arm." >&2
fi

echo "[rl] head=${HEAD}  monitor=${MONITOR}  loss=${LOSS_ARM}(λ=${POS_WEIGHT_MIN})  bs=${BATCH_SIZES}"
echo "[rl] head lr PINNED at ${HEAD_LR}  |  budget ${REFIT_EPOCHS} ep (joint arms hold the first 10)"
echo "[rl] bare-only: sr_hc=${SR_HC} sr_pad=${SR_PAD}  |  rails=[${STD_BAND_RAISE_LO}x, ${STD_BAND_RAISE_HI}x] action=${STD_BAND_ACTION}"
echo "[rl] protocol: train_splits='${TRAIN_SPLITS}' (val curves kept), sweep on ${SWEEP_SPLIT}, bench on ${BENCH_SPLIT}"
