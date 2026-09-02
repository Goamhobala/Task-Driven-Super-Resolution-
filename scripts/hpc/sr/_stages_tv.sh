#!/bin/bash
# Shared tune/fit/bench engine for the FINAL (train+val refit) SR series.
# NOT submitted directly — each r*_new.sh sets its config and sources this.
#
# This is the sibling of _stages.sh. Same arms, same recipe-v2 dynamics, same
# model code. Two deliberate differences, and nothing else:
#
#   1. DATASET   LABELS=new -> ROSA_New (the final curated dataset). The _all
#                series stays pointed at ROSA_all; run dirs, Optuna studies and
#                benchmark model_names are keyed on EXP_TAG (r0_new vs r0_all),
#                so old and new results can never mix in the store.
#
#   2. PROTOCOL  STAGE=tune is UNCHANGED — Optuna still trains on `train` and
#                scores on `val`, because that is what the holdout is for.
#                STAGE=fit then RE-FOLDS val into the training set
#                (data.train_splits = [train, val]) and reports on `test`
#                alone. Hyperparameters were already paid for out of val; once
#                chosen, withholding those tiles from the fit throws away ~17%
#                of the data for no inferential gain.
#
# Consequence of (2): during the refit there is NO honest holdout, so the
# val-driven machinery is removed rather than allowed to peek at data the model
# now trains on. src/sr/configs/joint_sr_trainval.yaml does this:
#   * limit_val_batches: 0     — no val loop at all
#   * EarlyStopping dropped    — FIXED, pre-registered epoch budget, identical
#                                across arms (same fairness rule as the loss
#                                ablation). Recipe v2's cosine has T_max =
#                                max_epochs, so the budget ends at LR 0: the
#                                schedule always completes.
#   * monitor: null            — the tested checkpoint is the END of the
#                                budget, saved as unet_s2rosa_jointsr_final.ckpt
#                                (NOT ..._best.ckpt, which by convention means
#                                "argmax over val" — the two names must never
#                                be confusable downstream).
#
# ---- Interface (identical to _stages.sh unless noted) -----------------------
# Experiment scripts must set:
#   EXP_TAG      e.g. r2a_new (drives the run dir + study + benchmark name)
#   LABELS       new | all | cdngi | overture | osm   (label/dataset source)
#   UPSAMPLER    sen2sr | sen2sr_full | sr4rs | bicubic
#   FREEZE_SR    true | false
#   SR_PAD       reflect-pad in native px (0 = off, 8 = border-artifact fix)
#
# Optional (submit-time or experiment-script) — see _stages.sh for the full
# prose on each; they behave identically here:
#   WARM_START_CKPT  stage-1 UNet init (r6/r7 staged protocol). _warm_tv.sh
#                derives it from the stage-1 arm's FINAL ckpt.
#   LOSS_ARM     any unet.losses.build_loss arm (+ its hps).
#   Recipe v2:   CLIP LR_SCHEDULE SR_WARMUP_EPOCHS L2SP_LAMBDA REG
#   SR_HC        native (default) | on | off — the FFT hard constraint as a
#                TREATMENT, crossed with the generator (the HC 2x2,
#                docs/hc_2x2_plan.md). native = each upsampler's shipped
#                behaviour, i.e. every pre-existing arm, and appends no flags at
#                all. HC_MASK_PATH supplies the shipped mask when the generator
#                does not ship one (sr4rs). Tagged (HC_TAG) into run dir, study
#                and bench model_name.
#   SR_SNAPSHOT_EVERY   fit-stage SR-weights-only snapshots every N epochs.
#   Adaptive post-SR normalisation (docs/adaptive_norm_plan.md, default OFF):
#     ADAPTIVE_NORM=1 / ADAPTIVE_NORM_M / NORM_RECALIBRATE=off|pre|post|auto.
#     Both tag the run dir, study and bench model_name (ANORM_TAG), so an
#     adaptive-norm arm can never land in a frozen-stats row of the same arm.
#   STD_BAND_ACTION=warn|raise   what a std-band exit DOES. Default: the fit
#                warns and continues (a band exit is a result, not a fault),
#                the tune prunes the trial. `warn` disarms the tune's abort
#                too — for a pinned-lr_sr cell, where a pruned trial silently
#                removes its lr from the ranking. Tagged (RAILS_TAG).
#   STD_BAND_RAISE_LO / STD_BAND_RAISE_HI   the post-SR std band's hard rails,
#                as multiples of the run's STARTING std (production 0.5 / 4.0).
#                Set BOTH or neither. For the lr_sr mechanism grid only
#                (docs/lrsr_grid_ablation_plan.md §3): the tune prunes on band
#                exit, so with lr_sr pinned the production band would prune
#                every trial of the extreme cells and the guard would decide
#                which cells exist. Tagged (RAILS_TAG) into run dir, study and
#                bench model_name — loosened rails are a protocol difference
#                and must never share a row with a production-band arm.
#
# NEW here:
#   REFIT_EPOCHS     the pre-registered budget (default 100). This is now a
#                nothing-stops-it-early budget, so it is a BETWEEN-ARM CONSTANT
#                of the protocol: change it for one arm and the comparison is
#                void. Check the walltime — every arm now runs the full count.
#   CHAIN_BENCH      at the END of a fit, re-enter the arm at STAGE=bench in the
#                same allocation, so ONE submission is fit -> test -> theta*
#                sweep -> bench. DEFAULT 1 (changed 2026-09-02): a fit whose
#                score never reached the store is an unfinished run, and the
#                bench takes minutes next to the fit's hours. Skipped (with a
#                note) under SKIP_TEST=1, whose point is that test stays unseen.
#                BUDGET THE WALLTIME FOR BOTH — a job killed at the wall after
#                training loses the bench with it (re-run it alone with
#                STAGE=bench; nothing is recomputed). CHAIN_BENCH=0 opts out.
#   BEST_PARAMS      overlay TEXT for an arm that searches nothing: planted as
#                <run dir>/best_params.yaml at the fit/bench stages, so
#                STAGE=tune is not required at all. Must match what
#                sr.tune.write_best_overlay would have written for those pinned
#                constants; pin that with a test, not by eye.
#   FIT_EARLY_STOP=1  re-add EarlyStopping to the REFIT. rl3 ONLY (whitelisted
#                by EXP_TAG — every other arm's budget is fixed), and holdout
#                fits only (TRAIN_SPLITS=train + FIT_VAL_LOOP=1).
#                ES_MONITOR / ES_MODE / ES_PATIENCE / ES_MIN_DELTA configure it;
#                ES_TAG (_es<patience>) keeps an early-stopped row out of a
#                fixed-budget one. The budget becomes a CEILING for that arm and
#                the cosine no longer completes — see the block for the full
#                cost. Default 0 = every existing arm, nothing appended.
#   TRAIN_SPLITS     "train val" (default). Set "train" to reproduce the old
#                holdout protocol on ROSA_New without switching engines; the
#                run dir / study / bench name then gain a _holdout tag so the
#                two protocols can never land in the same store row.
#   NORM_CONFIG      normalisation stats. DEFAULT = <DATASET_DIR>/norm_stats.yaml,
#                i.e. the file `sentinel2data.cli norm-stats` writes into the
#                dataset itself — the only copy guaranteed to have been computed
#                from THIS dataset's splits/train.csv. Unlike _stages.sh, this
#                engine does NOT hard-code the repo copy; it falls back to it
#                only if the dataset has none, and then refuses to run without
#                NORM_FALLBACK_OK=1.
#
# STAGE=tune   Optuna on train/val (unchanged). CHAIN_FIT=1 continues into fit
#              in the same allocation. Early stop: `touch <run dir>/STOP`.
#              Rescue a killed search: rerun with N_TRIALS=0.
# STAGE=fit    Refit on train+val for REFIT_EPOCHS on ONE GPU, then test, then
#              the theta* sweep, then (CHAIN_BENCH, default 1) the bench — one
#              submission end to end. RESUME_FIT=1 continues from last.ckpt.
# STAGE=bench  Score unet_s2rosa_jointsr_final.ckpt into the shared store.
#
# Replication contract: only SEED, STAGE and the loss block are meant to vary.
set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"

: "${EXP_TAG:?experiment script must set EXP_TAG}"
: "${LABELS:-new}"
: "${UPSAMPLER:?experiment script must set UPSAMPLER (sen2sr|sen2sr_full|sr4rs|bicubic)}"
: "${FREEZE_SR:?experiment script must set FREEZE_SR (true|false)}"
: "${SR_PAD:?experiment script must set SR_PAD (0 = off)}"

# --- FFT hard constraint x generator (docs/hc_2x2_plan.md) -------------------
# The constraint is architecture-agnostic (a pure function of lr, sr and the
# shipped mask), so it can be taken OFF SEN2SR (r2b) or put ON SR4RS (r4a).
#   native  each upsampler's shipped default — sen2sr applies it, sr4rs and
#           bicubic do not. EVERY EXISTING ARM. No flag is appended to any
#           command line in this mode, so their invocations stay byte-identical.
#   on|off  forced. The treatment is the whole bundle: positivity clamp +
#           frequency splice. The pad component travels with it too, but as an
#           ARM-SCRIPT setting (SR_PAD), not from here — the r4b-at-pad-8
#           control has to stay expressible.
# HC_TAG goes into the run dir, the Optuna study AND the bench model_name, like
# REG_TAG/ANORM_TAG. That is what makes the r2b redefinition safe: r2b_new used
# to mean "SEN2SR+HC, pad 0" and now means "SEN2SR, no HC", so any legacy
# sr_r2b_new_* rows can never collide with the new sr_r2b_new_nohc_* ones in the
# append-only store.
SR_HC="${SR_HC:-native}"
HC_MASK_PATH="${HC_MASK_PATH:-}"
case "$SR_HC" in
native) HC_TAG="" ;;
on) HC_TAG="_hc" ;;
off) HC_TAG="_nohc" ;;
*)
  echo "ERROR: SR_HC must be native|on|off, got '${SR_HC}'." >&2
  exit 2
  ;;
esac
if [ "$SR_HC" = "on" ] && [ "$UPSAMPLER" = "bicubic" ]; then
  echo "ERROR: SR_HC=on with UPSAMPLER=bicubic. The constraint splices the" >&2
  echo "  bicubic upsampling of the input into the SR output, so on a bicubic" >&2
  echo "  'generator' the cell is a near-identity, not a treatment." >&2
  exit 2
fi
if [ "$SR_HC" = "on" ] && [ "$UPSAMPLER" = "sr4rs" ] && [ -z "$HC_MASK_PATH" ]; then
  echo "ERROR: SR_HC=on with UPSAMPLER=sr4rs needs HC_MASK_PATH — SEN2SR-Lite's" >&2
  echo "  hard_constraint.safetensor. SEN2SR_DIR points at SR4RS_RGBN here, which" >&2
  echo "  ships no mask. Reuse the shipped file byte-for-byte: its cutoff is the" >&2
  echo "  value the SEN2SR paper optimised (Table 4), and re-deriving one would" >&2
  echo "  make the two rows of the 2x2 different operators." >&2
  exit 2
fi
# Appended to the tune/fit command lines ONLY when the constraint is forced, so
# every native arm's invocation is unchanged (the same discipline as HEAD_TAG).
HC_ARGS_TUNE=()
HC_ARGS_FIT=()
if [ "$SR_HC" != "native" ]; then
  HC_ARGS_TUNE=(--sr-hc "$SR_HC")
  HC_ARGS_FIT=(--model.sr_hc "$SR_HC")
  if [ -n "$HC_MASK_PATH" ]; then
    HC_ARGS_TUNE+=(--hc-mask-path "$HC_MASK_PATH")
    HC_ARGS_FIT+=(--model.hc_mask_path "$HC_MASK_PATH")
  fi
fi

STAGE="${STAGE:-tune}"
SEED="${SEED:-0}"
# The ARM script that sourced this engine, as an absolute path. Resolved HERE,
# at the top, because the fit stage `cd`s into RUN_DIR and a relative $0 would
# stop resolving after that. Used by CHAIN_BENCH (bottom of the file) to re-enter
# this same arm at STAGE=bench, the way train_both.sbatch re-enters it — a fresh
# process, so the bench resolves every variable exactly as a standalone bench
# would, with nothing leaking out of the fit.
ARM_SCRIPT="${ARM_SCRIPT:-$(cd "$(dirname "$0")" 2>/dev/null && pwd)/$(basename "$0")}"
# 0 was a DDP-era guard (GDAL handles + forked ranks). Search runs one
# single-GPU process per GPU and the refit is single-GPU, and the datasets
# open rasters lazily inside __getitem__, so forked loader workers are safe.
# Default is computed below, after SEARCH_GPUS is known.
NUM_WORKERS="${NUM_WORKERS:-}"
PRECISION="${PRECISION:-bf16-mixed}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN}"
WARM_START_CKPT="${WARM_START_CKPT:-}"

# --- The train+val protocol switch -------------------------------------------
# Space-separated split names for the REFIT's train loader. The default IS the
# protocol; "train" reverts to the classic holdout fit and tags itself so the
# two never mix.
TRAIN_SPLITS="${TRAIN_SPLITS:-train val}"
PROTO_TAG=""
MERGE_VAL=1
case " ${TRAIN_SPLITS} " in
*" test "*)
  echo "ERROR: TRAIN_SPLITS must never contain 'test' — that is the held-out" >&2
  echo "  evaluation split. Got '${TRAIN_SPLITS}'." >&2
  exit 2
  ;;
*" val "*) : ;;
*)
  PROTO_TAG="_holdout"
  MERGE_VAL=0
  ;;
esac

# --- Recipe v2 training dynamics (defaults = the agreed recipe) --------------
REG="${REG:-true}"
REG_TAG=""
if [ "$REG" = "false" ] || [ "$REG" = "0" ]; then
  REG_TAG="_noreg"
  CLIP="${CLIP:-0}"
  LR_SCHEDULE="${LR_SCHEDULE:-none}"
  SR_WARMUP_EPOCHS="${SR_WARMUP_EPOCHS:-0}"
fi
CLIP="${CLIP:-1.0}"                         # gradient clip (global L2; 0=off)
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"        # cosine | none
SR_WARMUP_EPOCHS="${SR_WARMUP_EPOCHS:-1.0}" # SR-group ramp; model auto-off
# for frozen/bicubic/warm-start

# --- Hard hold on the SR group (docs/rl_lightning_campaign_plan.md §2) -------
# Epochs during which lr_sr is EXACTLY 0 while the head trains alone; after the
# boundary the SR group runs its own cosine over the remaining budget, with
# SR_WARMUP_EPOCHS re-based to that boundary. Not a ramp and not a small LR: an
# LR gate, so Adam's update is identically zero while its moments warm on the
# real gradients.
#
# This is what lets the rl campaign run ONE 30-epoch job per arm-seed instead of
# a stage pair: with lr_sr=0 the joint arm's first HOLD epochs ARE a frozen-arm
# run, so the branches diverge only at the boundary and the frozen arm's
# remaining epochs are the matched-budget control. No warm_start_head, no
# inheritance, no stage-1 dependency.
#
# DEFAULT 0 = no hold, i.e. exactly the behaviour every arm in the append-only
# store was trained under — and NOTHING is appended to any command line at 0
# (same discipline as HC_ARGS/RAILS_ARGS/HEAD_TAG), so their invocations stay
# byte-identical. It is deliberately NOT tagged into the run dir / study /
# model_name: the only arms that use it are the rl joint rungs, whose EXP_TAG
# already carries the rung. If a held and an unheld variant of the SAME EXP_TAG
# are ever both wanted, tag it before running the second one.
SR_HOLD_EPOCHS="${SR_HOLD_EPOCHS:-0}"
HOLD_ARGS_TUNE=()
HOLD_ARGS_FIT=()
if awk -v h="$SR_HOLD_EPOCHS" 'BEGIN { exit !(h + 0 == h && h > 0) }' </dev/null 2>/dev/null; then
  HOLD_ARGS_TUNE=(--sr-hold-epochs "$SR_HOLD_EPOCHS")
  HOLD_ARGS_FIT=(--model.sr_hold_epochs "$SR_HOLD_EPOCHS")
  if [ "$FREEZE_SR" = "true" ] || [ "$UPSAMPLER" = "bicubic" ]; then
    echo "NOTE: SR_HOLD_EPOCHS=${SR_HOLD_EPOCHS} on a front-end with no trainable" >&2
    echo "  SR parameters (freeze_sr=${FREEZE_SR}, upsampler=${UPSAMPLER}) — the" >&2
    echo "  model auto-disables it. Harmless; the frozen rl arms pass it anyway so" >&2
    echo "  one submit line covers the whole series." >&2
  fi
  if [ "$LR_SCHEDULE" != "cosine" ]; then
    echo "ERROR: SR_HOLD_EPOCHS=${SR_HOLD_EPOCHS} with LR_SCHEDULE=${LR_SCHEDULE}." >&2
    echo "  The gate lives inside the cosine LambdaLR; with lr_schedule=none there" >&2
    echo "  is no scheduler at all and lr_sr would be live from step 1 — a joint" >&2
    echo "  arm silently running without its hold. Refusing." >&2
    exit 2
  fi
fi
L2SP_LAMBDA="${L2SP_LAMBDA:-0.0}" # 0 = dormant L2-SP anchor

# --- Read-out head: U-Net (default) or linear probe (docs/sr_linear_probe.md) -
# HEAD=linear replaces the 24 M-param U-Net with a 1x1 conv (4 weights + 1 bias)
# applied after the existing z-score adapter — the rl-series. It removes the
# decoder's ability to compensate for a bad SR front-end, so every bit of
# structure in the prediction must have been put there by the upsampler.
#
# DEFAULT IS `unet`, and every derived string below is EMPTY in that case, so
# each r*-arm's run dir, study name and bench model_name are byte-identical to
# what they were before this block existed. Verify that before trusting any
# comparison against a row already in the (append-only) store.
#
# HEAD_TAG goes into RUN_DIR as well as STUDY_NAME/MODEL_NAME (unlike MON_TAG,
# which is deliberately kept out of RUN_DIR). That is safe only because
# _warm_head_tv.sh reconstructs the same tag when it resolves a stage-1 run dir
# — change one and you must change the other.
HEAD="${HEAD:-unet}"
case "$HEAD" in
unet | linear) : ;;
*)
  echo "ERROR: HEAD must be unet|linear, got '${HEAD}'." >&2
  exit 2
  ;;
esac
HEAD_TAG=""
[ "$HEAD" != "unet" ] && HEAD_TAG="_${HEAD}"

# Gradient clipping split (docs/sr_linear_probe.md §6.2). Lightning's
# `gradient_clip_val` is a SINGLE GLOBAL L2 norm over all trainable parameters.
# For the U-Net arms that is fine — one 24 M-param group. For the linear probe
# it is not: in rl1 the group is 5 parameters, in rl2 it is those 5 plus ~240 k
# SEN2SR parameters, so the same setting means something different in each arm
# and the nuisance variable moves with the treatment. Under HEAD=linear the
# Trainer-level clip is therefore switched OFF and the value is handed to the
# module as `clip_sr`, which clips the SR parameter group only and leaves the
# head unclipped (configure_gradient_clipping override). CLIP stays the single
# source of the 1.0 — the arms do not carry their own copy.
CLIP_TRAINER="$CLIP"
CLIP_SR="0"
if [ "$HEAD" = "linear" ]; then
  CLIP_TRAINER="0"
  CLIP_SR="$CLIP"
fi

# Head warm start (LP-FT). Set by _warm_head_tv.sh for rl2/rl4 — the joint arm
# takes its frozen twin's FINAL head so the probe is fully converged on the
# frozen-SR input distribution before any gradient reaches the generator.
# DELIBERATELY NOT `warm_start_unet`: model.py zeroes the SR warmup ramp when
# warm_start_unet is set (correct for a staged U-Net start, wrong here), so
# routing the head through that flag would silently drop sr_warmup_epochs.
WARM_START_HEAD="${WARM_START_HEAD:-}"

# --- Adaptive post-SR normalisation (docs/adaptive_norm_plan.md) -------------
# The post-SR z-score uses FROZEN dataset stats; SR4RS's output is unanchored
# and can drift out from under them under task-only fine-tuning. Both default
# OFF, so every arm already in the benchmark store keeps its exact recipe.
#   ADAPTIVE_NORM=1        per-batch EMA of the post-SR moments
#   ADAPTIVE_NORM_M=0.01   EMA momentum (PINNED, never searched)
#   NORM_RECALIBRATE=pre   exact PreciseBN recompute: off|pre|post|auto
#                          `pre` is the complete, zero-risk fix for the FROZEN
#                          arms (r1/r5); `post` cleans EMA lag out of the
#                          shipped ckpt; `auto` picks per arm.
# Both contribute to ANORM_TAG so the run dir, Optuna study and benchmark
# model_name can never collide with the frozen-stats rows of the same arm
# (the store is append-only -- §4.6).
ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
ADAPTIVE_NORM_M="${ADAPTIVE_NORM_M:-0.01}"
NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"
case "$NORM_RECALIBRATE" in
off | pre | post | auto) : ;;
*)
  echo "ERROR: NORM_RECALIBRATE must be off|pre|post|auto, got '${NORM_RECALIBRATE}'." >&2
  exit 2
  ;;
esac
ANORM_TAG=""
ADAPTIVE_NORM_FLAG="false"
if [ "$ADAPTIVE_NORM" = "1" ] || [ "$ADAPTIVE_NORM" = "true" ]; then
  ADAPTIVE_NORM_FLAG="true"
  ANORM_TAG="_anorm"
fi
if [ "$NORM_RECALIBRATE" != "off" ]; then
  ANORM_TAG="${ANORM_TAG}_recal${NORM_RECALIBRATE}"
fi

# --- Std-band rails (docs/lrsr_grid_ablation_plan.md §3) ---------------------
# The post-SR std band is a HARD guard: `sr.tune` runs the model with
# std_band_action=raise, so a trial whose band leaves [lo, hi] x its starting
# std is PRUNED, and the fit logs adapt_band_exit=1 and keeps going. Both rails
# default to the production band (0.5x / 4.0x, src/sr/configs/joint_sr.yaml) and
# NOTHING is appended to any command line unless they are set here — so every
# arm already in the store keeps its exact invocation.
#
# They exist as envs for ONE purpose: the lr_sr mechanism grid, whose whole
# point is to observe the collapse the guard exists to kill. With lr_sr pinned
# per cell, the production band would band-exit-prune every trial of the bare
# lane at lr_sr=1e-4, no best_params would be written, and the GUARD (not the
# design) would decide which cells exist. The grid therefore sets 0.01 / 100 —
# the raise can never fire, while the warn stream and the variance-floor
# diagnostic keep printing. Do NOT instead silence the check with
# adaptive_norm_check_every=0: that kills the diagnostics the grid is FOR.
#
# Rules that come with them:
#   * BOTH or NEITHER. A half-loosened band is a different treatment on one
#     side only, and the collapse side is the one that matters.
#   * IDENTICAL across every cell of one grid (the pair rule).
#   * SET FOR EVERY STAGE of a cell. RAILS_TAG lands in the run dir, so a tune
#     with rails followed by a fit without them looks for best_params.yaml in a
#     directory that does not exist — loud, not silent, but still a waste of a
#     queue slot. Set them in the arm script (r2grid_new.sh does), not at submit.
#
# STD_BAND_ACTION is the companion knob: WHAT an exit does.
#   fit   ALWAYS warn unless this env says otherwise — joint_sr.yaml's default,
#         and the reason a band exit can never end a refit (the 2026-08-19
#         r4b_new kill is what set that default).
#   tune  `sr.tune` hard-codes `raise` so a hopeless corner is PRUNED. That is
#         wrong for a pinned-lr_sr cell: with lr_sr fixed, a pruned trial
#         removes its lr from the ranking, while a completed one records "this
#         lr, at this adaptation rate, scored X" — a terrible X included.
#         STD_BAND_ACTION=warn switches the search to that behaviour. Trials
#         are still pruned by MedianPruner on the OBJECTIVE; only the moment-
#         based abort goes away.
STD_BAND_RAISE_LO="${STD_BAND_RAISE_LO:-}"
STD_BAND_RAISE_HI="${STD_BAND_RAISE_HI:-}"
STD_BAND_ACTION="${STD_BAND_ACTION:-}"
RAILS_TAG=""
RAILS_ARGS_TUNE=()
RAILS_ARGS_FIT=()
if [ -n "$STD_BAND_ACTION" ]; then
  case "$STD_BAND_ACTION" in
  warn | raise) : ;;
  *)
    echo "ERROR: STD_BAND_ACTION must be warn|raise, got '${STD_BAND_ACTION}'." >&2
    exit 2
    ;;
  esac
  RAILS_TAG="_rails" # band-guard policy differs from production: tag it
  RAILS_ARGS_TUNE+=(--std-band-action "$STD_BAND_ACTION")
  RAILS_ARGS_FIT+=(--model.std_band_action "$STD_BAND_ACTION")
  if [ "$STD_BAND_ACTION" = "raise" ] && [ "$STAGE" = "fit" ]; then
    echo "WARN: STD_BAND_ACTION=raise on a FIT. A band exit will then KILL this" >&2
    echo "  refit mid-budget. That is what took out r4b_new at epoch 13 of 100" >&2
    echo "  on 2026-08-19, on a decelerating trend that crossed the bound by" >&2
    echo "  0.004. A band exit in a fit is the experiment's RESULT — it is" >&2
    echo "  logged as adapt_band_exit=1 and the run should finish. Are you sure?" >&2
  fi
fi
if [ -n "$STD_BAND_RAISE_LO" ] || [ -n "$STD_BAND_RAISE_HI" ]; then
  if [ -z "$STD_BAND_RAISE_LO" ] || [ -z "$STD_BAND_RAISE_HI" ]; then
    echo "ERROR: set BOTH STD_BAND_RAISE_LO and STD_BAND_RAISE_HI or neither" >&2
    echo "  (got lo='${STD_BAND_RAISE_LO}' hi='${STD_BAND_RAISE_HI}'). The band is" >&2
    echo "  one treatment; loosening only the growth side leaves the collapse" >&2
    echo "  side — the one the runaway happens on — at the production rail." >&2
    exit 2
  fi
  if ! awk -v lo="$STD_BAND_RAISE_LO" -v hi="$STD_BAND_RAISE_HI" \
    'BEGIN { exit !(lo + 0 == lo && hi + 0 == hi && lo > 0 && lo < hi) }' \
    </dev/null 2>/dev/null; then
    echo "ERROR: STD_BAND_RAISE_LO='${STD_BAND_RAISE_LO}' / HI='${STD_BAND_RAISE_HI}'" >&2
    echo "  is not a band. They are MULTIPLES of the run's starting std, so they" >&2
    echo "  must be numeric with 0 < lo < hi (e.g. 0.01 and 100)." >&2
    exit 2
  fi
  RAILS_TAG="_rails"
  RAILS_ARGS_TUNE+=(--std-band-raise-lo "$STD_BAND_RAISE_LO"
    --std-band-raise-hi "$STD_BAND_RAISE_HI")
  RAILS_ARGS_FIT+=(--model.std_band_raise_lo "$STD_BAND_RAISE_LO"
    --model.std_band_raise_hi "$STD_BAND_RAISE_HI")
  if [ "$ADAPTIVE_NORM_FLAG" != "true" ]; then
    echo "WARN: STD_BAND_RAISE_* set but adaptive_norm is OFF. The band check runs" >&2
    echo "  inside the adaptive-norm EMA update, so with frozen stats the rails" >&2
    echo "  are inert — and the run dir still carries ${RAILS_TAG}." >&2
  fi
fi

SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"

# --- Val loop during a HOLDOUT fit ------------------------------------------
# joint_sr_trainval.yaml is layered LAST at the fit stage and sets
# `limit_val_batches: 0` UNCONDITIONALLY — correct under the merged protocol
# (val tiles are training tiles then, so a "val_iou" would be train IoU in
# disguise), but it also applies when TRAIN_SPLITS=train reverts to the holdout
# protocol, where val IS honestly held out and there is no reason not to look.
#
# FIT_VAL_LOOP=1 restores the loop for a holdout fit, and ONLY for one:
# requesting it under the merged protocol is refused rather than silently
# ignored. What it does NOT restore is any form of selection — the trainval
# overlay's callback list still has no EarlyStopping and its ModelCheckpoint
# still runs monitor:null / save_on_train_epoch_end, so the checkpoint remains
# the END of a fixed, pre-registered budget. The val loop only LOGS.
#
# That is why it carries no tag: it changes what is recorded, not what is
# trained. (One caveat worth knowing: the loop advances global RNG, so a
# FIT_VAL_LOOP=1 run is not step-for-step identical to the same seed without
# it. It is a between-arm constant wherever it is used, so no contrast moves.)
#
# The rl campaign needs it: "frozen val-AP plateaus before epoch 10" and "the
# joint arms' hold phases reproduce the frozen curves" are its falsifiable
# checks, and both are per-epoch val curves
# (docs/rl_lightning_campaign_plan.md §2, §6.1). DEFAULT 0 = the behaviour every
# existing arm and the loss pilot were run under; nothing is appended at 0.
FIT_VAL_LOOP="${FIT_VAL_LOOP:-0}"
VAL_ARGS_FIT=()
if [ "$FIT_VAL_LOOP" = "1" ]; then
  if [ "$MERGE_VAL" = "1" ]; then
    echo "ERROR: FIT_VAL_LOOP=1 with TRAIN_SPLITS='${TRAIN_SPLITS}'. val is" >&2
    echo "  folded into the training set under this protocol, so the curve would" >&2
    echo "  be a training curve wearing a val label. Use TRAIN_SPLITS=train." >&2
    exit 2
  fi
  VAL_ARGS_FIT=(--trainer.limit_val_batches "${LIMIT_VAL_BATCHES:-1.0}")
fi

# LABELS -> dataset dir + code-level mask_source (+ mask_dirname when raster).
# LABELS=new reads the ONCE-OFF pre-rasterised HR label COGs (the labels are
# frozen; per-crop graph rasterisation was the GPU-starving bottleneck).
# Generate them one time per dataset (standalone script, no PYTHONPATH/GPU;
# login node is fine):
#   $VENV_DIR/bin/python $REPO_DIR/src/sentinel2data/dataset/rasterize_hr_masks.py \
#     --dataset-dir <DATASET_DIR> --out-dirname mask_new_2pt5
# Submit-time MASK_SOURCE=graph reverts to on-the-fly rasterisation.
case "$LABELS" in
new)
  DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_New}"
  MASK_SOURCE="${MASK_SOURCE:-raster}" # pre-rasterised graph labels
  MASK_DIRNAME="${MASK_DIRNAME:-mask_new_2pt5}"
  ;;
all)
  DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_all}"
  MASK_SOURCE="graph"
  ;;
cdngi)
  DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
  MASK_SOURCE="graph"
  ;;
overture)
  DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_Overture}"
  MASK_SOURCE="graph"
  ;;
osm)
  DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_New}"
  MASK_SOURCE="raster"
  MASK_DIRNAME="${MASK_DIRNAME:-mask_osm_2pt5}"
  ;; # OSM HR rasters
*)
  echo "ERROR: LABELS must be new|all|cdngi|overture|osm, got '${LABELS}'." >&2
  exit 2
  ;;
esac
MASK_DIRNAME="${MASK_DIRNAME:-}" # empty for the graph (on-the-fly) sources

# --- Tune budget (train/val — UNCHANGED from _stages.sh) ---------------------
N_TRIALS="${N_TRIALS:-30}"
SEARCH_GPUS="${SEARCH_GPUS:-2}"
TUNE_EPOCHS="${TUNE_EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-imagenet}"
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-1e-2}"
LR_SR_MIN="${LR_SR_MIN:-1e-7}" # searched only when SR is learned & unfrozen
# 1e-3 was 100x the design default (1e-5) and the whole upper decade is
# known-wasted budget: on 2026-08-13 a sampled lr_sr=3.1e-4 drove the post-SR
# std out of its band inside 1,400 steps on SEN2SR -- the arm most resistant to
# this, since its FFT constraint pins the means. Under Adam the per-step weight
# displacement is ~lr regardless of gradient scale, so the rate at which the
# pretrained SR is destroyed is set by the ABSOLUTE lr_sr, not by lr_sr/lr.
# That is also why lr and lr_sr stay INDEPENDENTLY sampled rather than being
# reparametrised as a ratio alpha=lr_sr/lr: a ratio would couple the SR's
# destruction rate to the UNet's lr, dragging a trial that wants a fast UNet
# toward a destructive SR lr for no physical reason. Do not "simplify" it back.
LR_SR_MAX="${LR_SR_MAX:-1e-4}"
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-4.77222}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-4.77222}"
ENCODERS="${ENCODERS:-resnet34}" # NOT searched: encoder constancy is the control
BATCH_SIZES="${BATCH_SIZES:-4}"  # PINNED, not searched (2026-08-12). `length` is
# fixed per epoch, so a bs=1 trial takes 4x the
# optimiser steps of a bs=4 trial and wins the
# tune on step count alone -- batch size is a
# confound, not a hyperparameter. 4 is a
# between-arm constant for the whole SR series.
# NB never change this on a RESUME_FIT: it
# changes steps/epoch and breaks cosine T_max.

# --- Model-selection criterion (2026-08-16) ----------------------------------
# val_ap = threshold-free selection (binned AP; docs/ap_threshold_protocol_plan
# .md §1.1) for BOTH series. The Python default stays val_iou, so the loss
# pilot and every legacy path are byte-untouched — this shell default is what
# flips the _new-series protocol. MON_TAG goes into STUDY_NAME (an AP-era
# re-tune must never resume a val_iou-era study) and MODEL_NAME (the store is
# append-only; θ*-era rows must not be confusable with old θ=0.5 rows) — NOT
# into RUN_DIR, whose tag string _warm_tv.sh reconstructs for warm starts.
MONITOR="${MONITOR:-val_ap}"
MON_TAG=""
[ "$MONITOR" != "val_iou" ] && MON_TAG="_${MONITOR#val_}"

# --- Early stopping on a HOLDOUT fit (default OFF) ---------------------------
# joint_sr_trainval.yaml drops EarlyStopping on purpose: under the merged
# protocol there is no honest val signal to stop on, and a FIXED, pre-registered
# budget is what keeps the operating point from being data-dependent per arm
# (the loss ablation's fairness rule). Both reasons are protocol reasons, not
# engine reasons, so this knob re-adds the callback for the one case where
# neither applies: a HOLDOUT fit (TRAIN_SPLITS=train) that is already running
# the val loop (FIT_VAL_LOOP=1), where val is genuinely unseen.
#
# WHAT IT COSTS, stated once so no arm discovers it later:
#   * the budget stops being a between-arm constant. An early-stopped arm and a
#     full-budget arm did not train for the same number of epochs, so their
#     difference is no longer "the treatment" alone. ES_TAG marks it — in the
#     run dir, the study AND the bench model_name — for exactly that reason: an
#     early-stopped row must never land beside a fixed-budget row of the same
#     arm in an append-only store.
#   * the cosine no longer completes. T_max = max_epochs, so stopping at epoch
#     k < REFIT_EPOCHS ends the run at a non-zero lr, part-way down the
#     schedule. Harmless for a converged 5-parameter probe; NOT harmless for an
#     arm whose measured quantity is what the schedule was still doing.
#   * the checkpoint is still the LAST epoch, not the best one. The overlay's
#     ModelCheckpoint keeps `monitor: null` / `save_on_train_epoch_end`, so
#     early stopping changes WHERE the budget ends, never how the reported
#     checkpoint is chosen. No val-argmax selection is introduced.
#
# So: right for a frozen arm that provably plateaus (the rl campaign's rl3),
# wrong for any arm whose trajectory IS the result (rl4, the lr_sr grid, every
# joint arm) — there, a run that ends early has stopped recording the thing it
# was launched to record. That is not left to the caller's judgement: the block
# below WHITELISTS rl3 and refuses every other EXP_TAG, so no other arm can
# acquire a truncated budget or an _es row by a submit-time flag.
#
# Set it for EVERY STAGE of an arm, like the rails: ES_TAG lands in the run dir,
# so a tune tagged one way and a fit the other looks for best_params.yaml in a
# directory that does not exist.
FIT_EARLY_STOP="${FIT_EARLY_STOP:-0}"
ES_TAG=""
ES_ARGS_FIT=()
if [ "$FIT_EARLY_STOP" = "1" ]; then
  # WHITELIST: rl3 AND NOTHING ELSE.
  # Every other arm in every series runs the fixed, pre-registered budget, and
  # that is not a default anyone may opt out of at submit time — it is the
  # fairness rule the R-series, the loss ablation and the rl ladder all rest on.
  # rl3 is the single exception: a FROZEN generator with a 5-parameter probe,
  # whose plateau is pre-registered (campaign plan §6.1), so its late epochs
  # measure nothing. Allowing any other arm here would let one truncated budget
  # into a comparison as a submit-time typo, and it would mint an _es row for an
  # arm whose fixed-budget rows are what the write-up reports.
  # Widening this list is a protocol decision: make it in the plan first, and
  # only then here.
  case "$EXP_TAG" in
  rl3*) ;;
  *)
    echo "ERROR: FIT_EARLY_STOP=1 on EXP_TAG='${EXP_TAG}'. Early stopping is" >&2
    echo "  allowed for rl3 ONLY — the frozen-SR4RS probe arm, whose plateau" >&2
    echo "  before epoch 10 is pre-registered (docs/rl_lightning_campaign_plan" >&2
    echo "  .md §6.1) and whose generator cannot drift, so its late epochs" >&2
    echo "  measure nothing. Every other arm's budget is FIXED and identical" >&2
    echo "  across arms; truncating one would make its difference from the" >&2
    echo "  others 'the treatment plus a shorter run', and would put an _es row" >&2
    echo "  in a store whose rows for that arm are fixed-budget." >&2
    echo "  If this is a deliberate protocol change, write it into the plan and" >&2
    echo "  widen the whitelist in ${BASH_SOURCE[0]} — not at submit time." >&2
    exit 2
    ;;
  esac
  ES_MONITOR="${ES_MONITOR:-$MONITOR}"
  ES_MODE="${ES_MODE:-max}"
  ES_PATIENCE="${ES_PATIENCE:-5}"
  ES_MIN_DELTA="${ES_MIN_DELTA:-0.0}"
  if [ "$MERGE_VAL" = "1" ]; then
    echo "ERROR: FIT_EARLY_STOP=1 with TRAIN_SPLITS='${TRAIN_SPLITS}'. val is" >&2
    echo "  folded into the training set under this protocol, so stopping on" >&2
    echo "  '${ES_MONITOR}' would be stopping on a training metric. Use" >&2
    echo "  TRAIN_SPLITS=train (which also needs FIT_VAL_LOOP=1)." >&2
    exit 2
  fi
  if [ "$FIT_VAL_LOOP" != "1" ]; then
    echo "ERROR: FIT_EARLY_STOP=1 without FIT_VAL_LOOP=1. The fit would run no" >&2
    echo "  val loop at all (joint_sr_trainval.yaml sets limit_val_batches: 0)," >&2
    echo "  so '${ES_MONITOR}' would never be logged and EarlyStopping(strict)" >&2
    echo "  would abort the run at the first check." >&2
    exit 2
  fi
  case "$ES_MODE" in min | max) ;; *)
    echo "ERROR: ES_MODE must be min|max, got '${ES_MODE}'." >&2
    exit 2
    ;;
  esac
  ES_TAG="_es${ES_PATIENCE}"
  # `+=` APPENDS to the overlay's callback list rather than replacing it, so the
  # ModelCheckpoint / EpochSnapshotCheckpoint / LearningRateMonitor entries
  # survive; the init_args that follow bind to the callback just appended.
  # Verified against `sr.cli fit --print_config` (2026-08-30).
  ES_ARGS_FIT=(
    "--trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping"
    "--trainer.callbacks.init_args.monitor=${ES_MONITOR}"
    "--trainer.callbacks.init_args.mode=${ES_MODE}"
    "--trainer.callbacks.init_args.patience=${ES_PATIENCE}"
    "--trainer.callbacks.init_args.min_delta=${ES_MIN_DELTA}"
  )
fi

# Loader workers per training process: split the job's CPU allocation across
# the stage's processes (search fans out SEARCH_GPUS tuners; fit/test run one).
# Workers spend most time blocked on the prefetch queue, so no cores are
# reserved for the mains.
#
# THE OLD NOTE HERE ("with pre-rasterised masks, 1-2 workers already keep the
# GPU fed") WAS CALIBRATED ON THE U-NET ARMS AND EXPIRED WITH HEAD=linear.
# Those arms ran forward+backward through 24 M parameters per batch; an rl arm
# runs an SR forward under no_grad (freeze_sr) and backprops through FIVE
# parameters. GPU work per batch collapsed; bytes read per batch did not. The
# frozen rl arms are I/O-bound, so this number is now load-bearing — budget CPUs
# generously (--cpus-per-task 8+) rather than relying on the default 4.
#
# SEARCH_GPUS is CAPPED TO THE VISIBLE GPU COUNT INSIDE THE TUNE STAGE, which
# used to run AFTER this block: a job that asked for 2 GPUs and got 1 divided
# its CPUs by 2 anyway and ran half the workers it could afford. Resolve the cap
# here instead, before anything reads it.
if [ "${STAGE}" = "tune" ]; then
  # Deliberately NOT a one-liner. `grep -c` prints "0" AND exits 1 when it
  # matches nothing, so under `set -o pipefail` a
  #   $(command -v nvidia-smi && nvidia-smi ... | grep -c ... || echo 0)
  # fires BOTH the grep's "0" and the fallback's "0" and yields a two-line
  # value, which then makes `[ "$x" -gt 0 ]` emit "integer expression expected"
  # and quietly evaluate false — i.e. the cap silently stops working in exactly
  # the no-GPU case it exists to handle. (`set -e` does not catch it: the
  # assignment takes the status of the LAST command in the substitution, and a
  # failing `if` condition is exempt.)
  _VIS_GPUS=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    _VIS_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | grep -c '^GPU ' || true)
    _VIS_GPUS="${_VIS_GPUS//[!0-9]/}"        # strip anything not a digit
    [ -z "${_VIS_GPUS}" ] && _VIS_GPUS=0
  fi
  if [ "${_VIS_GPUS}" -gt 0 ] && [ "${SEARCH_GPUS}" -gt "${_VIS_GPUS}" ]; then
    echo "NOTE: SEARCH_GPUS=${SEARCH_GPUS} but ${_VIS_GPUS} GPU(s) visible — capping now"
    echo "  (before NUM_WORKERS is derived from it)."
    SEARCH_GPUS="${_VIS_GPUS}"
  fi
fi
if [ -z "${NUM_WORKERS}" ]; then
  JOB_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-4}}"
  if [ "${STAGE}" = "tune" ]; then
    NUM_WORKERS=$((JOB_CPUS / SEARCH_GPUS))
  else
    NUM_WORKERS=$((JOB_CPUS - 1))
  fi
  [ "${NUM_WORKERS}" -lt 1 ] && NUM_WORKERS=1
fi
echo "loader: num_workers=${NUM_WORKERS} (job_cpus=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-4}}, search_gpus=${SEARCH_GPUS})"

# --- Fit budget (train+val, FIXED — no early stopping by default) ------------
# Pre-registered and identical across arms. Nothing truncates it unless an arm
# opts into FIT_EARLY_STOP=1 above (holdout fits only, and tagged when it does),
# so budget the SLURM walltime for the full count on the SLOWEST arm (sr4rs).
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_joint_final}"

# --- Loss (unet.losses.build_loss; empty = legacy Dice + pos-weighted BCE) ---
LOSS_ARM="${LOSS_ARM:-gap_ce}"
PSTAR="${PSTAR:-gap_ce}"
GAP_R="${GAP_R:-4}"
GAP_K="${GAP_K:-60.0}"
TL_ELL="${TL_ELL:-5}"
TL_THETA="${TL_THETA:-0.375}"
GAP_THETA="${GAP_THETA:-0.55836}"
# official gap binarization
# R-SERIES RULE: the loss is a FROZEN CONTROL across R-arms. Pin the pilot
# winner's config at submit time: SEARCH_THETAS=false TL_THETA=<θ*>
# GAP_THETA=<θ*> POS_WEIGHT_MIN=<λ*> POS_WEIGHT_MAX=<λ*> (min==max = a
# constant). Leaving SEARCH_THETAS=true re-searches loss hps per R-arm and
# confounds the SR comparison.
SEARCH_THETAS="${SEARCH_THETAS:-false}"
# mix_w — the P*<->region ratio of the pstar_* compounds (2026-08-05). Searched
# for those arms only (consumption-gated in sr.tune, same rule as the θs); the
# bce_dice anchor stays frozen at 0.5/0.5 by build_loss's design. Kept in sync
# with the Lightning twin.
SEARCH_MIX_W="${SEARCH_MIX_W:-false}"
MIX_W="${MIX_W:-0.6075946831862098}"
MIX_W_MIN="${MIX_W_MIN:-0.25}"
MIX_W_MAX="${MIX_W_MAX:-0.75}"
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
CL_ALPHA="${CL_ALPHA:-0.3}"
CL_ITERS="${CL_ITERS:-5}"
SKEL_W="${SKEL_W:-1.0}"
SKEL_RADIUS="${SKEL_RADIUS:-1}"
WARMUP_START="${WARMUP_START:-30}"
WARMUP_RAMP="${WARMUP_RAMP:-10}"

LOSS_TAG=""
LOSS_ARGS_TUNE=() # sr.tune flags (argparse)
LOSS_ARGS_FIT=()  # sr.cli fit/test flags (LightningCLI --model.*)
if [ -n "$LOSS_ARM" ]; then
  # '+' is not filesystem/wandb-friendly -> bce_dice+cldice => bce_dice-cldice
  LOSS_TAG="_$(echo "$LOSS_ARM" | tr '+' '-')"
  LOSS_ARGS_TUNE=(--loss-arm "$LOSS_ARM" --pstar "$PSTAR"
    --gap-r "$GAP_R" --gap-k "$GAP_K"
    --tl-ell "$TL_ELL" --tl-theta "$TL_THETA"
    --gap-theta "$GAP_THETA" --search-thetas "$SEARCH_THETAS"
    --mix-w "$MIX_W" --search-mix-w "$SEARCH_MIX_W"
    --mix-w-min "$MIX_W_MIN" --mix-w-max "$MIX_W_MAX"
    --tversky-alpha "$TVERSKY_ALPHA"
    --cl-alpha "$CL_ALPHA" --cl-iters "$CL_ITERS"
    --skel-w "$SKEL_W" --skel-radius "$SKEL_RADIUS"
    --warmup-start "$WARMUP_START" --warmup-ramp "$WARMUP_RAMP")
  # NB tl_theta/gap_theta/pos_weight are NOT in the fit belt: the tune pins
  # them (searched or fixed) into best_params.yaml, and an explicit --model.*
  # here would override the pinned values with the env defaults. The overlay
  # is authoritative for those dims. (Ported from the LS twin, 2026-08-04.)
  LOSS_ARGS_FIT=(--model.loss_arm "$LOSS_ARM" --model.pstar "$PSTAR"
    --model.gap_r "$GAP_R" --model.gap_k "$GAP_K"
    --model.tl_ell "$TL_ELL"
    --model.tversky_alpha "$TVERSKY_ALPHA"
    --model.cl_alpha "$CL_ALPHA" --model.cl_iters "$CL_ITERS"
    --model.sr_w "$SKEL_W" --model.sr_radius "$SKEL_RADIUS"
    --model.warmup_start "$WARMUP_START" --model.warmup_ramp "$WARMUP_RAMP")
fi
# =============================================================================

BASE_CONFIG="$REPO_DIR/src/sr/configs/joint_sr.yaml"
TRAINVAL_CONFIG="$REPO_DIR/src/sr/configs/joint_sr_trainval.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"

# --- Norm stats: read them from the DATASET, not from the repo ---------------
# `sentinel2data.cli norm-stats` writes <dataset_dir>/norm_stats.yaml by
# default, so every dataset already ships the stats computed from ITS OWN
# splits/train.csv — which is the only file that can be correct for it.
# The repo copy at src/unet/configs/norm_stats.yaml is a hand-copy of one
# dataset's file (its header still names the dataset it came from); pointing
# every experiment at that single path means the stats silently stop matching
# the moment you switch datasets, and nothing in the run would tell you.
# So: prefer the dataset's own file, fall back to the repo copy only if the
# dataset has none, and say loudly which one is in use. NORM_CONFIG=<path>
# overrides both.
#
# RUNS_ROOT is shared with _warm_tv.sh, which reconstructs the STAGE-1 run dir
# from it — override one and you must override both, so they read the same var.
# (Resolved BEFORE the norm stats so the train+val generation below has a
# guaranteed-writable fallback location.)
RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"
RUN_DIR="${RUNS_ROOT}/sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${RAILS_TAG}${ES_TAG}${PROTO_TAG}_seed${SEED}"

# --- name query: PRINT_RUN_DIR=1 ---------------------------------------------
# A pool driver has to know a cell's RUN_DIR and MODEL_NAME BEFORE running it,
# to decide whether that stage is already done. Re-deriving the tag chain in the
# driver is precisely the footgun the refit scripts' RUN_TAG comment warns
# about: one tag out of sync and the guard inspects a directory the run will
# never write, so every stage looks "not done" and a finished 100-epoch refit is
# silently redone. So the names are asked for, not reconstructed.
#
# Prints and exits — no mkdir, no norm stats, no venv, nothing. Cheap enough to
# call per stage. Every tag it interpolates is resolved above this line; if a
# new tag is ever added BELOW it, add it here too or this lies.
if [ "${PRINT_RUN_DIR:-0}" = "1" ]; then
  echo "RUN_DIR=${RUN_DIR}"
  echo "MODEL_NAME=${MODEL_NAME:-sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${RAILS_TAG}${ES_TAG}${PROTO_TAG}${MON_TAG}}"
  exit 0
fi

mkdir -p "$RUN_DIR"

# --- Planted overlay: an arm that carries its own best_params.yaml -----------
# Normally STAGE=tune writes best_params.yaml and STAGE=fit consumes it. An arm
# that searches NOTHING has no reason to pay for that: the rl campaign pins the
# head lr, the loss, λ, the batch size and (per rung) lr_sr, so its "tune" was a
# 1-trial 1-epoch pass whose only product was a file of constants it was handed.
# Such an arm can instead set BEST_PARAMS to the overlay text and skip the stage
# entirely — the same idiom the seed-refit scripts use
# (scripts/hpc/sr/refit/_refit_lib.sh), one level earlier.
#
# WHAT THE ARM OWES: an overlay byte-equivalent to what sr.tune would have
# written for those constants — `sr.tune.write_best_overlay` is the schema's
# owner and a hand-written copy is a second author. Pin it with a test
# (tests/test_rl_campaign_hpc.py compares each arm's heredoc against
# write_best_overlay's output for the same pinned params) rather than by reading.
#
# NOT PLANTED AT STAGE=tune: a search writes its own overlay, and quietly
# planting one first would leave a file the study is about to overwrite — a
# tune that ran and a tune that did not would be indistinguishable afterwards.
if [ -n "${BEST_PARAMS:-}" ] && [ "$STAGE" != "tune" ]; then
  _PLANT="${RUN_DIR}/best_params.yaml"
  if [ -f "$_PLANT" ] && ! printf '%s\n' "$BEST_PARAMS" | cmp -s - "$_PLANT"; then
    # A different overlay is already here. If weights were trained under it,
    # replacing it silently would make the run dir describe a config that did
    # not produce its checkpoints — the one thing an overlay exists to prevent.
    if [ -f "${RUN_DIR}/checkpoints/unet_s2rosa_jointsr_final.ckpt" ] \
       || [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
      echo "ERROR: ${_PLANT} differs from this arm's planted overlay, and this" >&2
      echo "  run dir already holds checkpoints — they were trained under the" >&2
      echo "  file on disk, not under the heredoc in the arm script." >&2
      echo "  Diff them, then either restore the arm's constants or start a new" >&2
      echo "  run dir (EXP_TAG=... / SEED=...). OVERWRITE_PARAMS=1 replaces it" >&2
      echo "  anyway, which orphans those checkpoints from their config." >&2
      if [ "${OVERWRITE_PARAMS:-0}" != "1" ]; then exit 2; fi
    else
      echo "NOTE: replacing ${_PLANT} (no checkpoints trained under it yet)."
    fi
  fi
  printf '%s\n' "$BEST_PARAMS" > "$_PLANT"
  echo "planted overlay: ${_PLANT} (this arm searches nothing; STAGE=tune not required)"
fi

# §4.7 stats provenance under the train+val refit: norm_stats.yaml is computed
# on the TRAIN split, but the refit trains on train+val. NORM_TV=1 switches the
# FIT stage to <dataset>/norm_stats_tv.yaml. Deliberately OPT-IN: the convention
# must be held CONSTANT across every arm inside a comparison, so flipping it
# silently mid-series would void the series. The tune stage always keeps
# train-only stats — val is a holdout there. Test zones never contribute under
# either convention. Bench does not read norm stats at all (it restores them
# from the checkpoint), so NORM_TV is a fit-stage concern only.
#
# If the file is missing it is GENERATED, not treated as an error: a hard fail
# here burns a whole GPU allocation on a one-line omission. Three properties
# make the auto-generation safe on a shared filesystem with many arms in flight:
#
#   * DETERMINISTIC — the numbers are a streaming reduction over the tiles
#     listed in splits/{train,val}.csv, so two jobs that generate it
#     concurrently produce byte-identical output. A race cannot yield arms
#     trained under disagreeing stats, which is the only failure that would
#     actually matter.
#   * ATOMIC PUBLISH — written to a per-PID temp file and `mv`d into place
#     (same filesystem, so the rename is atomic). No job can ever read a
#     half-written YAML.
#   * SINGLE SCAN — an mkdir lock (atomic on POSIX) means one job does the I/O
#     while the others wait for the file to appear. Waiters take over if the
#     holder dies, so a killed job cannot wedge the queue.
#
# NORM_TV_AUTO=0 restores the old hard failure for anyone who would rather be
# told than have a file appear underneath them.
NORM_CONFIG_DATASET="${DATASET_DIR}/norm_stats.yaml"
NORM_CONFIG_TV="${DATASET_DIR}/norm_stats_tv.yaml"
NORM_CONFIG_REPO="$REPO_DIR/src/unet/configs/norm_stats.yaml"
NORM_TV_WAIT="${NORM_TV_WAIT:-1800}" # s to wait on another job's generation
USE_TV_STATS=0
if [ "${NORM_TV:-0}" = "1" ] && [ "$STAGE" = "fit" ] && [ "$MERGE_VAL" = "1" ]; then
  USE_TV_STATS=1
elif [ "${NORM_TV:-0}" = "1" ]; then
  echo "NOTE: NORM_TV=1 ignored for stage='${STAGE}' (merge_val=${MERGE_VAL})."
  echo "  Train+val stats apply to the REFIT only: tune scores on val as a"
  echo "  holdout, and bench restores the stats from the checkpoint."
fi

generate_tv_stats() { # $1 = destination path; echoes nothing, returns 0/1
  local dest="$1" tmp="$1.tmp.$$" t0 rc
  t0=$(date +%s)
  echo "  generating $(basename "$dest") over train+val ..."
  PYTHONPATH="$REPO_DIR/src" "$VENV_DIR/bin/python" -m sentinel2data.cli norm-stats \
    --dataset-dir "$DATASET_DIR" --splits train --splits val --out "$tmp"
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$tmp" ]; then
    rm -f "$tmp"
    return 1
  fi
  mv -f "$tmp" "$dest" || {
    rm -f "$tmp"
    return 1
  }
  echo "  wrote ${dest} in $(($(date +%s) - t0))s"
  return 0
}

if [ "$USE_TV_STATS" = "1" ] && [ -z "${NORM_CONFIG:-}" ] && [ ! -f "$NORM_CONFIG_TV" ]; then
  if [ "${NORM_TV_AUTO:-1}" != "1" ]; then
    echo "ERROR: NORM_TV=1 but ${NORM_CONFIG_TV} does not exist, and" >&2
    echo "  NORM_TV_AUTO=0 disabled generating it. Create it with:" >&2
    echo "    ${VENV_DIR}/bin/python -m sentinel2data.cli norm-stats \\" >&2
    echo "      --dataset-dir ${DATASET_DIR} --splits train --splits val \\" >&2
    echo "      --out ${NORM_CONFIG_TV}" >&2
    exit 2
  fi
  echo "NORM_TV=1: ${NORM_CONFIG_TV} not found — generating it."
  NORM_TV_LOCK="${NORM_CONFIG_TV}.lock"
  if mkdir "$NORM_TV_LOCK" 2>/dev/null; then
    trap 'rmdir "'"$NORM_TV_LOCK"'" 2>/dev/null || true' EXIT
    if ! generate_tv_stats "$NORM_CONFIG_TV"; then
      # Read-only dataset dir, quota, whatever. Fall back to a run-local copy:
      # the CONTENT is identical either way (same deterministic reduction over
      # the same split CSVs), so cross-arm comparability is preserved — the
      # only cost is that each arm recomputes it.
      echo "WARN: could not write ${NORM_CONFIG_TV} — falling back to a" >&2
      echo "  run-local copy under ${RUN_DIR}. Content is identical (the" >&2
      echo "  computation is deterministic), so arms stay comparable; only the" >&2
      echo "  redundant rescan is lost. Promote it into the dataset dir to fix." >&2
      NORM_CONFIG_TV="${RUN_DIR}/norm_stats_tv.yaml"
      if ! generate_tv_stats "$NORM_CONFIG_TV"; then
        echo "ERROR: train+val norm-stats generation failed. See above." >&2
        exit 2
      fi
    fi
    rmdir "$NORM_TV_LOCK" 2>/dev/null || true
    trap - EXIT
  else
    echo "  another job holds ${NORM_TV_LOCK}; waiting up to ${NORM_TV_WAIT}s ..."
    _waited=0
    while [ ! -f "$NORM_CONFIG_TV" ] && [ "$_waited" -lt "$NORM_TV_WAIT" ]; do
      sleep 10
      _waited=$((_waited + 10))
    done
    if [ ! -f "$NORM_CONFIG_TV" ]; then
      # The holder died (or is slower than the wait). Take the lock over rather
      # than wedging the queue — worst case two jobs write identical bytes.
      echo "  waited ${_waited}s with no file; assuming a dead holder and" >&2
      echo "  generating it here instead." >&2
      rmdir "$NORM_TV_LOCK" 2>/dev/null || true
      generate_tv_stats "$NORM_CONFIG_TV" || {
        echo "ERROR: train+val norm-stats generation failed. See above." >&2
        exit 2
      }
    else
      echo "  ${NORM_CONFIG_TV} appeared after ${_waited}s."
    fi
  fi
fi

if [ -n "${NORM_CONFIG:-}" ]; then
  NORM_SOURCE="explicit NORM_CONFIG override"
elif [ "$USE_TV_STATS" = "1" ] && [ -f "$NORM_CONFIG_TV" ]; then
  NORM_CONFIG="$NORM_CONFIG_TV"
  NORM_SOURCE="dataset train+val (NORM_TV=1)"
elif [ -f "$NORM_CONFIG_DATASET" ]; then
  NORM_CONFIG="$NORM_CONFIG_DATASET"
  NORM_SOURCE="dataset"
else
  NORM_CONFIG="$NORM_CONFIG_REPO"
  NORM_SOURCE="repo fallback"
fi

# The refit's checkpoint. Named _final, never _best: under this protocol no
# checkpoint was ever selected on a holdout, and the filename says so.
FINAL_CKPT_NAME="unet_s2rosa_jointsr_final"

LOG_FILE="${RUN_DIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"
echo "host=$(hostname)  exp=sr/${EXP_TAG}  stage=${STAGE}  seed=${SEED}  head=${HEAD}"
echo "labels=${LABELS} (mask_source=${MASK_SOURCE}${MASK_DIRNAME:+, mask_dirname=${MASK_DIRNAME}})  upsampler=${UPSAMPLER}  freeze_sr=${FREEZE_SR}  sr_pad=${SR_PAD}  loss_arm=${LOSS_ARM:-legacy}"
echo "hard constraint: sr_hc=${SR_HC}${HC_TAG:+  tag=${HC_TAG}}${HC_MASK_PATH:+  mask=${HC_MASK_PATH}}"
echo "recipe: reg=${REG}${REG_TAG:+ [${REG_TAG}]}  clip=${CLIP} (trainer=${CLIP_TRAINER}, sr_group=${CLIP_SR})  lr_schedule=${LR_SCHEDULE}  sr_warmup_epochs=${SR_WARMUP_EPOCHS}  l2sp_lambda=${L2SP_LAMBDA}  sr_snapshot_every=${SR_SNAPSHOT_EVERY}"
echo "head=${HEAD}  monitor=${MONITOR}  warm_start_head=${WARM_START_HEAD:-none}"
echo "sr_warmup_epochs=${SR_WARMUP_EPOCHS}  sr_hold_epochs=${SR_HOLD_EPOCHS}  lr_schedule=${LR_SCHEDULE}"
echo "adapter: adaptive_norm=${ADAPTIVE_NORM_FLAG} (m=${ADAPTIVE_NORM_M})  norm_recalibrate=${NORM_RECALIBRATE}${ANORM_TAG:+  tag=${ANORM_TAG}}"
if [ -n "$RAILS_TAG" ]; then
  _RAILS_DESC="production (the config's, [0.5x, 4.0x] unless joint_sr.yaml moved it)"
  if [ -n "$STD_BAND_RAISE_LO" ]; then
    _RAILS_DESC="[${STD_BAND_RAISE_LO}x, ${STD_BAND_RAISE_HI}x] of the starting std"
  fi
  echo "std band guard: rails=${_RAILS_DESC}  action=${STD_BAND_ACTION:-default (tune prunes, fit warns)}  tag=${RAILS_TAG}  (docs/lrsr_grid_ablation_plan.md §3)"
else
  echo "std band guard: production band from the config ([0.5x, 4.0x] unless joint_sr.yaml moved it), action=default (tune prunes the trial, fit warns and continues)"
fi
echo "protocol: tune on train/val -> refit on '${TRAIN_SPLITS}' (merge_val=${MERGE_VAL}) -> report on test"
echo "fit val loop: $([ "$FIT_VAL_LOOP" = "1" ] && echo "ON (holdout curves logged, never selected on)" || echo "off (joint_sr_trainval.yaml limit_val_batches=0)")"
if [ "$FIT_EARLY_STOP" = "1" ]; then
  echo "fit early stopping: ON  monitor=${ES_MONITOR} mode=${ES_MODE} patience=${ES_PATIENCE} min_delta=${ES_MIN_DELTA}  tag=${ES_TAG}  (budget is a CEILING for this arm, not a constant)"
else
  echo "fit early stopping: off (fixed ${REFIT_EPOCHS}-epoch budget)"
fi
echo "DATASET_DIR=${DATASET_DIR}  warm_start=${WARM_START_CKPT:-none}"
echo "norm_stats=${NORM_CONFIG}  [${NORM_SOURCE}]"

# --- Fail fast ---------------------------------------------------------------
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  echo "  (LABELS=${LABELS}. Upload the final dataset, or override DATASET_DIR.)" >&2
  exit 1
fi
for _s in splits/train.csv splits/val.csv splits/test.csv; do
  if [ ! -f "${DATASET_DIR}/${_s}" ]; then
    echo "ERROR: ${DATASET_DIR}/${_s} missing — the train+val protocol needs all" >&2
    echo "  three split CSVs (val is merged at fit; test is the only report set)." >&2
    exit 1
  fi
done
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: no norm stats for this dataset." >&2
  echo "  Looked for: ${NORM_CONFIG_DATASET}" >&2
  echo "  and:        ${NORM_CONFIG_REPO}" >&2
  echo "  Generate the dataset's own (this is the default output path):" >&2
  echo "    python -m sentinel2data.cli norm-stats --dataset-dir ${DATASET_DIR}" >&2
  exit 1
fi
# The repo fallback belongs to whichever dataset it was last copied from, so it
# is a coin flip on any other one. Refuse to guess silently.
if [ "${NORM_SOURCE}" = "repo fallback" ]; then
  echo "WARN: ${DATASET_DIR}/norm_stats.yaml does not exist; falling back to the" >&2
  echo "  repo copy ${NORM_CONFIG_REPO}, which was computed from a DIFFERENT" >&2
  echo "  dataset's splits/train.csv. Wrong mean/std shifts every input the model" >&2
  echo "  ever sees, and nothing downstream would flag it. Generate the real one:" >&2
  echo "    python -m sentinel2data.cli norm-stats --dataset-dir ${DATASET_DIR}" >&2
  echo "  (NORM_FALLBACK_OK=1 to proceed anyway.)" >&2
  if [ "${NORM_FALLBACK_OK:-0}" != "1" ]; then
    exit 1
  fi
  echo "  NORM_FALLBACK_OK=1 — proceeding on the repo copy." >&2
fi
if [ ! -f "${TRAINVAL_CONFIG}" ]; then
  echo "ERROR: ${TRAINVAL_CONFIG} missing — this engine needs the refit overlay." >&2
  exit 1
fi
case "${UPSAMPLER}" in
sen2sr | sen2sr_full)
  if [ ! -f "${SEN2SR_DIR}/model.safetensor" ]; then
    echo "ERROR: SEN2SR weights not at ${SEN2SR_DIR} (upsampler=${UPSAMPLER})." >&2
    echo "  Lite: prefetch with sr.sen2sr_loader.download_sen2sr on a login node;" >&2
    echo "  full: download the SEN2SR (Mamba) mlstac dir there yourself." >&2
    exit 1
  fi
  ;;
sr4rs)
  if [ ! -f "${SEN2SR_DIR}/gen_weights.safetensors" ]; then
    echo "ERROR: SR4RS extracted weights not at ${SEN2SR_DIR}/gen_weights.safetensors." >&2
    echo "  Run scripts/sr4rs/extract_sr4rs.py locally (TF venv), verify with" >&2
    echo "  'python -m sr.sr4rs_torch --model-dir ...', then upload the three" >&2
    echo "  gen_* files into ${SEN2SR_DIR}." >&2
    exit 1
  fi
  ;;
esac
if [ -n "${HC_MASK_PATH}" ] && [ ! -f "${HC_MASK_PATH}" ]; then
  echo "ERROR: HC_MASK_PATH=${HC_MASK_PATH} not found on $(hostname)." >&2
  echo "  It ships inside the SEN2SR-Lite model dir; prefetch that dir with" >&2
  echo "  sr.sen2sr_loader.download_sen2sr on a login node (the r2 arms already" >&2
  echo "  need it), or point HC_MASK_PATH at wherever it landed." >&2
  exit 1
fi
if [ -n "${WARM_START_CKPT}" ] && [ ! -f "${WARM_START_CKPT}" ]; then
  echo "ERROR: WARM_START_CKPT=${WARM_START_CKPT} not found — run the stage-1" >&2
  echo "  (frozen-SR) arm's STAGE=fit first; its final ckpt seeds this arm's UNet." >&2
  exit 1
fi
if [ -n "${WARM_START_HEAD}" ] && [ ! -f "${WARM_START_HEAD}" ]; then
  echo "ERROR: WARM_START_HEAD=${WARM_START_HEAD} not found — run the frozen twin" >&2
  echo "  arm's STAGE=fit first; its final ckpt seeds this arm's linear probe." >&2
  exit 1
fi
if [ -n "${WARM_START_HEAD}" ] && [ -n "${WARM_START_CKPT}" ]; then
  echo "ERROR: WARM_START_HEAD and WARM_START_CKPT are both set. The two warm-start" >&2
  echo "  paths are mutually exclusive: warm_start_unet auto-disables the SR warmup" >&2
  echo "  ramp, which the head path must keep (docs/sr_linear_probe.md §2)." >&2
  exit 2
fi
if [ -n "${WARM_START_HEAD}" ] && [ "$HEAD" != "linear" ]; then
  echo "ERROR: WARM_START_HEAD is set but HEAD=${HEAD}. There is no linear probe to" >&2
  echo "  warm-start. Did you mean WARM_START_CKPT (the staged U-Net path)?" >&2
  exit 2
fi
if [ "${MASK_SOURCE}" = "raster" ]; then
  # -print -quit: no pipe to `head`, so `find` can't die of SIGPIPE and trip
  # `set -o pipefail`.
  first_mask=$(find "${DATASET_DIR}"/*/"${MASK_DIRNAME}" -maxdepth 1 -name '*.tif' -print -quit 2>/dev/null)
  if [ -z "${first_mask}" ]; then
    echo "ERROR: MASK_SOURCE=raster but no masks under <split>/${MASK_DIRNAME}/." >&2
    if [ "${LABELS}" = "osm" ]; then
      echo "  Generate with OpenStreetMapTest/dataset_hr_masks.py --scale 4" >&2
    else
      echo "  Generate ONCE with (standalone, login node is fine):" >&2
      echo "    ${VENV_DIR}/bin/python ${REPO_DIR}/src/sentinel2data/dataset/rasterize_hr_masks.py \\" >&2
      echo "      --dataset-dir ${DATASET_DIR} --out-dirname ${MASK_DIRNAME}" >&2
      echo "  (or MASK_SOURCE=graph to rasterise on the fly — slow.)" >&2
    fi
    exit 1
  fi
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "python=$(which python)"

# --- HEAD=linear capability preflight ----------------------------------------
# The rl-series depends on model/tune changes that are NOT part of the U-Net
# path (docs/sr_linear_probe.md §4). If they are absent, every flag this engine
# passes for HEAD=linear is either rejected by argparse or — worse, for the
# LightningCLI path — could be ignored, and the arm would quietly train a 24 M
# -param U-Net under an `rl*` tag. That row would then sit in the append-only
# store looking like a linear probe. Refuse to start instead.
if [ "$HEAD" = "linear" ]; then
  _missing=$(
    python - <<'PY'
import inspect
missing = []
try:
    from sr.model import JointSRUNetLightning
    params = inspect.signature(JointSRUNetLightning.__init__).parameters
    for name in ("head", "warm_start_head", "clip_sr"):
        if name not in params:
            missing.append(f"JointSRUNetLightning.__init__({name}=...)")
    # hasattr is useless here: LightningModule defines configure_gradient_clipping
    # as a no-op base method, so it is ALWAYS present and an unimplemented
    # per-group clip would sail through. Compare identities instead.
    import lightning.pytorch as pl
    if (JointSRUNetLightning.configure_gradient_clipping
            is pl.LightningModule.configure_gradient_clipping):
        missing.append("JointSRUNetLightning.configure_gradient_clipping override "
                       "(base method is inherited unchanged — per-group clipping "
                       "is NOT implemented)")
except Exception as exc:                      # import error = missing anyway
    missing.append(f"sr.model import failed: {exc}")
try:
    from sr.tune import parse_args
    # parse_args() builds the parser inline, so introspect it the only way that
    # does not require inventing an argv: let argparse render its own help.
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    helptext = buf.getvalue()
    for flag in ("--head", "--warm-start-head", "--clip-sr"):
        if flag not in helptext:
            missing.append(f"sr.tune {flag}")
except Exception as exc:
    missing.append(f"sr.tune introspection failed: {exc}")
print("\n".join(missing))
PY
  )
  if [ -n "${_missing}" ]; then
    echo "ERROR: HEAD=linear, but the linear-probe support is not in this checkout." >&2
    echo "  Missing:" >&2
    echo "${_missing}" | sed 's/^/    - /' >&2
    echo "" >&2
    echo "  Implement docs/sr_linear_probe.md §4 before running any rl arm:" >&2
    echo "    src/sr/model.py  head hparam ('unet'|'linear'); LinearProbeHead (1x1" >&2
    echo "                     conv, bias init = logit(road base rate), weights 0);" >&2
    echo "                     skip build_model when linear; warm_start_head" >&2
    echo "                     (SEPARATE from warm_start_unet — §2); clip_sr with a" >&2
    echo "                     configure_gradient_clipping override (§6.2); key the" >&2
    echo "                     fp32 autocast island on head=='linear', NOT on the" >&2
    echo "                     presence of an SR net (§6.3 — this is the rl0 bug)." >&2
    echo "    src/sr/tune.py   --head / --warm-start-head / --clip-sr; drop the" >&2
    echo "                     encoder_name categorical when linear." >&2
    echo "" >&2
    echo "  Then run Gates A and A2 (§10) locally before spending cluster time." >&2
    exit 2
  fi
  echo "preflight: linear-probe support present."
fi

if [ "${UPSAMPLER}" = "sen2sr_full" ] && ! python -c "import mamba_ssm" 2>/dev/null; then
  echo "ERROR: upsampler=sen2sr_full but mamba_ssm is not importable in ${VENV_DIR}." >&2
  echo "  Install on a GPU node with matching torch/CUDA:  uv pip install mamba-ssm" >&2
  exit 1
fi

# ============================== STAGE: tune ==================================
# IDENTICAL to _stages.sh: the search trains on `train` and scores on `val`.
# The holdout is spent here, deliberately and once.
if [ "$STAGE" = "tune" ]; then
  STORAGE="${STORAGE:-sqlite:///${RUN_DIR}/study.db}"
  SAMPLER_OFFSET="${SAMPLER_OFFSET:-0}"
  STUDY_NAME="sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${RAILS_TAG}${ES_TAG}${PROTO_TAG}${MON_TAG}_seed${SEED}"

  run_tuner() { # $1=gpu id (empty = no pin)  $2=n-trials  $3=seed
    local gpu="$1" ntrials="$2" seed="$3" pin=""
    [ -n "$gpu" ] && pin="CUDA_VISIBLE_DEVICES=$gpu"
    env $pin python -m sr.tune \
      --base-config "$BASE_CONFIG" \
      --base-config "$NORM_CONFIG" \
      --dataset-dir "$DATASET_DIR" \
      --sen2sr-dir "$SEN2SR_DIR" \
      --mask-source "$MASK_SOURCE" \
      ${MASK_DIRNAME:+--mask-dirname "$MASK_DIRNAME"} \
      --upsampler "$UPSAMPLER" \
      --freeze-sr "$FREEZE_SR" \
      --sr-pad "$SR_PAD" \
      ${HC_ARGS_TUNE[@]+"${HC_ARGS_TUNE[@]}"} \
      ${WARM_START_CKPT:+--warm-start-unet "$WARM_START_CKPT"} \
      ${HEAD_TAG:+--head "$HEAD"} \
      ${HEAD_TAG:+--clip-sr "$CLIP_SR"} \
      ${WARM_START_HEAD:+--warm-start-head "$WARM_START_HEAD"} \
      --out "$RUN_DIR" \
      --num-workers "$NUM_WORKERS" \
      --devices 1 \
      --n-trials "$ntrials" \
      --max-epochs "$TUNE_EPOCHS" \
      --patience "$PATIENCE" \
      --precision "$PRECISION" \
      --clip "$CLIP_TRAINER" \
      --lr-schedule "$LR_SCHEDULE" \
      --sr-warmup-epochs "$SR_WARMUP_EPOCHS" \
      ${HOLD_ARGS_TUNE[@]+"${HOLD_ARGS_TUNE[@]}"} \
      --l2sp-lambda "$L2SP_LAMBDA" \
      --adaptive-norm "$ADAPTIVE_NORM_FLAG" \
      --adaptive-norm-momentum "$ADAPTIVE_NORM_M" \
      --norm-recalibrate "$NORM_RECALIBRATE" \
      ${RAILS_ARGS_TUNE[@]+"${RAILS_ARGS_TUNE[@]}"} \
      --seed "$seed" \
      --train-seed "$SEED" \
      --study-name "$STUDY_NAME" \
      --storage "$STORAGE" \
      --monitor "$MONITOR" \
      --encoder-weights "$ENCODER_WEIGHTS" \
      --lr-min "$LR_MIN" --lr-max "$LR_MAX" \
      --lr-sr-min "$LR_SR_MIN" --lr-sr-max "$LR_SR_MAX" \
      --pos-weight-min "$POS_WEIGHT_MIN" --pos-weight-max "$POS_WEIGHT_MAX" \
      --encoders $ENCODERS \
      --batch-sizes $BATCH_SIZES \
      ${LOSS_ARGS_TUNE[@]+"${LOSS_ARGS_TUNE[@]}"}
  }

  N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
  if [ "${N_GPUS}" -eq 0 ] && [ "${SEARCH_GPUS}" -gt 1 ]; then
    echo "WARN: no CUDA device visible — running ONE unpinned tuner on CPU (smoke-test mode)." >&2
    SEARCH_GPUS=1
  elif [ "${N_GPUS}" -gt 0 ] && [ "${SEARCH_GPUS}" -gt "${N_GPUS}" ]; then
    echo "WARN: SEARCH_GPUS=${SEARCH_GPUS} but only ${N_GPUS} GPU(s) visible — capping to ${N_GPUS}." >&2
    SEARCH_GPUS="${N_GPUS}"
  fi

  echo "=== OPTUNA SEARCH on train/val (n_trials=$N_TRIALS across ${SEARCH_GPUS} GPU(s), ${TUNE_EPOCHS} epochs/trial) ==="
  echo "    stop early (keeps study + writes overlay):  touch ${RUN_DIR}/STOP"
  if [ "$SEARCH_GPUS" -le 1 ]; then
    run_tuner "" "$N_TRIALS" "$((SEED * 1000 + SAMPLER_OFFSET))"
  else
    PER_WORKER=$(((N_TRIALS + SEARCH_GPUS - 1) / SEARCH_GPUS))
    echo "  fanning out ${SEARCH_GPUS} workers x ${PER_WORKER} trials each"
    pids=()
    for ((g = 0; g < SEARCH_GPUS; g++)); do
      run_tuner "$g" "$PER_WORKER" "$((SEED * 1000 + SAMPLER_OFFSET + g))" &
      pids+=($!)
      sleep 3 # stagger so worker 0 creates the study before the others attach
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    [ "$fail" -eq 0 ] || {
      echo "ERROR: an Optuna search worker failed (see log above)." >&2
      exit 1
    }
  fi
  echo "=== SEARCH DONE ===  best_params.yaml + study.db in $RUN_DIR"
  echo "Next (refit on train+val, then test):"
  echo "  bash scripts/hpc/submit.sh sr/${EXP_TAG}.sh STAGE=fit SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}"
  exit 0
fi

# ============================== STAGE: bench =================================
# Score the FINAL checkpoint into the shared benchmark store, at 2.5 m against
# the experiment's own GT, through the same joint_sr_dataset helpers training
# used. Default split is `test` — the only split this protocol reports.
if [ "$STAGE" = "bench" ]; then
  CKPT="${RUN_DIR}/checkpoints/${FINAL_CKPT_NAME}.ckpt"
  if [ ! -f "$CKPT" ]; then
    if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
      echo "WARN: ${FINAL_CKPT_NAME}.ckpt missing; benchmarking last.ckpt instead." >&2
      CKPT="${RUN_DIR}/checkpoints/last.ckpt"
    else
      echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — run STAGE=fit first." >&2
      exit 1
    fi
  fi

  STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks}" # SHARED across experiments
  MODEL_NAME="${MODEL_NAME:-sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${RAILS_TAG}${ES_TAG}${PROTO_TAG}${MON_TAG}}"
  LABEL_SOURCE="${LABEL_SOURCE:-${LABELS}}"
  BENCH_SPLIT="${BENCH_SPLIT:-test}"
  TILE_METRICS="${TILE_METRICS:-apls}"
  # Per-chip extras, empty = off. Set them so a seed-N bench carries the SAME
  # columns as the seed-0 rows it will be averaged with — a ragged store makes
  # cross_seed_ci drop whichever metric a seed happens to lack.
  # Buffered P/R/F1 tolerance sweep. ON by default at 1..5 px (2.5-25 m at
  # 2.5 m GSD): all five radii share one distance transform per chip, so the
  # sweep costs barely more than a single radius, and the fit stage's test
  # sensitivity sweep already records exactly these tolerances — a bench row
  # without them cannot be read against its own arm's curve. Set BUFFER_PX=""
  # to switch it off. NB rows benched BEFORE this became the default have no
  # buffered_* columns, so an arm whose seeds straddle the change is ragged:
  # re-bench the older seeds (scripts/local/rebench_all.py) rather than
  # reporting a buffered metric averaged over whichever seeds happen to have it.
  BUFFER_PX="${BUFFER_PX:-1,2,3,4,5}"
  AP_BINS="${AP_BINS:-}"              # e.g. 101

  # val tiles are TRAINING tiles under this protocol — scoring on them would be
  # a train-set number sitting in the same store as honest test numbers.
  if [ "$MERGE_VAL" = "1" ] && [ "$BENCH_SPLIT" = "val" ]; then
    echo "ERROR: BENCH_SPLIT=val, but val was folded into training (TRAIN_SPLITS='${TRAIN_SPLITS}')." >&2
    echo "  That score would be a training score. Use BENCH_SPLIT=test." >&2
    exit 2
  fi

  CONFIG_ARGS=()
  [ -f "${RUN_DIR}/best_params.yaml" ] && CONFIG_ARGS=(--config-yaml "${RUN_DIR}/best_params.yaml")
  MASK_ARGS_BENCH=(--mask-source "$MASK_SOURCE")
  [ "$MASK_SOURCE" = "raster" ] && MASK_ARGS_BENCH+=(--mask-dirname "$MASK_DIRNAME")
  METRIC_ARGS=()
  if [ -n "${TILE_METRICS}" ]; then
    IFS=',' read -r -a _TMS <<<"${TILE_METRICS}"
    for _tm in "${_TMS[@]}"; do METRIC_ARGS+=(--tile-metric "${_tm}"); done
  fi

  # --- θ resolution: BENCH_THRESHOLD env > sweep.json > hard error -----------
  # The runner's silent fallback (checkpoint hparam, 0.5 — the SR configs
  # never set one) is exactly how the store filled with θ=0.5 rows nobody
  # chose. This stage now refuses to score without an explicit operating point.
  if [ -n "${BENCH_THRESHOLD:-}" ]; then
    THETA="$BENCH_THRESHOLD"
    THETA_SRC="BENCH_THRESHOLD (env override)"
  elif [ -f "${RUN_DIR}/sweep.json" ]; then
    THETA=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_threshold'])" "${RUN_DIR}/sweep.json")
    THETA_SRC="${RUN_DIR}/sweep.json"
  else
    echo "ERROR: no operating point for the bench row — refusing the silent θ=0.5 default." >&2
    echo "  STAGE=fit now ends with the post-refit θ* sweep that writes ${RUN_DIR}/sweep.json;" >&2
    echo "  for an older run, produce it with:" >&2
    echo "    python -m benchmarking.cli sweep --dataset-dir ${DATASET_DIR} --checkpoint ${CKPT} \\" >&2
    echo "      --model sr --model-name ${MODEL_NAME} --split val --sen2sr-dir ${SEN2SR_DIR} \\" >&2
    echo "      --mask-source ${MASK_SOURCE}${MASK_DIRNAME:+ --mask-dirname ${MASK_DIRNAME}} --out ${RUN_DIR}/sweep.json" >&2
    echo "  or set BENCH_THRESHOLD explicitly." >&2
    exit 2
  fi
  echo "bench θ = ${THETA}  [${THETA_SRC}]"

  # --- push the bench row onto the FIT's wandb run --------------------------
  # Without this the only test numbers in wandb are the training loop's, which
  # are scored at θ=0.5 — not the operating point this protocol reports. The
  # bench row is scored at θ* (swept post-refit on SEEN train+val data), so
  # pushing it into the same run's summary puts the reported number where the
  # curves are. Summary keys are bench_<split>/* — they never collide with the
  # logged curves. The SR pipeline writes no train_meta.json, so the run id
  # comes off the refit's own wandb dir, exactly as the test stage resolves it.
  WANDB_BENCH_ARGS=()
  if [ "${WANDB_MODE:-online}" != "disabled" ] && [ "${BENCH_WANDB:-1}" = "1" ]; then
    if _LATEST=$(readlink -f "${RUN_DIR}/wandb/latest-run" 2>/dev/null) && [ -n "$_LATEST" ]; then
      WANDB_BENCH_ARGS=(--wandb-run-id "${_LATEST##*-}" \
                        --wandb-project "$WANDB_PROJECT" \
                        --wandb-run-name "$MODEL_NAME")
      echo "bench -> wandb run ${_LATEST##*-} (project ${WANDB_PROJECT})"
    else
      echo "WARN: no ${RUN_DIR}/wandb/latest-run — bench metrics will not reach wandb." >&2
      echo "  (The store row is still written; push it later with benchmarking.cli" >&2
      echo "   eval --wandb-run-id, or read it with \`report\`.)" >&2
    fi
  fi

  echo "=== BENCH (ckpt=$(basename "$CKPT"), model_name=${MODEL_NAME}, seed=${SEED}, split=${BENCH_SPLIT}, θ=${THETA}, gt=${MASK_SOURCE}, tile_metrics=${TILE_METRICS:-none}) ==="
  python -m benchmarking.cli eval \
    --dataset-dir "$DATASET_DIR" \
    --checkpoint "$CKPT" \
    --model sr \
    --model-name "$MODEL_NAME" \
    --seed "$SEED" \
    --store-dir "$STORE_DIR" \
    --split "$BENCH_SPLIT" \
    --sen2sr-dir "$SEN2SR_DIR" \
    --exp-tag "$EXP_TAG" \
    --label-source "$LABEL_SOURCE" \
    --threshold "$THETA" \
    ${METRIC_ARGS[@]+"${METRIC_ARGS[@]}"} \
    ${BUFFER_PX:+--buffer-px "$BUFFER_PX"} \
    ${AP_BINS:+--ap-bins "$AP_BINS"} \
    ${WANDB_BENCH_ARGS[@]+"${WANDB_BENCH_ARGS[@]}"} \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    "${MASK_ARGS_BENCH[@]}"

  echo "=== BENCH DONE ===  store: ${STORE_DIR}"
  echo "Report: python -m benchmarking.cli report --store-dir ${STORE_DIR}"
  exit 0
fi

# ============================== STAGE: fit ===================================
if [ "$STAGE" != "fit" ]; then
  echo "ERROR: STAGE must be tune, fit or bench, got '${STAGE}'." >&2
  exit 2
fi

BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/${FINAL_CKPT_NAME}.ckpt"
if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run STAGE=tune first." >&2
  exit 1
fi
echo "--- best hyperparameters (chosen on val, before the merge) ---"
cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"
RESUME_ARGS=()
if [ "${RESUME_FIT:-0}" = "1" ]; then
  if [ -f "$LAST_CKPT" ]; then
    echo "=== RESUME_FIT=1: continuing the refit from ${LAST_CKPT} ==="
    RESUME_ARGS=(--ckpt_path "$LAST_CKPT")
  else
    echo "WARN: RESUME_FIT=1 but ${LAST_CKPT} not found — refitting from scratch." >&2
  fi
fi

# The SR treatment (and loss arm) is passed explicitly (belt) even though the
# best_params overlay records it too (braces) — drift is impossible.
MODEL_ARGS=(--model.upsampler "$UPSAMPLER" --model.freeze_sr "$FREEZE_SR"
  --model.sr_pad "$SR_PAD" --model.sen2sr_dir "$SEN2SR_DIR"
  --model.lr_schedule "$LR_SCHEDULE"
  --model.sr_warmup_epochs "$SR_WARMUP_EPOCHS"
  --model.l2sp_lambda "$L2SP_LAMBDA"
  --model.adaptive_norm "$ADAPTIVE_NORM_FLAG"
  --model.adaptive_norm_momentum "$ADAPTIVE_NORM_M"
  --model.norm_recalibrate "$NORM_RECALIBRATE"
  --model.sr_snapshot_every "$SR_SNAPSHOT_EVERY")
# Appended only when the constraint is forced (see the SR_HC block): the native
# arms' fit/test command lines stay byte-identical.
if [ "$SR_HC" != "native" ]; then
  MODEL_ARGS+=("${HC_ARGS_FIT[@]}")
fi
# Same discipline for the std-band rails: nothing is appended unless
# STD_BAND_RAISE_LO/HI (or STD_BAND_ACTION) were set. The fit's action is `warn`
# from joint_sr.yaml either way, so the rails do not change whether the run
# survives a band exit — they change where the exit is DECLARED, i.e. what the
# adapt_band_exit metric and the post-hoc envelope crossing are measured
# against. (An explicit STD_BAND_ACTION=raise DOES change it, which is why the
# block above shouts about that combination.)
if [ ${#RAILS_ARGS_FIT[@]} -gt 0 ]; then
  MODEL_ARGS+=("${RAILS_ARGS_FIT[@]}")
fi
# Same again for the hard hold: appended only when SR_HOLD_EPOCHS > 0.
if [ ${#HOLD_ARGS_FIT[@]} -gt 0 ]; then
  MODEL_ARGS+=("${HOLD_ARGS_FIT[@]}")
fi
if [ -n "$WARM_START_CKPT" ]; then
  MODEL_ARGS+=(--model.warm_start_unet "$WARM_START_CKPT")
fi
# Head args are appended ONLY for the linear probe, so the U-Net arms' command
# line is byte-identical to what it was before HEAD existed.
if [ -n "$HEAD_TAG" ]; then
  MODEL_ARGS+=(--model.head "$HEAD" --model.clip_sr "$CLIP_SR")
  if [ -n "$WARM_START_HEAD" ]; then
    MODEL_ARGS+=(--model.warm_start_head "$WARM_START_HEAD")
  fi
fi
if [ -n "$LOSS_ARM" ]; then
  MODEL_ARGS+=("${LOSS_ARGS_FIT[@]}")
fi

# joint_sr_trainval.yaml is layered LAST so its callback list and
# limit_val_batches win over joint_sr.yaml's val-monitored ones. train_splits is
# ALSO passed explicitly, so a holdout run (TRAIN_SPLITS=train) overrides the
# overlay's default rather than needing a second config file.
# shellcheck disable=SC2206
TRAIN_SPLITS_ARR=(${TRAIN_SPLITS})
SPLIT_ARGS=(--data.train_splits "[$(
  IFS=,
  echo "${TRAIN_SPLITS_ARR[*]}"
)]")

if [ "$FIT_EARLY_STOP" = "1" ]; then
  echo "=== REFIT on '${TRAIN_SPLITS}' (best config, <=${REFIT_EPOCHS} epochs, EARLY STOPPING on ${ES_MONITOR} (${ES_MODE}, patience ${ES_PATIENCE}), ${REFIT_GPUS} GPU) ==="
else
  echo "=== REFIT on '${TRAIN_SPLITS}' (best config, FIXED ${REFIT_EPOCHS} epochs, no early stopping, ${REFIT_GPUS} GPU) ==="
fi
python -m sr.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --config "$TRAINVAL_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  ${MASK_DIRNAME:+--data.mask_dirname "$MASK_DIRNAME"} \
  "${SPLIT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  ${VAL_ARGS_FIT[@]+"${VAL_ARGS_FIT[@]}"} \
  ${ES_ARGS_FIT[@]+"${ES_ARGS_FIT[@]}"} \
  --trainer.gradient_clip_val "$CLIP_TRAINER" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

# Log the test metrics to the SAME wandb run the refit just created.
if LATEST_RUN=$(readlink -f "$RUN_DIR/wandb/latest-run" 2>/dev/null) && [ -n "$LATEST_RUN" ]; then
  export WANDB_RUN_ID="${LATEST_RUN##*-}" # .../run-<timestamp>-<id> -> <id>
  export WANDB_RESUME=must
  echo "resuming wandb run ${WANDB_RUN_ID} for the test split"
else
  echo "WARN: could not locate the refit's wandb run; test will log to a fresh run" >&2
fi

if [ ! -f "$CKPT" ]; then
  if [ -f "$LAST_CKPT" ]; then
    echo "WARN: ${FINAL_CKPT_NAME}.ckpt missing; testing last.ckpt instead." >&2
    CKPT="$LAST_CKPT"
  else
    echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — refit produced none. Skipping test." >&2
    exit 1
  fi
fi

# Under the refit protocol this is the ONLY held-out evaluation. Under the
# pilot (TRAIN_SPLITS=train) it is a free preview — decisions still read the
# val bench, and the pilot never compares these test numbers between arms.
# SKIP_TEST=1 keeps test genuinely unseen; the Lightning twin has had this
# guard since the pilot was ported, this engine had not (added 2026-08-16).
# SKIP_TEST gates only the steps that READ test. The val theta* sweep below is
# NOT gated: the bench stage refuses to run without sweep.json, so skipping it
# turns a pilot fit into a run that can never be benched. (An earlier version of
# this guard exited here and did exactly that.)
if [ "${SKIP_TEST:-0}" = "1" ]; then
  echo "=== SKIP_TEST=1: not running the test split (pilot mode) ==="
else

# The ONLY held-out evaluation in this protocol.
echo "=== TEST (held-out split, ckpt=$(basename "$CKPT")) ==="
python -m sr.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  ${MASK_DIRNAME:+--data.mask_dirname "$MASK_DIRNAME"} \
  "${MODEL_ARGS[@]}" \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

fi   # end SKIP_TEST gate around the held-out test

# --- Post-refit θ* sweep (selection) -----------------------------------------
# θ* is selected AFTER the refit, on the val split — refit TRAINING data under
# this protocol (TRAIN_SPLITS='train val'), deliberately: a θ chosen on seen
# data cannot inflate test numbers, only cost a mildly suboptimal operating
# point (docs/ap_threshold_protocol_plan.md §1.2). The bench stage refuses to
# run without a θ (BENCH_THRESHOLD or this sweep.json).
SWEEP_SPLIT="${SWEEP_SPLIT:-val}"
MODEL_NAME="${MODEL_NAME:-sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${RAILS_TAG}${ES_TAG}${PROTO_TAG}${MON_TAG}}"
MASK_ARGS_SWEEP=(--mask-source "$MASK_SOURCE")
[ "$MASK_SOURCE" = "raster" ] && MASK_ARGS_SWEEP+=(--mask-dirname "$MASK_DIRNAME")

# What θ* is the argmax OF. iou|f1 are global pooled counts (a few dense urban
# chips dominate); iou_macro|f1_macro are the mean of the per-chip values (every
# chip weighs the same). All four land in sweep.json whichever is selected on,
# so switching later costs no inference — but a seed swept on one and its
# siblings on another are NOT at a comparable operating point.
SWEEP_CRITERION="${SWEEP_CRITERION:-iou}"

echo "=== θ* SWEEP (split=${SWEEP_SPLIT}, criterion=${SWEEP_CRITERION} — seen data, selection-only) ==="
python -m benchmarking.cli sweep \
  --criterion "$SWEEP_CRITERION" \
  --dataset-dir "$DATASET_DIR" \
  --checkpoint "$CKPT" \
  --model sr \
  --model-name "$MODEL_NAME" \
  --seed "$SEED" \
  --split "$SWEEP_SPLIT" \
  --sen2sr-dir "$SEN2SR_DIR" \
  --out "${RUN_DIR}/sweep.json" \
  "${MASK_ARGS_SWEEP[@]}"
THETA=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_threshold'])" "${RUN_DIR}/sweep.json")
echo "θ* = ${THETA}  -> ${RUN_DIR}/sweep.json"

# --- Test θ-sensitivity sweep (reporting ONLY, never selection) --------------
# The full IoU/F1(θ) curve on test plus the buffered-F1 tolerance sweep for the
# write-up: θ-flatness around θ*, the IoU@0.5 companion number, tolerances
# 1-5 px. purpose="sensitivity" is stamped in the JSON so it cannot later be
# mistaken for a selection artifact.
if [ "${SKIP_TEST:-0}" = "1" ]; then
  echo "=== SKIP_TEST=1: skipping the test sensitivity sweep too ==="
  echo "=== FIT DONE ===  sweep.json written; bench with STAGE=bench ==="
  exit 0
fi

echo "=== TEST θ SENSITIVITY SWEEP (criterion=${SWEEP_CRITERION}, buffer_px=1,2,3,4,5) ==="
python -m benchmarking.cli sweep \
  --criterion "$SWEEP_CRITERION" \
  --dataset-dir "$DATASET_DIR" \
  --checkpoint "$CKPT" \
  --model sr \
  --model-name "$MODEL_NAME" \
  --seed "$SEED" \
  --split test \
  --sen2sr-dir "$SEN2SR_DIR" \
  --buffer-px 1,2,3,4,5 \
  --out "${RUN_DIR}/test_sweep.json" \
  "${MASK_ARGS_SWEEP[@]}"

echo "=== DONE ===  outputs in $RUN_DIR"

# --- fit -> bench in ONE allocation (CHAIN_BENCH=1) --------------------------
# >>> chain-bench (extracted verbatim by tests/test_rl_campaign_hpc.py)
if [ "${CHAIN_BENCH:-1}" = "1" ] && [ "${SKIP_TEST:-0}" != "1" ]; then
  echo "=== CHAIN_BENCH=1: continuing into STAGE=bench in this allocation ==="
  # exec, not a call: the fit is finished, and re-entering the ARM (not this
  # engine) means the bench stage re-derives RUN_DIR, MODEL_NAME, the treatment
  # and theta* from the arm's own settings — identical to a standalone
  # `STAGE=bench` submission, with nothing carried over from the fit.
  exec env STAGE=bench bash "$ARM_SCRIPT"
fi
# <<< chain-bench
if [ "${CHAIN_BENCH:-1}" = "1" ]; then
  echo "NOTE: SKIP_TEST=1, so the fit did NOT chain into the bench. The bench"
  echo "  reads the test split, which is the split SKIP_TEST exists to keep unseen."
fi
echo "Bench: sbatch scripts/hpc/train.sbatch --SCRIPT=sr/${EXP_TAG}.sh STAGE=bench SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}"