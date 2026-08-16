#!/bin/bash
# Shared configuration for the RL-series (linear-probe read-out on the SR
# front-ends) — docs/sr_linear_probe.md. Sourced by every rl*_new.sh BEFORE
# _stages_tv.sh. Not submitted directly.
#
# WHY THIS FILE EXISTS. The plan (§4) specifies five self-contained ~20-line arm
# scripts. That works for the r-series, whose arms differ only in three lines of
# SR treatment. The rl-series additionally has to hold a ~15-line pinned loss
# block IDENTICAL across all five arms — and "identical" is not a stylistic
# preference here, it is the frozen-control rule the whole series rests on. Five
# hand-maintained copies of fifteen floats is a drift hazard with no upside, and
# a drift that broke the control would be silent. One sourced file makes the
# constancy structural instead of clerical.
#
# Everything below is overridable at submit time, so any single arm can still be
# perturbed for a diagnostic without editing this file — but note that doing so
# to only ONE arm voids the between-arm contrasts.

# --- The read-out --------------------------------------------------------
HEAD="linear"

# --- Model selection -----------------------------------------------------
# val_ap, not val_iou@0.5. A 5-parameter logistic regression has no reason to be
# calibrated at 0.5, and its logits are a smooth unsaturated projection with
# substantial mass near the boundary — so selecting on IoU@0.5 would reward
# whichever arm's SR output happens to place the decision boundary near 0.5,
# i.e. the nuisance variable would move with the treatment (§5.1). This matches
# _stages_tv.sh's own default; it is restated here because the series is void
# without it.
MONITOR="${MONITOR:-val_ap}"

# --- Search space (§6) ---------------------------------------------------
# A 5-parameter model on z-scored inputs converges at a far higher LR than a
# 24 M-param U-Net; the r-series band (1e-5..1e-2) may not contain the optimum.
# INFERRED, not measured — Gate B (§10.4) exists to check the best trial does
# not pin to an edge. If it lands on 3e-1, widen further and re-tune.
LR_MIN="${LR_MIN:-1e-3}"
LR_MAX="${LR_MAX:-3e-1}"
# lr_sr is unchanged from the r-series: the generator is the same net under the
# same task loss. Searched only on the joint arms (rl2/rl4); auto-skipped when
# FREEZE_SR=true.
LR_SR_MIN="${LR_SR_MIN:-1e-7}"
LR_SR_MAX="${LR_SR_MAX:-1e-4}"

# 30 trials for EVERY arm, frozen and joint alike — not scaled to the number of
# searched dimensions (§6.4). Giving the 2-D joint arms more trials would assign
# search quality by treatment, which is the same defect as searching batch size,
# one level up. The frozen arms are over-served; they are also the cheap ones.
N_TRIALS="${N_TRIALS:-30}"

# PINNED, never searched (§6.1). `length` is fixed per epoch, so a bs=2 trial
# takes twice the optimiser steps of a bs=4 trial inside the same epoch budget
# and wins the tune on step count alone. 4 is the between-series constant and
# the largest that fits the heaviest arms (rl3/rl4: SR4RS runs 256-ch convs,
# incl. a 9x9, at 512 px; 8 OOMs on 44 GB).
BATCH_SIZES="${BATCH_SIZES:-4}"

# Between-arm constant of the protocol. Do not change for one arm.
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"

# Kept at the r-series value so the joint arms' recipe stays comparable; costs
# nothing on the frozen arms, where model.py auto-disables it.
SR_WARMUP_EPOCHS="${SR_WARMUP_EPOCHS:-1.0}"

# CLIP is deliberately NOT pinned here. _stages_tv.sh already defaults it to 1.0
# and, under HEAD=linear, routes it to --model.clip_sr while switching the
# Trainer-level global clip off — so rl1's head and rl2's head are treated
# identically (§6.2) at the same 1.0 the r-series joint arms applied to their SR
# group. Setting it in this file would ALSO defeat the REG=false branch of
# _stages_tv.sh, which zeroes CLIP only if nothing has claimed it first.

# =============================================================================
# --- FROZEN LOSS CONTROL -----------------------------------------------------
# =============================================================================
# Read off r0_new's best_params.yaml, i.e. what the r-series anchor ACTUALLY ran
# — resolving open items §9.1 and §9.2. rl0 is r0_new's twin, so this is the
# closest available match to the frozen control.
#
# READ THIS BEFORE TRUSTING ANY rl-vs-r COMPARISON.
# Four things about that overlay are worth stating plainly rather than
# discovering later:
#
#   1. tl_theta, gap_theta, mix_w and pos_weight below were SEARCHED in the
#      r0_new tune, not pinned — they are that arm's Optuna optima, not
#      protocol constants. Pinning them here makes the rl-series STRICTER than
#      its twin series, which is the right direction (a constant cannot confound
#      a contrast) but is not the same recipe.
#   2. pos_weight=4.6165 is NOT the engine's pinned default (3.3523), so the
#      r0_new submit widened that band. Whatever λ* the loss pilot settled on,
#      this is not it.
#   3. r0_new ran at batch_size=8 and selected on val_iou. The rl arms run at
#      batch_size=4 and select on val_ap. Both are deliberate (§6.1, §5.1) and
#      both make rl-vs-r numbers qualitative — on top of the five-orders-of-
#      magnitude capacity gap, which already did.
#   4. If the r-series is re-tuned under BATCH_SIZES=4 / MONITOR=val_ap, these
#      numbers change and this block must be re-derived from the new overlay.
#
# The values are pinned as CONSTANTS (min == max, SEARCH_* = false) so no rl arm
# can land on a different loss configuration from its neighbours.
LOSS_ARM="${LOSS_ARM:-pstar_dice}"
PSTAR="${PSTAR:-gap_t4_ce}"

GAP_R="${GAP_R:-4}"
GAP_K="${GAP_K:-60.0}"
TL_ELL="${TL_ELL:-5}"
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
CL_ALPHA="${CL_ALPHA:-0.3}"
CL_ITERS="${CL_ITERS:-5}"
SKEL_W="${SKEL_W:-1.0}"          # -> model.sr_w (SkeletonRecall weight, NOT SR)
SKEL_RADIUS="${SKEL_RADIUS:-1}"  # -> model.sr_radius

# Loss-schedule offsets as r0_new ran them (engine defaults are 30/10).
WARMUP_START="${WARMUP_START:-15}"
WARMUP_RAMP="${WARMUP_RAMP:-5}"

# Searched in r0_new; CONSTANT here.
SEARCH_THETAS="${SEARCH_THETAS:-false}"
TL_THETA="${TL_THETA:-0.40582224484185075}"
GAP_THETA="${GAP_THETA:-0.6096934757736867}"

SEARCH_MIX_W="${SEARCH_MIX_W:-false}"
MIX_W="${MIX_W:-0.6075946831862098}"

# min == max is how this engine expresses "a constant, not a band".
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-4.616504933210799}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-4.616504933210799}"

# --- Sanity: the loss must be a constant, so refuse a half-pinned submit ------
if [ "$SEARCH_THETAS" != "false" ] || [ "$SEARCH_MIX_W" != "false" ] \
   || [ "$POS_WEIGHT_MIN" != "$POS_WEIGHT_MAX" ]; then
  echo "WARN: the rl-series loss control is NOT frozen for this submit:" >&2
  echo "  SEARCH_THETAS=${SEARCH_THETAS} SEARCH_MIX_W=${SEARCH_MIX_W}" >&2
  echo "  POS_WEIGHT=[${POS_WEIGHT_MIN}, ${POS_WEIGHT_MAX}]" >&2
  echo "  Every rl contrast (rl1-rl0, rl2-rl1, rl3-rl0, rl4-rl3) assumes the loss" >&2
  echo "  is identical across arms. If this is deliberate, it must be done to ALL" >&2
  echo "  FIVE arms or the series is void. (RL_LOOSE_LOSS_OK=1 to silence.)" >&2
  if [ "${RL_LOOSE_LOSS_OK:-0}" != "1" ]; then exit 2; fi
fi

echo "[rl] head=linear  monitor=${MONITOR}  n_trials=${N_TRIALS}  bs=${BATCH_SIZES}"
echo "[rl] lr band [${LR_MIN}, ${LR_MAX}]  lr_sr band [${LR_SR_MIN}, ${LR_SR_MAX}]"
echo "[rl] loss FROZEN: ${LOSS_ARM}(pstar=${PSTAR}) tl_theta=${TL_THETA}"
echo "[rl]              gap_theta=${GAP_THETA} mix_w=${MIX_W} pos_weight=${POS_WEIGHT_MIN}"
