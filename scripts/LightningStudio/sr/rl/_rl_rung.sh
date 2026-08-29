#!/bin/bash
# =============================================================================
# THE lr_sr LADDER — one rung of a joint rl arm (docs/rl_lightning_campaign_plan.md §2).
# Sourced by rl2.sh and rl4.sh only, AFTER _rl_common.sh and BEFORE the engine.
# Not run directly.
#
# Both rows (SEN2SR and SR4RS) climb the SAME ladder:
#
#     LRSR ∈ {1e-3, 1e-4, 1e-5, 1e-6}
#
# The rungs are chosen by DOSE, not by nominal value. Adaptation goes as
# lr_sr × joint epochs, and 20 joint epochs against the R-series' ~100 shifts
# the informative band up ~5x:
#
#   1e-6  ≈ the R-series bare arms' landed dose (4.5e-7 × 100 ≈ 1e-6 × 20)
#   1e-5  \_ bracket the clean-adaptation regime
#   1e-4  /  (1e-4 × 20 ≈ r2a's 1.4e-5 × 100)
#   1e-3  deliberately probes the CEILING — the unconstrained-degeneracy
#         question lives there, and numerical death there is data, not a bug.
#
# THE ZERO-DOSE POINT IS FREE: every joint run's epoch-10 state is dose 0, which
# is why no rung is spent near-frozen (1e-7 was dropped for exactly that reason).
#
# PRE-REGISTERED CONTINGENCY: if the 1e-3 rung collapses within the first TWO
# joint epochs on BOTH rows, replace it with 3e-4. One dead rung is data; two is
# a wasted axis.
#
# RUN ORDER: extremes first, always — they carry the effect and the probable
# collapses, and the middle rungs interpolate.
#
# Overlay with the r2grid is BY DOSE (lr_sr × epochs) in captions, never by
# nominal value; nominal-value overlay died with the loss and budget change.
#
# SEEDS: 1 per rung, +1 on the extreme rungs (1e-3, 1e-6). All claims are
# descriptive: within-run trajectories and dose-response ORDERING are the
# primary evidence, per the grid-plan precedent — between-rung metric gaps are
# not interpreted against seed noise at n=1.
# =============================================================================

LRSR="${LRSR:?set LRSR=<lr_sr> — the pinned rung, e.g. LRSR=1e-3 (see the ladder above)}"

# lr_sr -> tag, normalised to 1-2 significant figures in scientific notation, so
# 1e-4, 1E-04 and 0.0001 all name the SAME rung and a typo'd 1.4e-5 names a
# DIFFERENT one rather than silently reusing a neighbour's run dir and study.
LS_TAG=$(awk -v v="$LRSR" 'BEGIN {
  if (v + 0 != v || v <= 0) exit 1
  split(sprintf("%.1e", v), a, "e")
  m = a[1]; sub(/\.0$/, "", m)
  printf "%se%d", m, a[2] + 0
}' </dev/null) || {
  echo "ERROR: LRSR must be a positive number, got '${LRSR}'." >&2
  exit 2
}

# min == max: Optuna's log-uniform then suggests the CONSTANT, which lands in
# best_params.yaml exactly like a searched value — the only reason STAGE=fit
# needs no special case. Pinned by tests/test_sr_hold_ramp.py.
LR_SR_MIN="$LRSR"
LR_SR_MAX="$LRSR"

# The hard hold. 10 of 30 epochs, identical on both rows and every rung: the
# hold phase is what makes each joint run carry its own frozen control and its
# own dose-0 point. Overriding it for ONE rung would make that rung's "gain"
# incomparable with its neighbours'.
SR_HOLD_EPOCHS="${SR_HOLD_EPOCHS:-10}"

# Appended to the arm's EXP_TAG by the caller, so run dirs, Optuna studies and
# benchmark rows are disjoint per rung by construction.
RUNG_TAG="_ls${LS_TAG}"

echo "[rl] RUNG lr_sr=${LRSR} (pinned, tag ${RUNG_TAG})  hold=${SR_HOLD_EPOCHS} of ${REFIT_EPOCHS} ep"
echo "[rl]   dose = lr_sr x joint epochs = ${LRSR} x $((REFIT_EPOCHS - SR_HOLD_EPOCHS))"
