#!/bin/bash
# =============================================================================
# ONE LANE of the rl campaign, start to finish (docs/rl_lightning_campaign_plan.md §3).
#
#   bash scripts/LightningStudio/sr/rl/lane.sh LANE=base
#   bash scripts/LightningStudio/sr/rl/lane.sh LANE=sen2sr
#   bash scripts/LightningStudio/sr/rl/lane.sh LANE=sr4rs
#
# Detach it (a lane is many hours; the Studio machine must stay on):
#   bash scripts/LightningStudio/job.sh run sr/rl/lane.sh LANE=sr4rs
#
# TOPOLOGY. Concurrency 2 on the current tier, so exactly two lanes run at once:
# `sen2sr` and `sr4rs`. Rungs are SERIAL within a lane — an embarrassingly
# parallel workload does not justify a multi-GPU box's interconnect premium, and
# T4/Kaggle is out (no bf16). Run `base` first: rl0 is the cheapest arm in the
# series and it is the anchor every other arm is reported against, so a failure
# there should not cost an SR4RS lane's walltime to discover.
#
# ORDER WITHIN A LANE. Frozen arm first (it is the joint arms' control and their
# hold phase should reproduce its curve — the §6.1 harness self-check), then the
# rungs EXTREMES FIRST: 1e-3 and 1e-6 carry the effect and the probable
# collapses, and they are also the two that get the second seed, so a lane cut
# short by a deadline still has both ends of the dose-response.
#
# STAGES PER RUN. tune (1 trial x 1 epoch: writes best_params.yaml AND is the
# §4 gate-1 timing/VRAM measurement) -> fit (the 30-epoch run + θ* sweep on val)
# -> bench (the test row into the shared store).
#
# GATES (plan §4), which this script enforces only as far as a shell can:
#   1. STOPS at the end of the FIRST tune of a lane unless GATE1_OK=1, so the
#      timing and peak-VRAM numbers are actually read by a human before a
#      multi-hour wave is launched.
#   2. ≥20% credit headroom before every launch wave — check it yourself; this
#      script cannot see your balance.
#   3. If measured cost blows the estimate by >2x, stop and re-plan. The cluster
#      is the free fallback for the SR4RS lane (a per-lane platform split is
#      admissible as long as it is stated, and no contrast crosses it).
# =============================================================================
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LS_ROOT="$(cd "$RL_DIR/../.." && pwd)"

for kv in "$@"; do
  case "$kv" in
  *=*) export "$kv" ;;
  *) echo "ERROR: expected KEY=VALUE, got '$kv'" >&2; exit 2 ;;
  esac
done

LANE="${LANE:?set LANE=base|sen2sr|sr4rs}"
# HEAD_LR is pinned at 3e-3 in _rl_common.sh; export it here only to override.

# Extremes first; the 1e-3 and 1e-6 rungs also carry the +1 seed.
RUNGS="${RUNGS:-1e-3 1e-6 1e-4 1e-5}"
EXTREME_RUNGS="${EXTREME_RUNGS:-1e-3 1e-6}"
SEEDS="${SEEDS:-0}"           # base seed for every run
EXTRA_SEED="${EXTRA_SEED:-1}" # second seed on the extremes and on rl1

case "$LANE" in
base)   FROZEN="";        JOINT="";        FROZEN_ONLY="rl0" ;;
sen2sr) FROZEN="rl1";     JOINT="rl2";     FROZEN_ONLY="" ;;
sr4rs)  FROZEN="rl3";     JOINT="rl4";     FROZEN_ONLY="" ;;
*)
  echo "ERROR: LANE must be base|sen2sr|sr4rs, got '${LANE}'." >&2
  exit 2
  ;;
esac

_first_tune_done=0

run_one() {  # arm seed [lrsr]
  local arm="$1" seed="$2" lrsr="${3:-}"
  local label="${arm}${lrsr:+ lr_sr=${lrsr}} seed=${seed}"
  for stage in tune fit bench; do
    echo ""
    echo "############################################################"
    echo "### rl lane ${LANE}: ${label}  STAGE=${stage}"
    echo "############################################################"
    ( export SEED="$seed"; [ -n "$lrsr" ] && export LRSR="$lrsr"
      bash "$LS_ROOT/run.sh" "sr/rl/${arm}.sh" "STAGE=${stage}" )
    if [ "$stage" = "tune" ] && [ "$_first_tune_done" = "0" ]; then
      _first_tune_done=1
      if [ "${GATE1_OK:-0}" != "1" ]; then
        echo ""
        echo "=== GATE 1 (plan §4): first timing run of lane '${LANE}' is done. ==="
        echo "  Read h/epoch off the log above and peak VRAM off nvidia-smi, then"
        echo "  re-launch with GATE1_OK=1 to run the lane. Estimates to check"
        echo "  against: frozen SR4RS ~0.15-0.25 h/ep, joint SR4RS ~0.2-0.3 h/ep,"
        echo "  peak ~8-12 GB at bs=4. On OOM: activation checkpointing on the"
        echo "  SR4RS blocks, then grad-accum 2x2 — NEVER a batch-size change."
        echo "  If measured cost is >2x the estimate, stop and re-plan (gate 3)."
        exit 0
      fi
    fi
  done
}

echo "=== rl lane '${LANE}'${HEAD_LR:+ — head_lr override ${HEAD_LR}} ==="

for arm in $FROZEN_ONLY $FROZEN; do
  for s in $SEEDS; do run_one "$arm" "$s"; done
  # rl1 is the SEN2SR row's reference point and gets the +1 seed (plan §2).
  if [ "$arm" = "rl1" ] && [ -n "$EXTRA_SEED" ]; then
    run_one "$arm" "$EXTRA_SEED"
  fi
done

for rung in $RUNGS; do
  [ -z "$JOINT" ] && break
  for s in $SEEDS; do run_one "$JOINT" "$s" "$rung"; done
  case " $EXTREME_RUNGS " in
  *" $rung "*)
    [ -n "$EXTRA_SEED" ] && run_one "$JOINT" "$EXTRA_SEED" "$rung"
    ;;
  esac
done

echo ""
echo "=== rl lane '${LANE}' COMPLETE ==="
echo "Report: python -m benchmarking.cli report --store-dir ${INSTAROAD_ROOT:-<root>}/benchmarks"
