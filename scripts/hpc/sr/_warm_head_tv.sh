#!/bin/bash
# Head warm-start plumbing for the RL-series joint arms (rl2/rl4) — sourced
# AFTER _rl_common.sh and BEFORE _stages_tv.sh. Not submitted directly.
#
# This is the linear-probe sibling of _warm_tv.sh, and the differences from it
# are the whole point:
#
#   1. It sets WARM_START_HEAD, never WARM_START_CKPT. model.py zeroes
#      sr_warmup_epochs when warm_start_unet is set — correct for a staged U-Net
#      start (the warm start IS the warmup), wrong here, where the ramp must
#      survive so the joint arm's recipe matches its r-series twin
#      (docs/sr_linear_probe.md §2). Routing the head through warm_start_unet
#      would silently drop the ramp and nothing in the run would say so.
#
#   2. It pins NOTHING from stage 1's best_params.yaml. _warm_tv.sh anchors the
#      stage-2 lr to a [best/100, best] fine-tuning band because stage 2 there
#      continues a CONVERGED 24 M-param critic that a from-scratch lr could walk
#      out of its minimum. That reasoning does not carry: this head has five
#      parameters and is re-converged in a handful of steps, so an lr band
#      inherited from the frozen arm would be an arbitrary constraint on the
#      joint arm's search. rl2/rl4 search (lr, lr_sr) over the full rl band.
#      The loss/batch/encoder knobs _warm_tv.sh pins from stage 1 are already
#      constants across the whole series via _rl_common.sh.
#
# So: this file resolves one path and validates it. That is all it should do.
#
# In:  STAGE1_TAG   the frozen twin (rl1_new for rl2_new, rl3_new for rl4_new,
#                   rl1b_new for rl2b_new, rl3a_new for rl4a_new)
#      SEED / LOSS_ARM / REG / TRAIN_SPLITS / HEAD / SR_HC — must match the
#      twin's run; the run-dir naming ties them together automatically.
# Out: WARM_START_HEAD   the twin's FINAL ckpt (env override respected)
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"

: "${STAGE1_TAG:?joint rl arm must set STAGE1_TAG (e.g. rl1_new)}"
SEED="${SEED:-0}"

# Reconstruct _stages_tv.sh's RUN_DIR tags. All of these must be re-derived here
# because _stages_tv.sh has not been sourced yet — and every one of them is a
# place the two files can silently disagree, so they are kept in the same order
# as the RUN_DIR assignment there:
#   sr_${EXP_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${PROTO_TAG}_seed${SEED}
#
# HC_TAG is derived from the arm's own SR_HC, which is exactly right: the HC
# lane of the rl 2x2 (rl2b/rl4a) must warm-start from a stage 1 that ran under
# the SAME constraint setting, or the probe would arrive converged on a
# different input distribution and LP-FT's whole argument (§2) evaporates. The
# native arms (rl2/rl4) get an empty tag, so their resolved path is unchanged.
SR_HC="${SR_HC:-native}"
case "$SR_HC" in
native) HC_TAG="" ;;
on) HC_TAG="_hc" ;;
off) HC_TAG="_nohc" ;;
*)
  echo "ERROR: SR_HC must be native|on|off, got '${SR_HC}'." >&2
  exit 2
  ;;
esac

HEAD="${HEAD:-unet}"
HEAD_TAG=""
[ "$HEAD" != "unet" ] && HEAD_TAG="_${HEAD}"

LOSS_ARM="${LOSS_ARM:-}"
LOSS_TAG=""
[ -n "$LOSS_ARM" ] && LOSS_TAG="_$(echo "$LOSS_ARM" | tr '+' '-')"

REG="${REG:-true}"
REG_TAG=""
{ [ "$REG" = "false" ] || [ "$REG" = "0" ]; } && REG_TAG="_noreg"

ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"
ANORM_TAG=""
{ [ "$ADAPTIVE_NORM" = "1" ] || [ "$ADAPTIVE_NORM" = "true" ]; } && ANORM_TAG="_anorm"
[ "$NORM_RECALIBRATE" != "off" ] && ANORM_TAG="${ANORM_TAG}_recal${NORM_RECALIBRATE}"

TRAIN_SPLITS="${TRAIN_SPLITS:-train val}"
PROTO_TAG=""
case " ${TRAIN_SPLITS} " in
  *" val "*) : ;;
  *) PROTO_TAG="_holdout" ;;
esac

RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"
STAGE1_RUN="${STAGE1_RUN:-${RUNS_ROOT}/sr_${STAGE1_TAG}${HC_TAG}${HEAD_TAG}${LOSS_TAG}${REG_TAG}${ANORM_TAG}${PROTO_TAG}_seed${SEED}}"

# _final, not _best: this protocol never selects a checkpoint on a holdout, and
# the FINAL head is what §2 calls for — a 5-parameter near-convex problem is
# converged long before the end of the budget, so any mid-training snapshot
# would introduce an arbitrary constant needing its own justification.
STAGE1_CKPT_NAME="${STAGE1_CKPT_NAME:-unet_s2rosa_jointsr_final}"
WARM_START_HEAD="${WARM_START_HEAD:-${STAGE1_RUN}/checkpoints/${STAGE1_CKPT_NAME}.ckpt}"

if [ ! -f "$WARM_START_HEAD" ]; then
  echo "ERROR: frozen-twin ckpt not found: ${WARM_START_HEAD}" >&2
  echo "  ${STAGE1_TAG}'s fit must COMPLETE before this arm's tune can start" >&2
  echo "  (docs/sr_linear_probe.md §10). Run first:" >&2
  echo "    bash scripts/hpc/submit.sh sr/${STAGE1_TAG}.sh STAGE=tune SEED=${SEED}" >&2
  echo "    bash scripts/hpc/submit.sh sr/${STAGE1_TAG}.sh STAGE=fit  SEED=${SEED}" >&2
  echo "  Looked in: ${STAGE1_RUN}" >&2
  if [ ! -d "$STAGE1_RUN" ]; then
    echo "  (that run dir does not exist at all — check SEED, LOSS_ARM, REG," >&2
    echo "   SR_HC, ADAPTIVE_NORM/NORM_RECALIBRATE and TRAIN_SPLITS match the" >&2
    echo "   twin's submit)" >&2
  fi
  exit 1
fi

# Guard the exact confusion this file exists to prevent.
if [ "$HEAD" != "linear" ]; then
  echo "ERROR: _warm_head_tv.sh sourced with HEAD=${HEAD}. This resolves a LINEAR" >&2
  echo "  PROBE warm start; the staged U-Net path is _warm_tv.sh." >&2
  exit 2
fi
if [ -n "${WARM_START_CKPT:-}" ]; then
  echo "ERROR: WARM_START_CKPT is set alongside the head warm start. Mutually" >&2
  echo "  exclusive — warm_start_unet auto-disables the SR warmup ramp (§2)." >&2
  exit 2
fi

echo "[warm-head] ${STAGE1_TAG} -> linear probe init: ${WARM_START_HEAD}"
echo "[warm-head] LP-FT: the probe is converged on the FROZEN-SR input distribution"
echo "[warm-head] before any gradient reaches the generator, so SR trainability is"
echo "[warm-head] the only difference between this arm and ${STAGE1_TAG}."
echo "[warm-head] nothing pinned from stage 1: (lr, lr_sr) searched over the rl band."
