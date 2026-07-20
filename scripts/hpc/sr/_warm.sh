#!/bin/bash
# Shared stage-2 (warm-start) plumbing for r6/r7 — sourced BEFORE _stages.sh
# by the staged arm scripts. Not submitted directly.
#
# In:  STAGE1_TAG  (e.g. r5_all) — the frozen-SR arm this arm warm-starts from.
#      SEED / LOSS_ARM / loss hps — must match the stage-1 run (the run-dir
#      naming ties them together automatically).
# Out: WARM_START_CKPT           — stage-1 best ckpt (env override respected)
#      LR_MIN/LR_MAX, POS_WEIGHT_MIN/MAX, BATCH_SIZES, ENCODERS — pinned to
#      stage-1's best_params.yaml so the stage-2 search covers ONLY lr_sr
#      (the loss stays fixed across stages: it defines the task-critic).
#      PIN_FROM_STAGE1=0 disables the pinning (full joint re-search).
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"

: "${STAGE1_TAG:?staged arm script must set STAGE1_TAG (e.g. r5_all)}"
SEED="${SEED:-0}"
LOSS_ARM="${LOSS_ARM:-}"
LOSS_TAG=""
[ -n "$LOSS_ARM" ] && LOSS_TAG="_$(echo "$LOSS_ARM" | tr '+' '-')"

RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"
STAGE1_RUN="${STAGE1_RUN:-${RUNS_ROOT}/sr_${STAGE1_TAG}${LOSS_TAG}_seed${SEED}}"
STAGE1_BEST="${STAGE1_RUN}/best_params.yaml"
WARM_START_CKPT="${WARM_START_CKPT:-${STAGE1_RUN}/checkpoints/unet_s2rosa_jointsr_best.ckpt}"

if [ ! -f "$WARM_START_CKPT" ]; then
  echo "ERROR: stage-1 ckpt not found: ${WARM_START_CKPT}" >&2
  echo "  Run first:  bash scripts/hpc/submit.sh sr/${STAGE1_TAG}.sh STAGE=tune SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}" >&2
  echo "  then:       bash scripts/hpc/submit.sh sr/${STAGE1_TAG}.sh STAGE=fit  SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}" >&2
  exit 1
fi

if [ "${PIN_FROM_STAGE1:-1}" != "0" ]; then
  if [ ! -f "$STAGE1_BEST" ]; then
    echo "ERROR: ${STAGE1_BEST} missing (ckpt exists but no best_params overlay?)." >&2
    echo "  Re-run the stage-1 tune, or set PIN_FROM_STAGE1=0 for a full re-search." >&2
    exit 1
  fi
  _yamlval () { awk -v k="$1:" '$1==k {print $2; exit}' "$STAGE1_BEST"; }
  _lr=$(_yamlval lr)
  _pw=$(_yamlval pos_weight)
  _bs=$(_yamlval batch_size)
  _enc=$(_yamlval encoder_name)
  if [ -z "$_lr" ] || [ -z "$_bs" ] || [ -z "$_enc" ]; then
    echo "ERROR: could not parse lr/batch_size/encoder_name from ${STAGE1_BEST}." >&2
    exit 1
  fi
  # Pin by collapsing each searched range/set to the stage-1 best value;
  # lr_sr keeps its full range — it is the stage-2 search.
  LR_MIN="$_lr"; LR_MAX="$_lr"
  BATCH_SIZES="$_bs"
  ENCODERS="$_enc"
  # pos_weight is only recorded under the legacy loss (arms fix it via LOSS_ARM)
  if [ -n "$_pw" ]; then POS_WEIGHT_MIN="$_pw"; POS_WEIGHT_MAX="$_pw"; fi
  echo "[warm] stage-1 ${STAGE1_TAG}: lr=${_lr} pos_weight=${_pw:-<loss-arm>} batch=${_bs} encoder=${_enc}"
  echo "[warm] pinned; stage-2 searches lr_sr only. UNet init: ${WARM_START_CKPT}"
fi
