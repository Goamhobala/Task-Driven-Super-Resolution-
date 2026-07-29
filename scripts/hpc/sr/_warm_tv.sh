#!/bin/bash
# Shared stage-2 (warm-start) plumbing for the FINAL series' r6/r7 arms —
# sourced BEFORE _stages_tv.sh by r6_new.sh / r7a_new.sh / r7b_new.sh.
# Not submitted directly.
#
# Identical in intent to _warm.sh, with one difference that matters: under the
# train+val protocol the stage-1 arm never produced a val-selected
# `..._best.ckpt`. Its deliverable is `unet_s2rosa_jointsr_final.ckpt` — the end
# of the pre-registered budget. So that is what stage 2 warm-starts from.
#
# In:  STAGE1_TAG  (e.g. r5_new) — the frozen-SR arm this arm warm-starts from.
#      SEED / LOSS_ARM / loss hps / REG / TRAIN_SPLITS — must match the stage-1
#      run; the run-dir naming ties them together automatically.
# Out: WARM_START_CKPT           — stage-1 FINAL ckpt (env override respected)
#      POS_WEIGHT_MIN/MAX, BATCH_SIZES, ENCODERS — pinned to stage-1's
#      best_params.yaml (they define the critic/task, not the optimizer).
#      LR_MIN/LR_MAX — a FINE-TUNING band anchored to stage-1's best lr:
#      [best/100, best], log-uniform. Stage 2 fine-tunes a CONVERGED UNet and
#      the from-scratch optimum is often destructively high for fine-tuning
#      (Adam steps are ~lr-sized regardless of gradient scale, so a converged
#      critic can be walked out of its minimum exactly while the SR net gets
#      its formative gradients). The band lets TPE keep the stage-1 value if
#      it truly transfers — the search covers (lr, lr_sr), 2-D.
#      PIN_LR=1 collapses the band to stage-1's exact lr (old 1-D behaviour).
#      PIN_FROM_STAGE1=0 disables all pinning (full joint re-search).
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"

: "${STAGE1_TAG:?staged arm script must set STAGE1_TAG (e.g. r5_new)}"
SEED="${SEED:-0}"
LOSS_ARM="${LOSS_ARM:-}"
LOSS_TAG=""
[ -n "$LOSS_ARM" ] && LOSS_TAG="_$(echo "$LOSS_ARM" | tr '+' '-')"

# Mirror _stages_tv.sh's run-dir tags so stage 1 and stage 2 always agree on
# which run they mean. Both must be re-derived here because _stages_tv.sh has
# not been sourced yet.
REG="${REG:-true}"
REG_TAG=""
{ [ "$REG" = "false" ] || [ "$REG" = "0" ]; } && REG_TAG="_noreg"
TRAIN_SPLITS="${TRAIN_SPLITS:-train val}"
PROTO_TAG=""
case " ${TRAIN_SPLITS} " in
  *" val "*) : ;;
  *) PROTO_TAG="_holdout" ;;
esac

RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"
STAGE1_RUN="${STAGE1_RUN:-${RUNS_ROOT}/sr_${STAGE1_TAG}${LOSS_TAG}${REG_TAG}${PROTO_TAG}_seed${SEED}}"
STAGE1_BEST="${STAGE1_RUN}/best_params.yaml"
# _final, not _best: this protocol never selects a checkpoint on a holdout.
STAGE1_CKPT_NAME="${STAGE1_CKPT_NAME:-unet_s2rosa_jointsr_final}"
WARM_START_CKPT="${WARM_START_CKPT:-${STAGE1_RUN}/checkpoints/${STAGE1_CKPT_NAME}.ckpt}"

if [ ! -f "$WARM_START_CKPT" ]; then
  echo "ERROR: stage-1 ckpt not found: ${WARM_START_CKPT}" >&2
  echo "  Run first:  bash scripts/hpc/submit.sh sr/${STAGE1_TAG}.sh STAGE=tune SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}" >&2
  echo "  then:       bash scripts/hpc/submit.sh sr/${STAGE1_TAG}.sh STAGE=fit  SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}" >&2
  if [ -f "${STAGE1_RUN}/checkpoints/unet_s2rosa_jointsr_best.ckpt" ]; then
    echo "  NB a val-selected ..._best.ckpt DOES exist in that run dir. That is a" >&2
    echo "  holdout-protocol artefact; warm-starting the final series from it would" >&2
    echo "  quietly reintroduce val-based selection. Refit stage 1 under this" >&2
    echo "  protocol, or set STAGE1_CKPT_NAME explicitly if you mean to." >&2
  fi
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
  # Critic/task knobs: collapse to the stage-1 best value.
  BATCH_SIZES="$_bs"
  ENCODERS="$_enc"
  # pos_weight is only recorded under the legacy loss (arms fix it via LOSS_ARM)
  if [ -n "$_pw" ]; then POS_WEIGHT_MIN="$_pw"; POS_WEIGHT_MAX="$_pw"; fi
  # UNet lr: fine-tuning band [best/100, best] (see header); PIN_LR=1 = exact pin.
  if [ "${PIN_LR:-0}" = "1" ]; then
    LR_MIN="$_lr"; LR_MAX="$_lr"
    _lr_note="lr pinned to ${_lr} (PIN_LR=1); stage-2 searches lr_sr only"
  else
    LR_MIN="$(awk -v lr="$_lr" 'BEGIN{printf "%.6e", lr/100}')"
    LR_MAX="$_lr"
    _lr_note="lr searched in fine-tune band [${LR_MIN}, ${LR_MAX}]; stage-2 searches (lr, lr_sr)"
  fi
  echo "[warm] stage-1 ${STAGE1_TAG}: lr=${_lr} pos_weight=${_pw:-<loss-arm>} batch=${_bs} encoder=${_enc}"
  echo "[warm] ${_lr_note}. UNet init: ${WARM_START_CKPT}"
fi
