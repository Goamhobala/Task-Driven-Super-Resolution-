#!/bin/bash
# ONE CELL of the SR4RS lr_sr grid: fit -> bench at one pinned lr_sr.
#
# SUBMIT ONE CELL PER JOB. An SR4RS joint fit runs 2-4 days and the queue limit
# is 48 h, so a four-cell lane cannot finish in one allocation -- it would just
# burn a slot and leave three cells untouched. There is deliberately no lane
# pool for r4 (unlike grid/run_pool.sh, whose SEN2SR-Lite cells are ~10x
# cheaper and do fit).
#
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=48:00:00 \
#          -J r4grid_on_ls1e-4 -o slurm-%x-%j.txt \
#          scripts/hpc/train.sbatch --SCRIPT=sr/grid/run_pool_r4.sh \
#          HC=on LRSRS=1e-4
#
# LRSRS still accepts a list, so a cheap cell pair can share a job if you want
# -- but the default assumption is one cell, one job. RESUBMIT the same command
# to continue an unfinished cell: the fit resumes from last.ckpt and a finished
# one costs seconds.
#
# NO TUNE STAGE, unlike grid/run_pool.sh. Both axes are pinned -- lr at 2e-4,
# lr_sr by the cell -- so there is nothing to search. STAGE=fit refuses to start
# without ${RUN_DIR}/best_params.yaml, so this script PLANTS one per cell,
# composed from the cell's own coordinates. That is the whole difference from
# the r2 lane driver.
#
# Requires: HC (on|off), LRSRS (space-separated rates).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
HC="${HC:?set HC=on|off}"
LRSRS="${LRSRS:?set LRSRS='1e-4 1e-5 1e-6 1e-7'}"
CELL_SCRIPT="$REPO_DIR/scripts/hpc/sr/r4grid_new.sh"
# ckpt_epoch / fit_state / in_store — the same resume guards the R-series
# refits use, so a finished cell costs seconds and a partial one resumes.
source "$REPO_DIR/scripts/hpc/sr/refit/_refit_lib.sh"

SEED="${SEED:-0}"                       # one seed, no replicates
LR="${LR:-0.0002}"                      # pinned; see r4grid_new.sh
# 100, MATCHING r2grid -- verified against the completed cells, all eight of
# which reached epoch 100. This grid exists to be read against that one, so the
# budget is part of the comparison, not a tunable. DO NOT lower it to fit the
# 48 h queue: a 50-epoch r4 cell cannot be set beside a 100-epoch r2 cell, and
# the honest way to handle the wall clock is to RESUBMIT (the fit resumes from
# last.ckpt and loses nothing).
FIT_EPOCHS="${FIT_EPOCHS:-${REFIT_EPOCHS:-100}}"
STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_newdata}"
# Grid cells are EXPECTED to collapse or die numerically, and a dead cell you
# cannot resume or re-inspect is one you re-run from zero.
KEEP_LAST="${KEEP_LAST:-1}"
HC_MASK="${HC_MASK_PATH:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"

if [ "$HC" = "on" ]; then SR_PAD=8; HC_LINE="  sr_hc: 'on'
  hc_mask_path: ${HC_MASK}"; else SR_PAD=0; HC_LINE="  sr_hc: 'off'"; fi

for LRSR in $LRSRS; do
  NAMES=$(env HC="$HC" LRSR="$LRSR" SEED="$SEED" PRINT_RUN_DIR=1 \
              bash "$CELL_SCRIPT" 2>/dev/null)
  RUN_DIR=$(printf '%s\n' "$NAMES" | sed -n 's/^RUN_DIR=//p')
  MODEL_NAME=$(printf '%s\n' "$NAMES" | sed -n 's/^MODEL_NAME=//p')
  if [ -z "$RUN_DIR" ] || [ -z "$MODEL_NAME" ]; then
    echo "ERROR: could not resolve names for HC=${HC} LRSR=${LRSR}" >&2; exit 2
  fi
  RUN_TAG="${RUN_DIR##*/}"; RUN_TAG="${RUN_TAG%_seed${SEED}}"
  echo
  echo "##################################################################"
  echo "### ${RUN_TAG}  seed=${SEED}  lr=${LR}  lr_sr=${LRSR}"
  echo "##################################################################"

  mkdir -p "$RUN_DIR"
  # The overlay is COMPOSED, not copied from a tune: there is no tune. The loss
  # block is the frozen control every arm of this family carries verbatim, so
  # the grid sits on the same loss surface as r4a/r4b.
  cat > "$RUN_DIR/best_params.yaml" <<YAML
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  upsampler: sr4rs
  freeze_sr: false
  sr_pad: ${SR_PAD}
${HC_LINE}
  lr: ${LR}
  lr_sr: ${LRSR}
  loss_arm: gap_ce
  pstar: gap_ce
  gap_r: 4
  gap_k: 60.0
  tl_ell: 5
  tl_theta: 0.375
  gap_theta: 0.55836
  tversky_alpha: 0.7
  cl_alpha: 0.3
  cl_iters: 5
  sr_w: 1.0
  sr_radius: 1
  warmup_start: 30
  warmup_ramp: 10
  mix_w: 0.6075946831862098
  pos_weight: 4.77222
  lr_schedule: cosine
  sr_warmup_epochs: 1.0
  l2sp_lambda: 0.0
  adaptive_norm: true
  adaptive_norm_momentum: 0.01
  norm_recalibrate: post
  std_band_raise_lo: 0.01
  std_band_raise_hi: 100.0
data:
  batch_size: 4
  mask_source: raster
  mask_dirname: mask_new_2pt5
trainer:
  precision: bf16-mixed
YAML

  run_cell () { env HC="$HC" LRSR="$LRSR" SEED="$SEED" STAGE="$1" \
      STORE_DIR="$STORE_DIR" REFIT_EPOCHS="$FIT_EPOCHS" \
      ${NUM_WORKERS:+NUM_WORKERS="$NUM_WORKERS"} "${@:2}" bash "$CELL_SCRIPT"; }

  state=fresh
  [ "${FORCE_FIT:-0}" = "1" ] || state=$(fit_state "$RUN_DIR" "$FIT_EPOCHS")
  resume=0
  case "$state" in
    done) echo "### FIT complete — skipping" ;;
    sweep_only|resume) echo "### partial — RESUMING from last.ckpt"; resume=1 ;;
    sweep_only_nolast)
      echo "### weights complete, sweep.json and last.ckpt both gone — skipping" >&2
      continue ;;
    fresh) echo "### FIT (${FIT_EPOCHS} epochs, train+val)" ;;
  esac
  if [ "$state" != "done" ]; then
    # A cell that dies numerically is a RESULT, not a script failure: record
    # where it got to and carry on to the next rate.
    run_cell fit RESUME_FIT="$resume" || {
      echo "### ${RUN_TAG}: FIT FAILED — numerical death at a step is an observation" >&2
      continue; }
  fi

  if in_store "$STORE_DIR" "$MODEL_NAME" "$SEED" test; then
    echo "### BENCH already in the store — skipping (append-only, no dedupe)"
  elif [ ! -f "${RUN_DIR}/sweep.json" ]; then
    echo "### no sweep.json — bench has no theta*, skipping" >&2
  else
    run_cell bench MODEL_NAME="$MODEL_NAME" || echo "### BENCH FAILED — continuing" >&2
  fi
  [ "$KEEP_LAST" = "1" ] || rm -f "${RUN_DIR}/checkpoints/last.ckpt"
done

echo
echo "=== r4grid lane HC=${HC}: rates ${LRSRS} done ==="
