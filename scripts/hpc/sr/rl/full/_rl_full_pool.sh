#!/bin/bash
# One full-budget rl arm end to end: tune -> (fit -> bench) x SEEDS.
# Driven by pool_rl2_full.sh / pool_rl4_full.sh; not submitted directly.
#
# Requires: ARM (the script under rl/full/, without .sh).
#
# STAGES (space-separated, default "tune refit"):
#   tune    the search, skipped once best_params.yaml exists
#   refit   fit -> bench per seed, idempotent (resumes, skips finished work)
#   bench   bench each seed's last.ckpt as it stands, without waiting for or
#           checking the fit -- see "BENCH FROM last.ckpt" below.
#
# BENCH FROM last.ckpt (STAGES=bench)
# -----------------------------------
# For runs stopped by hand once their curves plateaued: scores
# checkpoints/last.ckpt under the arm's normal model name. No epoch is read and
# nothing is gated on how far the fit got. θ* comes from the run's sweep.json,
# which is swept on val from last.ckpt first if the fit never wrote one.
# Skipped only if the (model, seed, test) row is already in the store, so a
# rerun never adds a duplicate. Do NOT put `refit` in STAGES for these seeds:
# the refit path would resume them toward REFIT_EPOCHS.
#
# WHY THE OVERLAY IS COPIED, NOT RE-TUNED PER SEED
# ------------------------------------------------
# STAGE=fit refuses to start without ${RUN_DIR}/best_params.yaml, and RUN_DIR
# carries the SEED -- a seed-444 dir has never been tuned and never will be.
# Re-tuning per seed would give each seed its OWN lr and lr_sr, which makes them
# three different arms rather than three replicates of one. So the tuned seed's
# overlay is planted verbatim into each refit seed.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
ARM="${ARM:?set ARM=rl2_full|rl4_full}"
ARM_SCRIPT="$REPO_DIR/scripts/hpc/sr/rl/full/${ARM}.sh"
[ -f "$ARM_SCRIPT" ] || { echo "ERROR: no arm script at $ARM_SCRIPT" >&2; exit 2; }
source "$REPO_DIR/scripts/hpc/sr/refit/_refit_lib.sh"
# _refit_lib's ckpt_epoch / fit_state / in_store shell out to a bare `python`
# that must import torch and benchmarking, and they swallow the failure:
# ckpt_epoch prints -1 and in_store answers "not in the store". Nothing upstream
# activates the venv when this pool is submitted directly (the engine only does
# so later, inside each arm call), so without this every checkpoint read as
# unreadable, no fit ever read as done, and the store guard could not see rows.
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
else
  echo "ERROR: no venv at ${VENV_DIR} — the pool's resume/store guards need its torch." >&2
  exit 2
fi
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
python -c "import torch, benchmarking.store" 2>/dev/null || {
  echo "ERROR: $(command -v python) cannot import torch + benchmarking.store — the guards would silently misreport." >&2
  exit 2; }

TUNE_SEED="${TUNE_SEED:-0}"
SEEDS="${SEEDS:-444 666 888}"
STAGES="${STAGES:-tune refit}"
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_corrected}"

names_for () {   # seed -> RUN_DIR / MODEL_NAME, from the engine, never rebuilt
  env SEED="$1" PRINT_RUN_DIR=1 bash "$ARM_SCRIPT" 2>/dev/null
}
run_arm () {     # stage seed [KEY=VALUE ...]
  local stage="$1" seed="$2"; shift 2
  env SEED="$seed" STAGE="$stage" STORE_DIR="$STORE_DIR" \
      REFIT_EPOCHS="$REFIT_EPOCHS" \
      ${NUM_WORKERS:+NUM_WORKERS="$NUM_WORKERS"} "$@" bash "$ARM_SCRIPT"
}

TUNE_NAMES=$(names_for "$TUNE_SEED")
TUNE_DIR=$(printf '%s\n' "$TUNE_NAMES" | sed -n 's/^RUN_DIR=//p')
[ -n "$TUNE_DIR" ] || { echo "ERROR: could not resolve the arm's run dir" >&2; exit 2; }
OVERLAY="$TUNE_DIR/best_params.yaml"

echo "=== ${ARM}: tune seed ${TUNE_SEED} -> refit seeds ${SEEDS} ==="
echo "===   store=${STORE_DIR}  refit_epochs=${REFIT_EPOCHS} ==="

case " $STAGES " in *" tune "*)
  if [ -f "$OVERLAY" ]; then
    echo "### tune: ${OVERLAY} already exists — skipping the search"
  else
    echo "### TUNE (seed ${TUNE_SEED})"
    run_arm tune "$TUNE_SEED"
  fi
;; esac

[ -f "$OVERLAY" ] || {
  echo "ERROR: no overlay at ${OVERLAY} — the tune produced no best_params." >&2
  echo "  Every trial may have failed or pruned; check the study before refitting." >&2
  exit 3; }

case " $STAGES " in *" refit "*|*" bench "*) : ;; *) exit 0 ;; esac

bench_last () {   # seed run_dir model_name -> bench checkpoints/last.ckpt as-is
  local seed="$1" run_dir="$2" model_name="$3"
  local ckpt="$run_dir/checkpoints/last.ckpt"
  if [ ! -f "$ckpt" ]; then
    echo "### bench: no ${ckpt} — skipping" >&2
    return 0
  fi
  if in_store "$STORE_DIR" "$model_name" "$seed" test; then
    echo "### bench: ${model_name} seed ${seed} already in the store — skipping (append-only, no dedupe)"
    return 0
  fi
  # clDice is always in the row: an inherited TILE_METRICS=apls (sbatch exports
  # the submitting shell) would otherwise drop it, and a store missing it
  # cannot be reported against arms that have it.
  local tm="${TILE_METRICS:-apls,cldice}"
  case ",${tm}," in *,cldice,*) : ;; *) tm="${tm:+${tm},}cldice" ;; esac
  echo "### BENCH last.ckpt  model_name=${model_name}  seed=${seed}  tile_metrics=${tm}"
  run_arm bench "$seed" MODEL_NAME="$model_name" BENCH_CKPT="$ckpt" \
      BENCH_SWEEP_OUT="$run_dir/sweep.json" TILE_METRICS="$tm"
}

for SEED in $SEEDS; do
  if [ "$SEED" = "$TUNE_SEED" ]; then
    echo "### seed ${SEED} IS the tuned seed — skipping (its row comes from the tune's own fit)"
    continue
  fi
  NAMES=$(names_for "$SEED")
  RUN_DIR=$(printf '%s\n' "$NAMES" | sed -n 's/^RUN_DIR=//p')
  MODEL_NAME=$(printf '%s\n' "$NAMES" | sed -n 's/^MODEL_NAME=//p')
  [ -n "$RUN_DIR" ] && [ -n "$MODEL_NAME" ] || { echo "ERROR: names for seed ${SEED}" >&2; exit 2; }

  echo
  echo "##################################################################"
  echo "### ${ARM}  seed=${SEED}"
  echo "##################################################################"
  mkdir -p "$RUN_DIR"
  cp "$OVERLAY" "$RUN_DIR/best_params.yaml"

  case " $STAGES " in *" bench "*)
    bench_last "$SEED" "$RUN_DIR" "$MODEL_NAME" || \
      echo "### ${ARM} seed ${SEED}: BENCH FAILED — continuing" >&2
  ;; esac
  case " $STAGES " in *" refit "*) : ;; *) continue ;; esac

  state=fresh
  [ "${FORCE_FIT:-0}" = "1" ] || state=$(fit_state "$RUN_DIR" "$REFIT_EPOCHS")
  resume=0
  case "$state" in
    done) echo "### FIT complete — skipping" ;;
    sweep_only|resume) echo "### partial — RESUMING from last.ckpt"; resume=1 ;;
    sweep_only_nolast)
      echo "### weights complete but sweep.json AND last.ckpt are gone — skipping" >&2
      continue ;;
    fresh) echo "### FIT (${REFIT_EPOCHS} epochs, train+val)" ;;
  esac
  if [ "$state" != "done" ]; then
    run_arm fit "$SEED" RESUME_FIT="$resume" || {
      echo "### ${ARM} seed ${SEED}: FIT FAILED — continuing to the next seed" >&2
      continue; }
  fi

  if in_store "$STORE_DIR" "$MODEL_NAME" "$SEED" test; then
    echo "### BENCH already in the store — skipping (append-only, no dedupe)"
  elif [ ! -f "${RUN_DIR}/sweep.json" ]; then
    echo "### no sweep.json — bench has no theta*, skipping" >&2
  else
    run_arm bench "$SEED" MODEL_NAME="$MODEL_NAME" || \
      echo "### BENCH FAILED — continuing" >&2
  fi
  # Kept by default, same reasoning as _refit_lib.sh: last.ckpt is the resume
  # point and a 100-epoch joint fit does not fit one allocation.
  [ "${KEEP_LAST:-1}" = "1" ] || rm -f "${RUN_DIR}/checkpoints/last.ckpt"
done

echo
echo "=== ${ARM}: seeds ${SEEDS} done ==="
