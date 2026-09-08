#!/bin/bash
# One full-budget rl arm end to end: tune -> (fit -> bench) x SEEDS.
# Driven by pool_rl2_full.sh / pool_rl4_full.sh; not submitted directly.
#
# Requires: ARM (the script under rl/full/, without .sh).
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

TUNE_SEED="${TUNE_SEED:-0}"
SEEDS="${SEEDS:-444 666 888}"
STAGES="${STAGES:-tune refit}"
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_newdata}"

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

case " $STAGES " in *" refit "*) : ;; *) exit 0 ;; esac

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
  [ "${KEEP_LAST:-0}" = "1" ] || rm -f "${RUN_DIR}/checkpoints/last.ckpt"
done

echo
echo "=== ${ARM}: seeds ${SEEDS} done ==="
