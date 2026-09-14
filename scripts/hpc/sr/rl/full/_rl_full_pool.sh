#!/bin/bash
# One full-budget rl arm end to end: tune -> (fit -> bench) x SEEDS.
# Driven by pool_rl2_full.sh / pool_rl4_full.sh; not submitted directly.
#
# Requires: ARM (the script under rl/full/, without .sh).
#
# STAGES (space-separated, default "tune refit"):
#   tune    the search, skipped once best_params.yaml exists
#   refit   fit -> bench per seed, idempotent (resumes, skips finished work)
#   bench   bench a seed's CURRENT last.ckpt without waiting for the fit to
#           finish -- see "PARTIAL BENCH" below. Safe beside a running fit.
#
# PARTIAL BENCH (STAGES=bench)
# ----------------------------
# Scores the newest readable last*.ckpt of each seed at its own val θ*, as the
# fit would, but under its OWN model name, <arm model>_partial_epNNN:
#   * the store is append-only with no dedupe, and in_store matches on the name
#     -- a partial row under the arm's name would make the finished fit's bench
#     skip itself, leaving the store with an epoch-N number labelled final;
#   * the checkpoint is COPIED first and re-read, so a fit still writing it
#     cannot hand the scorer a torn file, and the row's epoch is pinned;
#   * θ* goes to partial_bench/epNNN/sweep.json, never the fit's sweep.json.
# The copy is deleted after a successful bench (KEEP_PARTIAL_CKPT=1 keeps it).
# BENCH_WANDB defaults to 0 here, so partial numbers never land in the fit's
# wandb summary under the keys the final bench will use; set 1 to push them.
# Rerunning at the same epoch costs seconds; at a later epoch it adds a new row.
# A seed whose fit is already complete is left to the regular bench.
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

partial_bench () {   # seed run_dir model_name -> bench last.ckpt as it stands now
  local seed="$1" run_dir="$2" model_name="$3" ck f ep best="" best_ep=-1
  ck="$run_dir/checkpoints"
  # last.ckpt is what the user resumes from, but a resume in a DIFFERENT
  # checkpoint dir writes last-v1.ckpt and freezes last.ckpt, so take the
  # newest epoch among them rather than trusting the name.
  for f in "$ck"/last.ckpt "$ck"/last-v*.ckpt; do
    [ -f "$f" ] || continue
    ep=$(ckpt_epoch "$f")
    [ "$ep" -gt "$best_ep" ] && { best="$f"; best_ep="$ep"; }
  done
  if [ -z "$best" ] || [ "$best_ep" -lt 0 ]; then
    echo "### partial bench: no readable last.ckpt under ${ck} — skipping" >&2
    return 0
  fi
  if [ "$(fit_state "$run_dir" "$REFIT_EPOCHS")" = "done" ]; then
    echo "### partial bench: fit is complete — the regular bench owns this seed, skipping"
    return 0
  fi

  local tag; tag=$(printf 'ep%03d' "$best_ep")
  local pb="$run_dir/partial_bench/$tag" name="${model_name}_partial_${tag}"
  echo "### PARTIAL BENCH  $(basename "$best") epoch=${best_ep}  model_name=${name}"
  if in_store "$STORE_DIR" "$name" "$seed" test; then
    echo "### ${name} already in the store — skipping (append-only, no dedupe)"
    return 0
  fi

  mkdir -p "$pb"
  if [ ! -f "$pb/model.ckpt" ]; then
    cp "$best" "$pb/model.ckpt.tmp"
    # Re-read the COPY: if the fit rewrote last.ckpt mid-copy the epoch differs
    # or the file will not load, and the scorer must never see that.
    if [ "$(ckpt_epoch "$pb/model.ckpt.tmp")" != "$best_ep" ]; then
      rm -f "$pb/model.ckpt.tmp"
      echo "### partial bench: copy of $(basename "$best") changed under us (fit saving?) — rerun" >&2
      return 1
    fi
    mv -f "$pb/model.ckpt.tmp" "$pb/model.ckpt"
  fi

  run_arm bench "$seed" MODEL_NAME="$name" BENCH_CKPT="$pb/model.ckpt" \
      BENCH_SWEEP_OUT="$pb/sweep.json" BENCH_WANDB="${BENCH_WANDB:-0}" || {
    echo "### PARTIAL BENCH FAILED (copy kept at ${pb}/model.ckpt)" >&2
    return 1; }
  [ "${KEEP_PARTIAL_CKPT:-0}" = "1" ] || rm -f "$pb/model.ckpt"
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
    partial_bench "$SEED" "$RUN_DIR" "$MODEL_NAME" || \
      echo "### ${ARM} seed ${SEED}: partial bench did not complete — continuing" >&2
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
