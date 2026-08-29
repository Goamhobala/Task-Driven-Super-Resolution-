#!/bin/bash
# Drive ONE R-series arm end to end inside ONE SLURM job:
#
#   phase 1  the TUNED seed, through the arm script: tune -> fit -> bench
#   phase 2  the REFIT seeds, through this arm's refit script (which plants the
#            overlay per seed and runs fit + bench itself)
#
#   ARM=r1a_new REFIT=r1a_new_gap_ce bash run_arm_pool.sh
#
# Normally you do not call this directly: pool_r1a.sh / pool_r1b.sh carry the
# arm, the refit script and the SBATCH headers.
#
# WHY TWO PHASES AND NOT JUST THE REFIT SCRIPT
# --------------------------------------------
# The refit script only ever PLANTS an overlay at a new seed — it cannot make
# one. The overlay comes from the tuned seed, whose run dir is the only place
# `sr.tune` ever writes it, and whose own fit and bench row are equally part of
# the arm's result. So the tuned seed goes through the arm script (which can
# tune) and every other seed goes through the refit script (which cannot).
#
# RESUMABILITY IS THE POINT
# -------------------------
# Re-submit freely. Phase 1's stages are guarded by what they would produce
# (best_params.yaml / fit_state / a store row) and phase 2's by `_refit_lib.sh`'s
# own per-seed guards, so a finished arm costs seconds and a half-trained refit
# resumes from last.ckpt. `set -e` is deliberately NOT set: a failing seed is
# reported and the pool moves on.
set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

: "${ARM:?run_arm_pool.sh needs ARM, the arm script stem (e.g. r1a_new)}"
: "${REFIT:?run_arm_pool.sh needs REFIT, the refit script stem (e.g. r1a_new_gap_ce)}"
LOSS_ARM="${LOSS_ARM:-gap_ce}"
TUNED_SEED="${TUNED_SEED:-66}"     # the series convention (r0/r2a/r2b all 66)
STAGES="${STAGES:-tune fit bench refit}"
ARM_SCRIPT="$REPO_DIR/scripts/hpc/sr/${ARM}.sh"
REFIT_SCRIPT="$REPO_DIR/scripts/hpc/sr/refit/${REFIT}.sh"

for f in "$ARM_SCRIPT" "$REFIT_SCRIPT"; do
  [ -f "$f" ] || { echo "ERROR: no such script: $f" >&2; exit 2; }
done

# --- the environment the guards need ----------------------------------------
# Same reasoning as sr/grid/run_pool.sh: ckpt_epoch returns -1 on any exception
# and in_store reports "absent" on any exception, so a broken venv makes both
# guards fail OPEN — a finished 100-epoch refit silently retrained, duplicate
# rows appended to an append-only store. Checked, loudly.
USER_NAME="${USER:-$(whoami)}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "WARN: no venv at ${VENV_DIR}; relying on whatever python is on PATH" >&2
fi
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/hpc/sr/refit/_refit_lib.sh"   # ckpt_epoch fit_state in_store
if ! python -c "import torch, benchmarking.store" 2>/dev/null; then
  echo "ERROR: python cannot import torch + benchmarking.store." >&2
  echo "  The resume guards would then fail OPEN: a finished refit would be" >&2
  echo "  retrained and the bench would append duplicate rows to an" >&2
  echo "  append-only store. Fix the venv (VENV_DIR=${VENV_DIR}) first." >&2
  exit 2
fi

STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks}"
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-$(( ${SLURM_CPUS_PER_TASK:-8} * ${SLURM_NTASKS:-1} ))}"
NUM_WORKERS=$(( TOTAL_CPUS - 1 ))
[ "$NUM_WORKERS" -lt 1 ] && NUM_WORKERS=1

# Ask the engine for the tuned seed's names rather than rebuilding the tag
# chain here (see the PRINT_RUN_DIR block in _stages_tv.sh).
NAMES=$(env LOSS_ARM="$LOSS_ARM" SEED="$TUNED_SEED" PRINT_RUN_DIR=1 \
            bash "$ARM_SCRIPT" 2>/dev/null)
TUNED_DIR=$(printf '%s\n' "$NAMES" | sed -n 's/^RUN_DIR=//p')
MODEL_NAME=$(printf '%s\n' "$NAMES" | sed -n 's/^MODEL_NAME=//p')
[ -n "$TUNED_DIR" ] || { echo "ERROR: could not resolve ${ARM}'s run dir" >&2; exit 2; }

echo "=============================================================="
echo "ARM POOL — ${ARM}   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  loss   : ${LOSS_ARM}      stages: ${STAGES}"
echo "  tuned  : seed ${TUNED_SEED} -> ${TUNED_DIR}"
echo "  refit  : ${REFIT}.sh (its own SEEDS default unless SEEDS is set here)"
echo "  model  : ${MODEL_NAME}"
echo "  cpus   : ${TOTAL_CPUS} -> num_workers=${NUM_WORKERS}"
echo "  job    : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no jobid})"
echo "=============================================================="

want () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
run_stage () {   # stage [KEY=VALUE ...]
  local stage="$1"; shift
  env LOSS_ARM="$LOSS_ARM" SEED="$TUNED_SEED" STAGE="$stage" \
      NUM_WORKERS="$NUM_WORKERS" REFIT_EPOCHS="$REFIT_EPOCHS" \
      STORE_DIR="$STORE_DIR" "$@" bash "$ARM_SCRIPT"
}

SUMMARY=()
# ---------------------------------------------------------------- phase 1 ---
if want tune; then
  if [ -f "${TUNED_DIR}/best_params.yaml" ] && [ "${FORCE_TUNE:-0}" != "1" ]; then
    echo "[${ARM} s${TUNED_SEED}] TUNE done (best_params.yaml present) — skipping"
  else
    echo "[${ARM} s${TUNED_SEED}] >>> TUNE  ($(date -u +%H:%M:%SZ))"
    if run_stage tune; then SUMMARY+=("tune s${TUNED_SEED}: OK")
    else
      echo "[${ARM} s${TUNED_SEED}] <<< TUNE FAILED — nothing downstream can run" >&2
      SUMMARY+=("tune s${TUNED_SEED}: FAILED"); STAGES=""
    fi
  fi
fi

if want fit; then
  if [ ! -f "${TUNED_DIR}/best_params.yaml" ]; then
    echo "[${ARM} s${TUNED_SEED}] no best_params.yaml — tune this seed first" >&2
    SUMMARY+=("fit s${TUNED_SEED}: NO TUNE"); STAGES=""
  else
    state=fresh
    [ "${FORCE_FIT:-0}" = "1" ] || state=$(fit_state "$TUNED_DIR" "$REFIT_EPOCHS")
    resume=0
    case "$state" in
      done) echo "[${ARM} s${TUNED_SEED}] FIT done (>=${REFIT_EPOCHS} epochs + sweep.json) — skipping" ;;
      sweep_only) echo "[${ARM} s${TUNED_SEED}] weights complete, no sweep.json — resuming TO THE SWEEP"; resume=1 ;;
      resume)     echo "[${ARM} s${TUNED_SEED}] partial fit — RESUMING from last.ckpt"; resume=1 ;;
      sweep_only_nolast)
        echo "[${ARM} s${TUNED_SEED}] weights complete, sweep.json gone, last.ckpt gone —" >&2
        echo "   refusing to retrain ${REFIT_EPOCHS} epochs (see _refit_lib.sh)." >&2
        SUMMARY+=("fit s${TUNED_SEED}: NEEDS SWEEP BY HAND"); STAGES="" ;;
      fresh) echo "[${ARM} s${TUNED_SEED}] >>> FIT (${REFIT_EPOCHS} epochs)  ($(date -u +%H:%M:%SZ))" ;;
    esac
    if [ -n "$STAGES" ] && [ "$state" != "done" ]; then
      if run_stage fit RESUME_FIT="$resume"; then SUMMARY+=("fit s${TUNED_SEED}: OK")
      else echo "[${ARM} s${TUNED_SEED}] <<< FIT FAILED — skipping its bench" >&2
           SUMMARY+=("fit s${TUNED_SEED}: FAILED"); STAGES="" ; fi
    fi
  fi
fi

if want bench; then
  if in_store "$STORE_DIR" "$MODEL_NAME" "$TUNED_SEED" test; then
    echo "[${ARM} s${TUNED_SEED}] BENCH test row already in the store — skipping"
    echo "   (append-only with no dedupe: a second row double-counts this arm)"
  elif [ ! -f "${TUNED_DIR}/sweep.json" ]; then
    echo "[${ARM} s${TUNED_SEED}] no sweep.json — bench has no θ*, skipping" >&2
    SUMMARY+=("bench s${TUNED_SEED}: NO SWEEP")
  else
    echo "[${ARM} s${TUNED_SEED}] >>> BENCH test  ($(date -u +%H:%M:%SZ))"
    if run_stage bench MODEL_NAME="$MODEL_NAME"; then SUMMARY+=("bench s${TUNED_SEED}: OK")
    else SUMMARY+=("bench s${TUNED_SEED}: FAILED"); fi
  fi
fi

# ---------------------------------------------------------------- phase 2 ---
# The refit script hard-refuses without a tuned lr, by design (it is meant to be
# BAKED IN, so the config a refit used is readable in the file that ran it). But
# on the cluster the tuned overlay is right there, so read it, run with it, and
# print it prominently — paste it into the script and the guard goes away.
if want refit; then
  if [ ! -f "${TUNED_DIR}/best_params.yaml" ]; then
    echo "[${REFIT}] no tuned overlay at ${TUNED_DIR} — cannot seed the refits" >&2
    SUMMARY+=("refit: NO TUNE")
  else
    LR="${LR:-$(awk '$1=="lr:" {print $2; exit}' "${TUNED_DIR}/best_params.yaml")}"
    if [ -z "$LR" ]; then
      echo "[${REFIT}] could not read lr from ${TUNED_DIR}/best_params.yaml" >&2
      SUMMARY+=("refit: NO LR")
    else
      echo ""
      echo "  >>> lr = ${LR}   (from seed ${TUNED_SEED}'s tune)"
      echo "  >>> PASTE THIS into scripts/hpc/sr/refit/${REFIT}.sh in place of"
      echo "  >>> __LR__ and delete its LR guard — the point of baking it in is"
      echo "  >>> that the config a refit used is readable in the file that ran it."
      echo ""
      echo "[${REFIT}] >>> REFIT SEEDS  ($(date -u +%H:%M:%SZ))"
      if env LR="$LR" NUM_WORKERS="$NUM_WORKERS" REFIT_EPOCHS="$REFIT_EPOCHS" \
             STORE_DIR="$STORE_DIR" ${SEEDS:+SEEDS="$SEEDS"} \
             bash "$REFIT_SCRIPT"; then
        SUMMARY+=("refit seeds: OK")
      else
        echo "[${REFIT}] <<< refit seeds FAILED (rc=$?)" >&2
        SUMMARY+=("refit seeds: FAILED")
      fi
    fi
  fi
fi

echo ""
echo "=============================================================="
echo "${ARM} finished  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for s in "${SUMMARY[@]}"; do echo "  ${s}"; done
echo ""
echo "Re-submit to pick up anything unfinished: every stage is guarded by what"
echo "it would produce, so finished work costs seconds."
echo "=============================================================="
