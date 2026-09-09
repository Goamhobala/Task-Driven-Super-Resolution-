#!/bin/bash
# Drive ONE HC LANE of the lr_sr mechanism grid — every pinned lr_sr, all three
# stages, inside ONE SLURM job.  (docs/lrsr_grid_ablation_plan.md)
#
#   HC=off LRSRS="1e-4 1e-5" bash run_pool.sh
#
# STAGES defaults to "tune fit bench" — seed 0 of each cell. Add `refit` to run
# a cell's REPLICATE SEEDS through its generated, overlay-baked refit script:
#
#   HC=off LRSRS="1e-4" STAGES=refit SEEDS="1 2" bash run_pool.sh
#
# Normally you do not call this directly: pool_r2a.sh / pool_r2b.sh carry the
# lane, the cell list and the SBATCH headers.
#
# ONE GPU, ONE CELL AT A TIME — NOT the loss pool's lane fan-out
# --------------------------------------------------------------
# loss/refit/run_pool.sh deals arms round-robin across NGPU lanes because those
# arms are independent. A grid cell is not: its three stages are a CHAIN. The
# fit reads the tune's best_params.yaml and the bench reads the fit's sweep.json,
# so a cell is inherently sequential, and two cells sharing a card would halve
# the throughput of both while doubling the blast radius of an OOM. The plan
# prices the lane at ~a weekend of one GPU (§8) on exactly that basis.
#
# RESUMABILITY IS THE POINT
# -------------------------
# Four cells x (tune + 100-epoch refit + bench) does not reliably fit one wall
# clock, so this pool is built to be RE-SUBMITTED. Every stage is guarded by
# what it would produce:
#
#   tune   best_params.yaml exists            -> skip
#   fit    fit_state (shared with the refit pool: >=REFIT_EPOCHS in the final
#          ckpt AND sweep.json) -> skip; partial -> RESUME_FIT=1 from last.ckpt
#   bench  the (model_name, seed, test) row is already in the store -> skip
#
# The tune guard is why best_params.yaml, not study.db: sr.tune writes the
# overlay once, after the study finishes, whereas re-entering a FINISHED study
# would happily add another N_TRIALS to it — a silently larger search budget for
# whichever cells happened to straddle a timeout, which is not a grid any more.
#
# Nothing here is destructive: a cell that fails is reported and the pool moves
# on (`set -e` is deliberately NOT set), so one numerically-dead cell — an
# EXPECTED outcome at lr_sr=1e-4, per §3 — cannot take the other three with it.
set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

: "${HC:?run_pool.sh needs HC=on|off (the hard-constraint lane)}"
: "${LRSRS:?run_pool.sh needs LRSRS, e.g. \"1e-4 1e-5 1e-6 1e-7\"}"
SEED="${SEED:-0}"                 # §2: seed 0 for all eight cells first
STAGES="${STAGES:-tune fit bench}"
CELL_SCRIPT="$REPO_DIR/scripts/hpc/sr/r2grid_new.sh"

# --- the fourth, OPT-IN stage: replicate seeds ------------------------------
# `refit` is NOT in the default STAGES, and should not be. §2 and §10: seed 0
# for every cell first, additional seeds ONLY for a cell that ends up carrying
# a QUANTITATIVE sentence. The grid already replicates along the lr_sr axis —
# four runs per lane with a dose-response between them — and §4 forbids leaning
# on between-cell metric differences at n=1 regardless, so seeds buy an error
# bar on one number, not the mechanism claim.
#
# It runs a DIFFERENT script per cell: grid/refit/r2grid_<lane>_ls<rate>.sh,
# which carries that cell's tuned overlay BAKED IN. It has to. A seed-N run dir
# has never been tuned and never will be (RUN_DIR carries the seed), so the
# overlay must be planted; and re-tuning per seed would hand each seed a
# different lr, which is a different arm, not a replicate. Generate those
# scripts from the synced run dirs with
#   python scripts/local/make_grid_refit_scripts.py
# SEEDS is passed through; each script's own default applies when unset.
SEEDS="${SEEDS:-}"
REFIT_DIR="$REPO_DIR/scripts/hpc/sr/grid/refit"

# last.ckpt is KEPT here, unlike the loss refit pool which deletes it after a
# successful bench. Grid cells are expected to collapse or die numerically (§3),
# and a dead cell you cannot resume or re-inspect is a cell you have to re-run
# from zero. The per-epoch SR snapshots dominate this run dir's size anyway.
KEEP_LAST="${KEEP_LAST:-1}"

# --- the environment the guards need ----------------------------------------
# _stages_tv.sh activates the venv itself, but not until well after the point
# this pool needs `python` for its OWN resume checks. Activate here too.
#
# This matters more than it looks: ckpt_epoch swallows errors and returns -1,
# and in_store returns "not present" on any exception. With no importable torch
# BOTH guards fail OPEN — every cell would look unfinished and a completed
# 100-epoch refit would be silently retrained. So it is checked, loudly.
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
  echo "  The resume guards would then fail OPEN: every finished cell would be" >&2
  echo "  re-tuned and re-fitted, and the bench would append duplicate rows to" >&2
  echo "  an append-only store. Fix the venv (VENV_DIR=${VENV_DIR}) first." >&2
  exit 2
fi

STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_corrected}"
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"

# One cell at a time, so the whole allocation is this cell's. One core is left
# for the training process itself.
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-$(( ${SLURM_CPUS_PER_TASK:-8} * ${SLURM_NTASKS:-1} ))}"
NUM_WORKERS=$(( TOTAL_CPUS - 1 ))
[ "$NUM_WORKERS" -lt 1 ] && NUM_WORKERS=1

read -r -a LRSR_ARR <<< "$LRSRS"
echo "=============================================================="
echo "lr_sr GRID POOL — lane HC=${HC}   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  cells  : ${#LRSR_ARR[@]}   lr_sr = ${LRSRS}"
echo "  stages : ${STAGES}          seed: ${SEED}   refit_epochs: ${REFIT_EPOCHS}"
echo "  cpus   : ${TOTAL_CPUS} -> num_workers=${NUM_WORKERS}"
echo "  job    : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no jobid})"
echo "  NOTE   : descriptive-only cells; test selects nothing (§4 pre-commitment)"
echo "=============================================================="

want () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

run_stage () {   # stage [KEY=VALUE ...]
  local stage="$1"; shift
  env HC="$HC" LRSR="$LRSR" SEED="$SEED" STAGE="$stage" \
      NUM_WORKERS="$NUM_WORKERS" REFIT_EPOCHS="$REFIT_EPOCHS" \
      STORE_DIR="$STORE_DIR" "$@" \
      bash "$CELL_SCRIPT"
}

SUMMARY=()
for LRSR in "${LRSR_ARR[@]}"; do
  echo ""
  echo "##############################################################"
  echo "### CELL  HC=${HC}  lr_sr=${LRSR}   $(date -u +%H:%M:%SZ)"
  echo "##############################################################"

  # Ask the engine for this cell's names rather than rebuilding the tag chain
  # here — see the PRINT_RUN_DIR block in _stages_tv.sh for why guessing it is
  # the one mistake that makes every guard inspect the wrong directory.
  NAMES=$(env HC="$HC" LRSR="$LRSR" SEED="$SEED" PRINT_RUN_DIR=1 \
              bash "$CELL_SCRIPT" 2>/dev/null)
  RUN_DIR=$(printf '%s\n' "$NAMES" | sed -n 's/^RUN_DIR=//p')
  MODEL_NAME=$(printf '%s\n' "$NAMES" | sed -n 's/^MODEL_NAME=//p')
  if [ -z "$RUN_DIR" ] || [ -z "$MODEL_NAME" ]; then
    echo "[cell ${LRSR}] cannot resolve run dir (bad LRSR? bad HC?) — SKIPPING" >&2
    SUMMARY+=("${LRSR}: UNRESOLVED")
    continue
  fi
  echo "    run_dir    : ${RUN_DIR}"
  echo "    model_name : ${MODEL_NAME}"
  cell_status=""

  # --- 1. TUNE (lr alone, 15 trials, loosened rails) -------------------------
  if want tune; then
    if [ "${FORCE_TUNE:-0}" != "1" ] && [ -f "${RUN_DIR}/best_params.yaml" ]; then
      echo "[cell ${LRSR}] TUNE done (best_params.yaml present) — skipping"
    else
      echo "[cell ${LRSR}] >>> TUNE  ($(date -u +%H:%M:%SZ))"
      if run_stage tune; then
        echo "[cell ${LRSR}] <<< TUNE ok"
      else
        echo "[cell ${LRSR}] <<< TUNE FAILED (rc=$?) — no best_params, skipping the rest" >&2
        SUMMARY+=("${LRSR}: TUNE FAILED")
        continue
      fi
    fi
  fi

  # --- 2. FIT (100 epochs) + test + val theta* sweep -------------------------
  if want fit; then
    if [ ! -f "${RUN_DIR}/best_params.yaml" ]; then
      echo "[cell ${LRSR}] no best_params.yaml — FIT needs a tune first, skipping" >&2
      SUMMARY+=("${LRSR}: NO TUNE")
      continue
    fi
    state=fresh
    [ "${FORCE_FIT:-0}" = "1" ] || state=$(fit_state "$RUN_DIR" "$REFIT_EPOCHS")
    resume=0
    case "$state" in
      done)  echo "[cell ${LRSR}] FIT done (>=${REFIT_EPOCHS} epochs + sweep.json) — skipping" ;;
      sweep_only)
        echo "[cell ${LRSR}] weights complete but no sweep.json — resuming TO THE SWEEP"
        echo "               (at max_epochs there is nothing left to train: minutes, not hours)"
        resume=1 ;;
      resume)
        echo "[cell ${LRSR}] partial fit — RESUMING from last.ckpt (FORCE_FIT=1 to restart)"
        resume=1 ;;
      sweep_only_nolast)
        echo "[cell ${LRSR}] weights complete, sweep.json missing, last.ckpt gone." >&2
        echo "               Refusing to retrain ${REFIT_EPOCHS} epochs; produce sweep.json" >&2
        echo "               by hand (see _refit_lib.sh) or set FORCE_FIT=1." >&2
        SUMMARY+=("${LRSR}: NEEDS SWEEP BY HAND")
        continue ;;
      fresh) echo "[cell ${LRSR}] >>> FIT (${REFIT_EPOCHS} epochs, train+val)  ($(date -u +%H:%M:%SZ))" ;;
    esac
    if [ "$state" != "done" ]; then
      if run_stage fit RESUME_FIT="$resume"; then
        echo "[cell ${LRSR}] <<< FIT ok"
      else
        rc=$?
        # A cell that dies numerically is a RESULT (§3), not a pool failure:
        # record the point it reached and carry on to the next cell.
        echo "[cell ${LRSR}] <<< FIT FAILED (rc=${rc}) — see the log for the step it" >&2
        echo "               reached; numerical death at step X is an observation." >&2
        SUMMARY+=("${LRSR}: FIT FAILED (rc=${rc})")
        continue
      fi
    fi
  fi

  # --- 3. BENCH on test ------------------------------------------------------
  if want bench; then
    if in_store "$STORE_DIR" "$MODEL_NAME" "$SEED" test; then
      echo "[cell ${LRSR}] BENCH test row already in the store — skipping"
      echo "               (the store is append-only with no dedupe: a second row"
      echo "                would double-count this cell's chips)"
    elif [ ! -f "${RUN_DIR}/sweep.json" ]; then
      echo "[cell ${LRSR}] no sweep.json — bench has no theta*, skipping" >&2
      cell_status="NO SWEEP"
    else
      echo "[cell ${LRSR}] >>> BENCH test  ($(date -u +%H:%M:%SZ))"
      if run_stage bench MODEL_NAME="$MODEL_NAME"; then
        echo "[cell ${LRSR}] <<< BENCH ok"
      else
        echo "[cell ${LRSR}] <<< BENCH FAILED (rc=$?) — continuing" >&2
        cell_status="BENCH FAILED"
      fi
    fi
  fi

  # --- 4. REFIT: replicate seeds of this cell (opt-in) -----------------------
  # Delegated to the cell's generated refit script, which plants its baked
  # overlay per seed and runs fit + bench through _refit_lib.sh's own guards —
  # the same guards used above, so a completed seed costs seconds here too.
  if want refit; then
    _cell_refit="${REFIT_DIR}/r2grid_${HC}_ls$(printf '%s' "$LRSR" | awk '{
      split(sprintf("%.1e", $0), a, "e"); m = a[1]; sub(/\.0$/, "", m)
      printf "%se%d", m, a[2] + 0 }').sh"
    if [ ! -f "$_cell_refit" ]; then
      echo "[cell ${LRSR}] no refit script at ${_cell_refit}" >&2
      echo "               generate it: python scripts/local/make_grid_refit_scripts.py" >&2
      cell_status="NO REFIT SCRIPT"
    else
      echo "[cell ${LRSR}] >>> REFIT seeds${SEEDS:+ (${SEEDS})}  ($(date -u +%H:%M:%SZ))"
      if env ${SEEDS:+SEEDS="$SEEDS"} STORE_DIR="$STORE_DIR" \
             NUM_WORKERS="$NUM_WORKERS" REFIT_EPOCHS="$REFIT_EPOCHS" \
             KEEP_LAST="$KEEP_LAST" bash "$_cell_refit"; then
        echo "[cell ${LRSR}] <<< REFIT seeds ok"
      else
        echo "[cell ${LRSR}] <<< REFIT seeds FAILED (rc=$?) — continuing" >&2
        cell_status="REFIT FAILED"
      fi
    fi
  fi

  [ "$KEEP_LAST" = "1" ] || rm -f "${RUN_DIR}/checkpoints/last.ckpt"
  SUMMARY+=("${LRSR}: ${cell_status:-OK}")
done

echo ""
echo "=============================================================="
echo "lane HC=${HC} finished  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for s in "${SUMMARY[@]}"; do echo "  ${s}"; done
echo ""
echo "Re-submit this same pool to pick up anything unfinished: every stage is"
echo "guarded by what it would produce, so finished cells cost seconds."
echo "=============================================================="
