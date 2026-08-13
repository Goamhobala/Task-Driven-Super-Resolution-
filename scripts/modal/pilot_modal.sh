#!/bin/bash
# Loss-pilot orchestrator for MODAL (serverless container, one GPU per call).
# Third port of the same harness: it runs the assigned arms SEQUENTIALLY
# through tune -> fit -> bench by calling the SHARED
# scripts/LightningStudio/loss/l*_new.sh arm scripts, exactly as
# pilot_seq.sh (Lightning) and pilot_kaggle.sh (Kaggle) do. Nothing about the
# protocol changes here — only where the container gets its paths from.
#
# IDEMPOTENT: re-invoke until it prints ALL ARMS DONE. It skips finished work,
# tops up a partial Optuna study to TARGET_TRIALS, rescues a killed tune
# (writes the overlay from the existing study), and resumes a killed fit from
# last.ckpt. State lives on a Modal Volume, so it survives container death,
# preemption and the 24 h Function timeout.
#
# Not meant to be run by hand — scripts/modal/modal_app.py sets the paths and
# invokes it inside the container. To drive it locally for a smoke test:
#   REPO_DIR=$PWD DATASET_DIR=... RUNS_ROOT=... bash scripts/modal/pilot_modal.sh
#
# HOW THIS DIFFERS FROM pilot_kaggle.sh (and why):
#   * No state-restore block. Kaggle starts empty every session and has to copy
#     the previous notebook Output back in; a Modal Volume is already there.
#   * PRECISION defaults to bf16-mixed, NOT 16-mixed. The default GPU (L4) has
#     native bf16, so arms run in the SAME numeric regime as the finished
#     Lightning L4 arms. Do not run pilot arms on a T4 here — it would silently
#     make precision a between-arm difference. (See the PRECISION guard below.)
#   * MAX_SECONDS is sized against Modal's 24 h Function timeout rather than
#     Kaggle's 12 h session cap; the exit-cleanly-and-resume behaviour is the
#     same, and matters more here because the driver commits the Volume on a
#     clean return.
set -euo pipefail
trap 'echo "!! pilot_modal.sh ABORTED at line $LINENO (exit $?)" >&2' ERR

# --- Paths (all set by modal_app.py; defaults match its mount layout) --------
export REPO_DIR="${REPO_DIR:-/root/InstaRoadPrototype}"
export INSTAROAD_ROOT="${INSTAROAD_ROOT:-/out}"
export DATASET_DIR="${DATASET_DIR:-/data/ROSA_New}"
export RUNS_ROOT="${RUNS_ROOT:-${INSTAROAD_ROOT}/runs}"
export STORE_DIR="${STORE_DIR:-${INSTAROAD_ROOT}/benchmarks_loss_pilot}"
export SEN2SR_DIR="${SEN2SR_DIR:-/data/models/SEN2SRLite_RGBN}"
# No venv in the container (deps are installed into the image's system python).
# _stages_tv.sh tolerates this: it warns and uses `which python`, same as Kaggle.
export VENV_DIR="${VENV_DIR:-/nonexistent}"

# --- Compute ------------------------------------------------------------------
export PRECISION="${PRECISION:-bf16-mixed}"
export SEARCH_GPUS="${SEARCH_GPUS:-1}"      # one tuner per visible GPU
export REFIT_GPUS="${REFIT_GPUS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"      # cpu=4 by default in modal_app.py
export WANDB_MODE="${WANDB_MODE:-online}"

# PROTOCOL CONSTANT (amendment 2026-08-03a). Batch is fixed at 8 for every arm.
# An arm that does not fit batch 8 belongs on a bigger card, it does NOT drop
# the batch. On Modal the fix is a one-word change (gpu="A10"/"L40S"), so there
# is no excuse to break the constant here.
export BATCH_SIZES="${BATCH_SIZES:-8}"

# bf16 is part of that same constancy argument. A T4/V100 has no native bf16 and
# Lightning would fall back or crawl; refuse rather than silently produce an arm
# that is not comparable to the others.
if [ "${PRECISION}" = "bf16-mixed" ] && command -v python3 >/dev/null 2>&1; then
  if ! python3 - <<'PY'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(0)          # CPU smoke test: let the engine's own guards speak
    # Compute capability, not torch.cuda.is_bf16_supported(): the latter counts
    # software emulation, which is exactly the case we must refuse. Native bf16
    # starts at Ampere (8.0). T4 = 7.5, V100 = 7.0, L4 = 8.9, A100 = 8.0.
    sys.exit(0 if torch.cuda.get_device_capability()[0] >= 8 else 1)
except Exception:
    sys.exit(0)              # no torch yet: the engine will complain properly
PY
  then
    echo "ERROR: PRECISION=bf16-mixed but this GPU has no native bf16." >&2
    echo "  $(python3 -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'unknown GPU')" >&2
    echo "  Pilot arms must share one numeric regime with the Lightning L4 arms." >&2
    echo "  Use gpu=\"L4\"/\"A10\"/\"L40S\"/\"A100\"/\"H100\" in modal_app.py, or set" >&2
    echo "  PRECISION=16-mixed EXPLICITLY and record it as an amendment." >&2
    exit 2
  fi
fi

ARMS="${ARMS:-l17_new l18_new l19_new}"
TARGET_TRIALS="${TARGET_TRIALS:-30}"
SEED="${SEED:-0}"
# Sized against Modal's 24 h Function timeout (modal_app.py passes
# timeout - 30 min). Stopping cleanly matters: a clean return is what lets the
# driver commit the Volume before the container is torn down.
MAX_SECONDS="${MAX_SECONDS:-79200}"           # 22 h
DEADLINE=$(( $(date +%s) + MAX_SECONDS ))

mkdir -p "$RUNS_ROOT" "$STORE_DIR"

echo "=============================================================="
echo " modal loss pilot"
echo "   repo     = ${REPO_DIR}  (sha ${INSTAROAD_GIT_SHA:-unknown})"
echo "   dataset  = ${DATASET_DIR}"
echo "   runs     = ${RUNS_ROOT}"
echo "   store    = ${STORE_DIR}"
echo "   gpu      = $(python3 -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'none visible')"
echo "   arms     = ${ARMS}"
echo "   precision=${PRECISION}  batch=${BATCH_SIZES}  workers=${NUM_WORKERS}  seed=${SEED}"
echo "   budget   = ${MAX_SECONDS}s"
echo "=============================================================="

# budgeted <ENV=VAL...> <cmd...>: run a stage under the remaining wall clock.
# SIGINT first (Lightning writes last.ckpt and closes wandb), SIGKILL 120 s
# later. On timeout: exit 0, so the driver's Volume commit still happens and
# the next invocation resumes.
budgeted () {
  local remaining=$(( DEADLINE - $(date +%s) ))
  if [ "$remaining" -le 300 ]; then
    echo "== TIME BUDGET REACHED — exiting cleanly so the Volume commits. Re-run to resume. =="
    exit 0
  fi
  local rc=0
  timeout -k 120 --signal=INT "$remaining" env "$@" || rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 130 ] || [ "$rc" -eq 137 ]; then
    echo "== TIME BUDGET hit mid-stage (rc=$rc) — progress checkpointed; exiting cleanly. Re-run to resume. =="
    exit 0
  fi
  return "$rc"
}

# Arm -> loss tag. Kept identical to pilot_seq.sh / pilot_kaggle.sh; extend all
# three together when an arm is added.
declare -A TAG=( [l1_new]=bce [l2_new]=gap_ce [l3_new]=tl_ce [l9_new]=gap_tl_ce
                 [l10_new]=wbce [l11_new]=sdice [l12_new]=lcdice
                 [l15_new]=balance_ce [l16_new]=dice
                 [l4a_new]=t2_ce [l4b_new]=t4_ce
                 [l17_new]=gap_t2_ce [l18_new]=gap_t4_ce [l19_new]=gap_t2t4_ce
                 [l5_new]=pstar_dice [l13_new]=pstar_sdice [l14_new]=pstar_lcdice )

trials_done () {  # $1 = study.db  $2 = study name -> COMPLETE+PRUNED count
  python3 - "$1" "$2" <<'PY'
import sys
try:
    import optuna
    s = optuna.load_study(study_name=sys.argv[2], storage=f"sqlite:///{sys.argv[1]}")
    print(sum(t.state.name in ("COMPLETE", "PRUNED") for t in s.trials))
except Exception:
    print(0)
PY
}

# PROBE=1: time one 1-epoch trial and exit. Run this ONCE on a new GPU type
# before committing credit — it is the only honest way to turn the $30 into a
# number of arms. Costs a couple of cents.
if [ "${PROBE:-0}" = "1" ]; then
  arm=$(echo $ARMS | awk '{print $1}')
  echo "=== PROBE: one 1-epoch trial of ${arm} (${TAG[$arm]}) at TUNE_LENGTH patches ==="
  t0=$(date +%s)
  STAGE=tune N_TRIALS=1 TUNE_EPOCHS=1 PATIENCE=0 SEARCH_GPUS=1 \
    bash "$REPO_DIR/scripts/LightningStudio/loss/${arm}.sh"
  probe=$(( $(date +%s) - t0 ))
  echo "=== PROBE DONE in ${probe}s (1 trial x 1 epoch, 1 GPU) ==="
  echo "tune/arm ≈ TARGET_TRIALS x ~6 eff. epochs x ${probe}s / SEARCH_GPUS;"
  echo "fit/arm  ≈ 50 x ${probe}s x (8830/TUNE_LENGTH) — spatial arms cost extra CPU."
  echo "Multiply the total by the GPU+CPU+RAM rate in scripts/modal/README.md"
  echo "to convert hours into credit before launching the real thing."
  exit 0
fi

for arm in $ARMS; do
  tag="${TAG[$arm]:?unknown arm $arm — extend the TAG table (in all three ports)}"
  run_dir="${RUNS_ROOT}/sr_r0_new_${tag}_holdout_seed${SEED}"
  study="sr_r0_new_${tag}_holdout_seed${SEED}"
  script="$REPO_DIR/scripts/LightningStudio/loss/${arm}.sh"
  echo "########## ${arm} (${tag}) ##########"

  if [ ! -f "${run_dir}/best_params.yaml" ]; then
    done_n=$(trials_done "${run_dir}/study.db" "$study")
    rem=$(( TARGET_TRIALS - done_n )); [ "$rem" -lt 0 ] && rem=0
    echo "== TUNE: ${done_n}/${TARGET_TRIALS} trials done -> running ${rem} more =="
    budgeted STAGE=tune N_TRIALS="$rem" bash "$script"
  else
    echo "== TUNE: best_params.yaml exists — skipping =="
  fi

  if [ ! -f "${run_dir}/checkpoints/unet_s2rosa_jointsr_final.ckpt" ]; then
    echo "== FIT (resume if possible) =="
    budgeted STAGE=fit RESUME_FIT=1 bash "$script"
  else
    echo "== FIT: final ckpt exists — skipping =="
  fi

  if [ ! -f "${run_dir}/.bench_done" ]; then
    echo "== BENCH (val) =="
    budgeted STAGE=bench bash "$script"
    touch "${run_dir}/.bench_done"
  else
    echo "== BENCH: done — skipping =="
  fi
done

echo "########## ALL ARMS DONE ##########"
echo "Report: python -m benchmarking.cli report --store-dir ${STORE_DIR} \\"
echo "          --metric f1 --metric iou --metric apls --metric cldice"
