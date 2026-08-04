#!/bin/bash
# Loss-pilot orchestrator for KAGGLE (2x T4, 12 h sessions, 30 GPU-h/week).
# Runs the assigned arms SEQUENTIALLY through tune -> fit -> bench, and is
# fully IDEMPOTENT: rerun this same script every session until it prints
# ALL ARMS DONE — it skips finished work, tops up a partially-run Optuna
# study to exactly TARGET_TRIALS, rescues a killed tune (writes the overlay
# from the existing study), and resumes a killed fit from last.ckpt.
#
# Setup, assuming the ROSADataset (with norm_stats.yaml and
# <split>/mask_new_2pt5/) is attached as a Kaggle Dataset. First session: run
# with PROBE=1 — times one 1-epoch trial and exits (extrapolate: tune ≈
# N_TRIALS x ~6 effective epochs x probe / 2 workers; fit ≈ 50 x probe x
# 8830/TUNE_LENGTH). T4 has NO bf16 -> PRECISION=16-mixed is pinned here.
#
# SCHEDULED ("Save & Run All") mode — three things interactive mode hides:
#   1. The container starts EMPTY. The notebook must clone the repo EVERY
#      run, before calling this script (private repo -> GitHub PAT stored in
#      Add-ons -> Secrets):
#        from kaggle_secrets import UserSecretsClient
#        s = UserSecretsClient()
#        tok, wb = s.get_secret("GITHUB_PAT"), s.get_secret("WANDB_API_KEY")
#        !git clone https://x-access-token:{tok}@github.com/<you>/InstaRoadPrototype.git \
#            /kaggle/working/InstaRoadPrototype
#        !WANDB_API_KEY={wb} DATASET_DIR=/kaggle/input/<slug>/ROSADataset \
#            bash /kaggle/working/InstaRoadPrototype/scripts/kaggle/pilot_kaggle.sh
#   2. Output is only COMMITTED if the notebook finishes CLEANLY — a run
#      killed at the 12 h cap saves NOTHING. MAX_SECONDS (default 10.5 h)
#      makes this script stop mid-pipeline and exit 0 before that; all
#      progress (study.db trials, last.ckpt) is checkpointed and resumes.
#   3. State does not carry between scheduled runs by itself: attach the
#      PREVIOUS version's Output as an input dataset (+ Add Input -> Your
#      Work -> the notebook). The restore block below copies its runs/ and
#      benchmarks back into /kaggle/working automatically.
set -euo pipefail
# Kaggle's `!bash ...` swallows exit codes (the notebook "succeeds" even if
# this script dies) — so shout on any abort instead of failing silently.
trap 'echo "!! pilot_kaggle.sh ABORTED at line $LINENO (exit $?)" >&2' ERR

export REPO_DIR="${REPO_DIR:-/kaggle/working/InstaRoadPrototype}"
export INSTAROAD_ROOT="${INSTAROAD_ROOT:-/kaggle/working}"
export DATASET_DIR="${DATASET_DIR:-/kaggle/input/rosa-new/ROSADataset}"
export RUNS_ROOT="${RUNS_ROOT:-/kaggle/working/runs}"
export STORE_DIR="${STORE_DIR:-/kaggle/working/benchmarks_loss_pilot}"
export PRECISION="${PRECISION:-16-mixed}"     # T4: no native bf16
export SEARCH_GPUS="${SEARCH_GPUS:-2}"        # one tuner per T4
export NUM_WORKERS="${NUM_WORKERS:-2}"        # Kaggle: 4 CPUs
export WANDB_MODE="${WANDB_MODE:-online}"
export BATCH_SIZES="${BATCH_SIZES:-8}"        # FIXED across arms (2026-08-03
                                              # amendment). VERIFY it fits the T4s
                                              # (16 GB, fp16, 512²) with one PROBE=1
                                              # step before scheduling — if it OOMs,
                                              # this arm belongs on Lightning, NOT
                                              # at a smaller batch.

ARMS="${ARMS:-l3_new l2_new l4a_new l4b_new}"         # wbce, tl_ce, gap_ce
TARGET_TRIALS="${TARGET_TRIALS:-30}"
SEED="${SEED:-0}"
MAX_SECONDS="${MAX_SECONDS:-67800}"           # 10.5 h: exit CLEANLY before the
                                              # 12 h cap so Kaggle commits output
DEADLINE=$(( $(date +%s) + MAX_SECONDS ))

# --- Scheduled-run state restore (prior output attached as input dataset) ----
if [ ! -d "$RUNS_ROOT" ] || [ -z "$(ls -A "$RUNS_ROOT" 2>/dev/null || true)" ]; then
  # glob probe, not `ls | head`: a failed ls under pipefail+set -e killed the
  # whole script inside the command substitution (the 33-second "success").
  prev=""
  for _d in /kaggle/input/*/runs; do
    [ -d "$_d" ] && { prev="$_d"; break; }
  done
  if [ -n "${prev:-}" ]; then
    echo "== RESTORE: copying previous session state from ${prev%/runs} =="
    mkdir -p "$RUNS_ROOT" && cp -r "$prev"/. "$RUNS_ROOT"/
    [ -d "${prev%/runs}/benchmarks_loss_pilot" ] && \
      cp -r "${prev%/runs}/benchmarks_loss_pilot" /kaggle/working/ || true
  fi
fi

# budgeted <ENV=VAL...> <cmd...>: run a stage under the remaining time budget.
# SIGINT first (Lightning checkpoints + closes wandb), SIGKILL 120 s later.
# On timeout: exit 0 so the scheduled run COMMITS its output.
budgeted () {
  local remaining=$(( DEADLINE - $(date +%s) ))
  if [ "$remaining" -le 300 ]; then
    echo "== TIME BUDGET REACHED — exiting cleanly so the output commits. Re-run to resume. =="
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

# --- deps (idempotent) --------------------------------------------------------
if [ ! -f /kaggle/working/.pilot_deps ]; then
  pip install -q lightning 'jsonargparse[signatures]>=4.27.7' \
      segmentation-models-pytorch optuna optuna-integration \
      rasterio scikit-image torchmetrics wandb pyyaml || true
  touch /kaggle/working/.pilot_deps
fi

declare -A TAG=( [l1_new]=bce [l2_new]=gap_ce [l3_new]=tl_ce [l9_new]=gap_tl_ce
                 [l10_new]=wbce [l11_new]=sdice [l12_new]=lcdice
                 [l15_new]=balance_ce [l16_new]=dice
                 [l4a_new]=t2_ce [l4b_new]=t4_ce
                 [l17_new]=gap_t2_ce [l18_new]=gap_t4_ce [l19_new]=gap_t2t4_ce )

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

if [ "${PROBE:-0}" = "1" ]; then
  arm=$(echo $ARMS | awk '{print $1}')
  echo "=== PROBE: one 1-epoch trial of ${arm} (${TAG[$arm]}) at TUNE_LENGTH patches ==="
  t0=$(date +%s)
  STAGE=tune N_TRIALS=1 TUNE_EPOCHS=1 PATIENCE=0 SEARCH_GPUS=1 \
    bash "$REPO_DIR/scripts/LightningStudio/loss/${arm}.sh"
  echo "=== PROBE DONE in $(( $(date +%s) - t0 ))s (1 trial x 1 epoch, 1 GPU) ==="
  echo "tune/arm ≈ TARGET_TRIALS x ~6 eff. epochs x this / SEARCH_GPUS;"
  echo "fit/arm  ≈ 50 x this x (8830/TUNE_LENGTH) — spatial arms cost extra CPU."
  exit 0
fi

for arm in $ARMS; do
  tag="${TAG[$arm]:?unknown arm $arm — extend the TAG table}"
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
