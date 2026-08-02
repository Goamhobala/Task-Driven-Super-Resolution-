#!/bin/bash
# Loss-pilot orchestrator for KAGGLE (2x T4, 12 h sessions, 30 GPU-h/week).
# Runs the assigned arms SEQUENTIALLY through tune -> fit -> bench, and is
# fully IDEMPOTENT: rerun this same script every session until it prints
# ALL ARMS DONE — it skips finished work, tops up a partially-run Optuna
# study to exactly TARGET_TRIALS, rescues a killed tune (writes the overlay
# from the existing study), and resumes a killed fit from last.ckpt.
#
# Setup (one notebook cell), assuming the ROSADataset (with norm_stats.yaml
# and <split>/mask_new_2pt5/) is attached as a Kaggle Dataset:
#
#   !git clone <your-repo-url> /kaggle/working/InstaRoadPrototype
#   !DATASET_DIR=/kaggle/input/<your-dataset>/ROSADataset \
#     bash /kaggle/working/InstaRoadPrototype/scripts/kaggle/pilot_kaggle.sh
#
# First session: run with PROBE=1 — times one 1-epoch trial and exits, so you
# can extrapolate the real cost before committing quota (multiply: tune ≈
# N_TRIALS x ~6 effective epochs x probe time / 2 workers; fit ≈ 50 x probe
# time x [full/TUNE_LENGTH ratio]).
#
# Session bookkeeping: /kaggle/working persists as the notebook's output —
# runs/, benchmarks and study.db all live there and carry across sessions if
# you re-attach the previous output (or just keep the same notebook session
# chain). T4 has NO bf16 -> PRECISION=16-mixed is pinned here.
set -euo pipefail

export REPO_DIR="${REPO_DIR:-/kaggle/working/InstaRoadPrototype}"
export INSTAROAD_ROOT="${INSTAROAD_ROOT:-/kaggle/working}"
export DATASET_DIR="${DATASET_DIR:-/kaggle/input/rosa-new/ROSADataset}"
export RUNS_ROOT="${RUNS_ROOT:-/kaggle/working/runs}"
export STORE_DIR="${STORE_DIR:-/kaggle/working/benchmarks_loss_pilot}"
export PRECISION="${PRECISION:-16-mixed}"     # T4: no native bf16
export SEARCH_GPUS="${SEARCH_GPUS:-2}"        # one tuner per T4
export NUM_WORKERS="${NUM_WORKERS:-2}"        # Kaggle: 4 CPUs
export WANDB_MODE="${WANDB_MODE:-online}"
export BATCH_SIZES="${BATCH_SIZES:-2 4}"      # 512² fp16 on 16 GB: 8 mostly OOMs

ARMS="${ARMS:-l10_new l3_new l2_new}"         # wbce, tl_ce, gap_ce
TARGET_TRIALS="${TARGET_TRIALS:-30}"
SEED="${SEED:-0}"

# --- deps (idempotent) --------------------------------------------------------
if [ ! -f /kaggle/working/.pilot_deps ]; then
  pip install -q lightning segmentation-models-pytorch optuna optuna-integration \
      rasterio scikit-image torchmetrics wandb pyyaml || true
  touch /kaggle/working/.pilot_deps
fi

declare -A TAG=( [l1_new]=bce [l2_new]=gap_ce [l3_new]=tl_ce [l9_new]=gap_tl_ce
                 [l10_new]=wbce [l11_new]=sdice [l12_new]=lcdice
                 [l15_new]=balance_ce [l16_new]=dice )

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
    STAGE=tune N_TRIALS="$rem" bash "$script"
  else
    echo "== TUNE: best_params.yaml exists — skipping =="
  fi

  if [ ! -f "${run_dir}/checkpoints/unet_s2rosa_jointsr_final.ckpt" ]; then
    echo "== FIT (resume if possible) =="
    STAGE=fit RESUME_FIT=1 bash "$script"
  else
    echo "== FIT: final ckpt exists — skipping =="
  fi

  if [ ! -f "${run_dir}/.bench_done" ]; then
    echo "== BENCH (val) =="
    STAGE=bench bash "$script"
    touch "${run_dir}/.bench_done"
  else
    echo "== BENCH: done — skipping =="
  fi
done

echo "########## ALL ARMS DONE ##########"
echo "Report: python -m benchmarking.cli report --store-dir ${STORE_DIR} \\"
echo "          --metric f1 --metric iou --metric apls --metric cldice"
