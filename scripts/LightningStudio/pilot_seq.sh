#!/bin/bash
# Loss-pilot orchestrator for LIGHTNING STUDIO (single H100 by default).
# Runs the assigned arms SEQUENTIALLY through tune -> fit -> bench.
# IDEMPOTENT: rerun after any interruption — it skips finished work, tops up
# a partial Optuna study to TARGET_TRIALS, rescues a killed tune, and resumes
# a killed fit from last.ckpt.
#
#   bash scripts/LightningStudio/pilot_seq.sh                     # dice arms
#   ARMS="l10_new l3_new" bash scripts/LightningStudio/pilot_seq.sh
#   PROBE=1 bash scripts/LightningStudio/pilot_seq.sh             # cost probe
#
# NB the region/dice arms DO tune — lr and batch at minimum. "Nothing to
# tune" would compare tuned CE arms against untuned dice arms (unfair), and
# lcDice's gradient scale (≈ Dice²/2 near 0) genuinely wants its own lr. The
# tune budget (N_TRIALS x TUNE_EPOCHS x TUNE_LENGTH) is the between-arm
# constant; dims are consumption-gated per arm in sr.tune.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

export NUM_WORKERS="${NUM_WORKERS:-4}"   # raster masks: forked loaders are safe

ARMS="${ARMS:-l16_new l11_new l12_new}"  # dice, sdice, lcdice
TARGET_TRIALS="${TARGET_TRIALS:-30}"
SEED="${SEED:-0}"
RUNS_ROOT="${RUNS_ROOT:-${INSTAROAD_ROOT}/runs}"

declare -A TAG=( [l1_new]=bce [l2_new]=gap_ce [l3_new]=tl_ce [l9_new]=gap_tl_ce
                 [l10_new]=wbce [l11_new]=sdice [l12_new]=lcdice
                 [l15_new]=balance_ce [l16_new]=dice )

trials_done () {
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
  echo "=== PROBE: one 1-epoch trial of ${arm} (${TAG[$arm]}) ==="
  t0=$(date +%s)
  STAGE=tune N_TRIALS=1 TUNE_EPOCHS=1 PATIENCE=0 SEARCH_GPUS=1 \
    bash "$LS_DIR/loss/${arm}.sh"
  echo "=== PROBE DONE in $(( $(date +%s) - t0 ))s ==="
  echo "tune/arm ≈ TARGET_TRIALS x ~6 eff. epochs x this;"
  echo "fit/arm  ≈ 50 x this x (8830/TUNE_LENGTH)."
  exit 0
fi

for arm in $ARMS; do
  tag="${TAG[$arm]:?unknown arm $arm — extend the TAG table}"
  run_dir="${RUNS_ROOT}/sr_r0_new_${tag}_holdout_seed${SEED}"
  study="sr_r0_new_${tag}_holdout_seed${SEED}"
  echo "########## ${arm} (${tag}) ##########"

  if [ ! -f "${run_dir}/best_params.yaml" ]; then
    done_n=$(trials_done "${run_dir}/study.db" "$study")
    rem=$(( TARGET_TRIALS - done_n )); [ "$rem" -lt 0 ] && rem=0
    echo "== TUNE: ${done_n}/${TARGET_TRIALS} trials done -> running ${rem} more =="
    STAGE=tune N_TRIALS="$rem" bash "$LS_DIR/loss/${arm}.sh"
  else
    echo "== TUNE: best_params.yaml exists — skipping =="
  fi

  if [ ! -f "${run_dir}/checkpoints/unet_s2rosa_jointsr_final.ckpt" ]; then
    echo "== FIT (resume if possible) =="
    STAGE=fit RESUME_FIT=1 bash "$LS_DIR/loss/${arm}.sh"
  else
    echo "== FIT: final ckpt exists — skipping =="
  fi

  if [ ! -f "${run_dir}/.bench_done" ]; then
    echo "== BENCH (val) =="
    STAGE=bench bash "$LS_DIR/loss/${arm}.sh"
    touch "${run_dir}/.bench_done"
  else
    echo "== BENCH: done — skipping =="
  fi
done

echo "########## ALL ARMS DONE ##########"
echo "Report: python -m benchmarking.cli report --store-dir ${STORE_DIR:-${INSTAROAD_ROOT}/benchmarks_loss_pilot} \\"
echo "          --metric f1 --metric iou --metric apls --metric cldice"
