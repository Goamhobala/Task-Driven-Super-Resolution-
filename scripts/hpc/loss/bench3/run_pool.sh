#!/bin/bash
# Bench the missing THIRD test seed for a list of arms, N at a time.
#
# WHY LANES ARE NOT TIED TO GPUS
# ------------------------------
# A refit is GPU-bound, so the refit pools ran one arm per GPU. A BENCH is not:
# ~86% of it is serial Python (APLS and clDice have no thread pool, and
# rasterio reads on the main thread), so one bench cannot use much more than one
# core and the GPU sits mostly idle between forward passes. Lanes here are
# therefore sized by CPU COUNT, not GPU count, and every lane shares the one
# GPU -- nine inference models are a few hundred MB each, so VRAM is not the
# constraint.
#
#   ARMS="dice:0 sdice:0 ..." NLANE=8 bash run_pool.sh
#
# Each entry is <arm>:<seed>. Idempotent: an arm already present in the store
# for that (model, seed, test) is skipped, because the store is append-only
# with uuid run_ids and has no dedupe -- benching twice would make every mean
# average those chips twice.
set -uo pipefail          # NB not -e: one failing arm must not kill the pool
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
USER_NAME="${USER:-$(whoami)}"
HERE="$REPO_DIR/scripts/hpc/loss/bench3"
export STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_test3}"

: "${ARMS:?run_pool.sh needs ARMS (space-separated <arm>:<seed>)}"
NLANE="${NLANE:-8}"

# arm -> EXP_TAG|LOSS_ARM|MODEL_NAME. Taken verbatim from the refit scripts so
# RUN_DIR resolves to the same directory those wrote into.
meta () {
  case "$1" in
    dice)               echo "r0_new|dice|sr_r0_new_dice_holdout" ;;
    sdice)              echo "r0_new|sdice|sr_r0_new_sdice_holdout" ;;
    lcdice)             echo "r0_new|lcdice|sr_r0_new_lcdice_holdout" ;;
    wbce)               echo "r0_new|wbce|sr_r0_new_wbce_holdout" ;;
    t2_ce)              echo "r0_new|t2_ce|sr_r0_new_t2_ce_holdout" ;;
    t4_ce)              echo "r0_new|t4_ce|sr_r0_new_t4_ce_holdout" ;;
    tl_ce)              echo "r0_new|tl_ce|sr_r0_new_tl_ce_holdout" ;;
    gap_ce)             echo "r0_new|gap_ce|sr_r0_new_gap_ce_holdout" ;;
    gap_t2_ce)          echo "r0_new|gap_t2_ce|sr_r0_new_gap_t2_ce_holdout" ;;
    gap_t4_ce)          echo "r0_new|gap_t4_ce|sr_r0_new_gap_t4_ce_holdout" ;;
    gap_t2t4_ce)        echo "r0_new|gap_t2t4_ce|sr_r0_new_gap_t2t4_ce_holdout" ;;
    gap_tl_ce)          echo "r0_new|gap_tl_ce|sr_r0_new_gap_tl_ce_holdout" ;;
    wbce_dice)          echo "r0_new_wbce|pstar_dice|sr_r0_new_wbce_dice_holdout" ;;
    gap_t2_ce_dice)     echo "r0_new_gapt2|pstar_dice|sr_r0_new_gap_t2_ce_dice_holdout" ;;
    gap_t2_ce_sdice)    echo "r0_new_gapt2|pstar_sdice|sr_r0_new_gap_t2_ce_sdice_holdout" ;;
    gap_t2_ce_lcdice)   echo "r0_new_gapt2|pstar_lcdice|sr_r0_new_gap_t2_ce_lcdice_holdout" ;;
    gap_tl_dice)        echo "r0_new_gaptl|pstar_dice|sr_r0_new_gap_tl_dice_holdout" ;;
    gap_tl_ce_sdice)    echo "r0_new_gaptl|pstar_sdice|sr_r0_new_gap_tl_ce_sdice_holdout" ;;
    gap_tl_ce_lcdice)   echo "r0_new_gaptl|pstar_lcdice|sr_r0_new_gap_tl_ce_lcdice_holdout" ;;
    gapt4_pstar_dice)   echo "r0_new_gapt4|pstar_dice|sr_r0_new_gapt4_pstar_dice_holdout" ;;
    gapt4_pstar_sdice)  echo "r0_new_gapt4|pstar_sdice|sr_r0_new_gapt4_pstar_sdice_holdout" ;;
    gapt4_pstar_lcdice) echo "r0_new_gapt4|pstar_lcdice|sr_r0_new_gapt4_pstar_lcdice_holdout" ;;
    *) echo "" ;;
  esac
}

in_store () {   # model, seed -> 0 if a test row already exists
  python - "$STORE_DIR" "$1" "$2" <<'PYEOF' 2>/dev/null
import sys
from pathlib import Path
store, name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    from benchmarking.store import load_runs
    runs = load_runs(Path(store))
except Exception:
    sys.exit(1)
if runs is None or getattr(runs, "empty", True) or "model_name" not in runs.columns:
    sys.exit(1)
hit = runs[(runs["model_name"] == name) & (runs["seed"] == seed)]
if "dataset_split" in runs.columns:
    hit = hit[hit["dataset_split"] == "test"]
sys.exit(0 if len(hit) else 1)
PYEOF
}

TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-$(( ${SLURM_CPUS_PER_TASK:-8} * ${SLURM_NTASKS:-1} ))}"
[ "$NLANE" -gt "$TOTAL_CPUS" ] && NLANE="$TOTAL_CPUS"
read -r -a ARR <<< "$ARMS"
echo "=============================================================="
echo "test-seed3 bench pool   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  entries : ${#ARR[@]}   lanes: ${NLANE}   cpus: ${TOTAL_CPUS}"
echo "  store   : ${STORE_DIR}"
echo "=============================================================="

for g in $(seq 0 $((NLANE - 1))); do
  (
    for i in "${!ARR[@]}"; do
      [ $((i % NLANE)) -eq "$g" ] || continue
      entry="${ARR[$i]}"; arm="${entry%%:*}"; seed="${entry##*:}"
      m=$(meta "$arm")
      if [ -z "$m" ]; then echo "[lane $g] UNKNOWN arm '$arm' — skipping" >&2; continue; fi
      IFS='|' read -r exp loss model <<< "$m"
      if in_store "$model" "$seed"; then
        echo "[lane $g] skip ${arm} seed${seed} (already in store)"; continue
      fi
      echo "[lane $g] >>> ${arm} seed${seed}  ($(date -u +%H:%M:%SZ))"
      if env MODEL_NAME="$model" EXP_TAG="$exp" LOSS_ARM="$loss" SEED="$seed" \
             STORE_DIR="$STORE_DIR" NUM_WORKERS=1 \
             bash "$HERE/_bench_one.sh"; then
        echo "[lane $g] <<< ${arm} seed${seed} OK"
      else
        echo "[lane $g] <<< ${arm} seed${seed} FAILED (rc=$?)" >&2
      fi
    done
  ) &
done
wait
echo "=== pool done  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
