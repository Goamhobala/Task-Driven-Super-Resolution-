#!/bin/bash
# Run a LIST of per-arm refit scripts inside ONE SLURM job, N arms at a time.
#
# WHY ARMS IN PARALLEL AND NOT DDP ACROSS GPUS
# --------------------------------------------
# Two GPUs could be spent two ways. `REFIT_GPUS=2` puts Lightning in DDP, which
# multiplies the EFFECTIVE batch by the device count — 8 per GPU becomes 16.
# Batch is a protocol constant (amendment 2026-08-03a) and seed 0 trained at 8,
# so DDP would make the refit incomparable to the seed it is a re-draw of: the
# "seed variance" would absorb a batch-size change. Running one arm per GPU
# keeps every fit byte-identical in configuration to seed 0 and delivers the
# same throughput.
#
# Each GPU gets its own LANE: a sequential list of arms it works through. A lane
# is a plain subshell with CUDA_VISIBLE_DEVICES pinned, so an arm that dies
# takes only its own lane's current item with it — the rest of a 48 h job
# survives.
#
#   ARMS="gap_tl_ce gap_t4_ce ..." NGPU=2 bash run_pool.sh
#
# Every per-arm script is itself idempotent (finished fits and benched rows are
# skipped), so re-submitting after a timeout resumes rather than restarts.
set -uo pipefail          # NB not -e: a failing arm must not kill the pool
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
REFIT_DIR="$REPO_DIR/scripts/hpc/loss/refit"

: "${ARMS:?run_pool.sh needs ARMS (space-separated script stems)}"
NGPU="${NGPU:-1}"

read -r -a ARM_ARR <<< "$ARMS"
echo "=============================================================="
echo "refit pool   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  arms : ${#ARM_ARR[@]}   (${ARMS})"
echo "  gpus : ${NGPU}          seeds: ${SEEDS:-1 2}"
echo "=============================================================="

# Deal the arms round-robin across lanes so each lane gets a mix rather than a
# contiguous block — arms differ in cost and a contiguous split can leave one
# GPU idle for hours at the end.
for g in $(seq 0 $((NGPU - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="$g"
    lane=0
    for i in "${!ARM_ARR[@]}"; do
      [ $((i % NGPU)) -eq "$g" ] || continue
      arm="${ARM_ARR[$i]}"
      script="$REFIT_DIR/${arm}.sh"
      if [ ! -f "$script" ]; then
        echo "[gpu $g] MISSING $script — skipping" >&2
        continue
      fi
      lane=$((lane + 1))
      echo "[gpu $g] >>> ${arm}  ($(date -u +%H:%M:%SZ))"
      if bash "$script"; then
        echo "[gpu $g] <<< ${arm} OK"
      else
        echo "[gpu $g] <<< ${arm} FAILED (rc=$?) — continuing" >&2
      fi
    done
    echo "[gpu $g] lane finished: ${lane} arm(s)"
  ) &
done

wait
echo "=== pool done  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
