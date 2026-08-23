#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="train-pair"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.txt


# Pair dispatcher: run TWO independent single-GPU experiments in ONE job,
# one per GPU, each chaining STAGE=fit -> STAGE=bench. Exists because the
# cluster caps concurrent jobs (not GPUs): one pair job = two arms in flight
# per job slot, doubling throughput vs train.sbatch under the cap.
#
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/l1_all.sh --B=loss/la0_all.sh
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/l2_all.sh --B=loss/l3_all.sh
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/l2_all.sh --B=loss/l2_all.sh \
#       A.GAP_R=3 B.GAP_R=9                     # within-arm hp grid, one job
#   sbatch scripts/hpc/train_pair.sbatch --A=loss/la0_all.sh --B=loss/la0_all.sh \
#       A.SEED=1 B.SEED=2                       # anchor replicates for the band
#
# Config tokens:
#   --A= / --B=   inner scripts, resolved under scripts/hpc/ (like train.sbatch)
#   A.KEY=VALUE   env override for side A only (B.KEY=VALUE for side B)
#   A.KEY=v1,v2   SWEEP: the side runs its whole chain once per value,
#                 sequentially on its GPU (one swept key per side; walltime
#                 multiplies by the number of values — raise --time!), e.g.
#                 the entire Phase C grid in ONE job:
#                   sbatch --time=48:00:00 scripts/hpc/train_pair.sbatch \
#                     --A=loss/l7_all.sh --B=loss/l8_all.sh BSTAR=bce \
#                     A.CL_ALPHA=0.2,0.3,0.5 B.SR_W=0.5,1,2
#                 (also handy for seed replicates: A.SEED=1,2)
#   KEY=VALUE     shared override, exported to both sides
#
# Notes:
#   * Single-GPU stages only (the loss arms; unet/sr STAGE=tune fans Optuna
#     over both GPUs itself — keep using train.sbatch/train_both.sbatch there).
#   * Each side chains STAGE=tune -> fit -> bench. tune is an instant no-op
#     for most loss arms; for the Phase B compounds (l5/l6) it runs the
#     unet.tune_loss mix_w search pinned to that side's ONE GPU
#     (SEARCH_GPUS=1 is forced — the side can't fan out).
#   * Both sides share the walltime — pair arms of similar duration (the
#     weighted-CE arms gap_ce/tl_ce are slower than bce/bce_dice: per-step CPU
#     skeletonization), or the faster side's GPU idles until the slower ends.
#   * Sides run under `wait PID`, so one side failing does NOT kill the other;
#     the job exits non-zero if either failed. Concurrent STAGE=bench into a
#     shared STORE_DIR is safe (sharded, atomic per-run writes).
#   * Each side gets its own log (pair_<side>_<jobid>.txt next to slurm out);
#     _stages.sh additionally tees per-stage logs into the run dirs.
set -uo pipefail   # NB no -e: background sides report via `wait`, not ERR

REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

SCRIPT_A="" ; SCRIPT_B=""
A_ENV=() ; B_ENV=()
for arg in "$@"; do
  case "$arg" in
    --A=*|--a=*)   SCRIPT_A="${arg#*=}" ;;
    --B=*|--b=*)   SCRIPT_B="${arg#*=}" ;;
    A.*=*)         A_ENV+=("${arg#A.}") ;;
    B.*=*)         B_ENV+=("${arg#B.}") ;;
    STAGE=*)       echo "WARN: STAGE is driven by train_pair.sbatch; ignoring '$arg'." >&2 ;;
    *=*)           export "${arg?}" ;;
    *)             echo "WARN: ignoring positional arg '$arg' (pair sides take none)." >&2 ;;
  esac
done

resolve() {  # name under scripts/hpc/, or a literal path
  if [ -f "$REPO_DIR/scripts/hpc/$1" ]; then echo "$REPO_DIR/scripts/hpc/$1";
  elif [ -f "$1" ]; then echo "$1";
  else echo ""; fi
}
[ -n "$SCRIPT_A" ] && [ -n "$SCRIPT_B" ] || {
  echo "usage: sbatch train_pair.sbatch --A=<script> --B=<script> [A.K=V] [B.K=V] [K=V]" >&2; exit 2; }
PATH_A=$(resolve "$SCRIPT_A"); PATH_B=$(resolve "$SCRIPT_B")
[ -n "$PATH_A" ] || { echo "ERROR: --A script not found: $SCRIPT_A" >&2; exit 2; }
[ -n "$PATH_B" ] || { echo "ERROR: --B script not found: $SCRIPT_B" >&2; exit 2; }

# Split the CPU allocation so the two sides' OMP/torch thread pools don't
# oversubscribe each other (matters for the CPU-skeletonizing weighted-CE arms).
THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / 2 )); [ "$THREADS" -lt 1 ] && THREADS=1

JOB_ID="${SLURM_JOB_ID:-local}"
run_chain() {  # one tune->fit->bench pass: $1 script, $2 gpu, rest: env
  local script="$1" gpu="$2"; shift 2
  env ${1:+"$@"} CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      STAGE=tune SEARCH_GPUS=1 bash "$script" \
  && env ${1:+"$@"} CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      STAGE=fit bash "$script" \
  && env ${1:+"$@"} CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      STAGE=bench bash "$script"
}

run_side() {  # $1 side label, $2 script path, $3 gpu index, rest: side env
  local side="$1" script="$2" gpu="$3"; shift 3
  local envs=() sweep_key="" sweep_vals="" sweep_idx=-1 i rc=0
  envs=(${1:+"$@"})
  for i in "${!envs[@]}"; do
    case "${envs[$i]#*=}" in *,*)
      if [ -n "$sweep_key" ]; then
        echo "ERROR side ${side}: only ONE swept KEY=v1,v2 per side." >&2; return 2
      fi
      sweep_key="${envs[$i]%%=*}"; sweep_vals="${envs[$i]#*=}"; sweep_idx=$i ;;
    esac
  done
  local log="pair_${side}_${JOB_ID}.txt"
  {
    echo "### side ${side}: $(basename "$script") on GPU ${gpu}  env=[${envs[*]:-}] ###"
    if [ -z "$sweep_key" ]; then
      run_chain "$script" "$gpu" ${envs[@]+"${envs[@]}"} || rc=1
    else
      local vals=() v
      IFS=',' read -r -a vals <<< "$sweep_vals"
      echo "### side ${side}: sweeping ${sweep_key} over [${sweep_vals}] (${#vals[@]} sequential chains) ###"
      for v in "${vals[@]}"; do
        envs[$sweep_idx]="${sweep_key}=${v}"
        echo "### side ${side}: ---- ${sweep_key}=${v} ---- ###"
        # a failed point doesn't kill the rest of the sweep; RC records it
        run_chain "$script" "$gpu" "${envs[@]}" || { rc=1; echo "### side ${side}: ${sweep_key}=${v} FAILED ###"; }
      done
    fi
    exit "$rc"
  } > "$log" 2>&1
}

echo "### train_pair ${JOB_ID}: A=$(basename "$PATH_A") [${A_ENV[*]:-}]  B=$(basename "$PATH_B") [${B_ENV[*]:-}] ###"
run_side A "$PATH_A" 0 ${A_ENV[@]+"${A_ENV[@]}"} & PID_A=$!
run_side B "$PATH_B" 1 ${B_ENV[@]+"${B_ENV[@]}"} & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "### side A exit=${RC_A} (pair_A_${JOB_ID}.txt)  side B exit=${RC_B} (pair_B_${JOB_ID}.txt) ###"
tail -n 3 "pair_A_${JOB_ID}.txt" "pair_B_${JOB_ID}.txt" 2>/dev/null || true
[ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ] || exit 1
echo "### train_pair DONE: both sides fit + bench ###"
