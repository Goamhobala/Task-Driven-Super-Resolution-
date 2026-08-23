#!/bin/bash
# =============================================================================
# Run TWO experiment arms (or one arm swept over a hyperparameter) from a single
# command, each chaining STAGE=tune -> fit -> bench. Port of train_pair.sbatch.
#
# On the cluster this ran the two sides concurrently, one per GPU, to beat the
# job cap. On Lightning there is no job cap, so it AUTO-ADAPTS:
#   * 2+ GPUs visible -> sides run in PARALLEL, pinned to GPU 0 and GPU 1.
#   * 1 GPU (free tier) -> side A runs to completion, then side B (SEQUENTIAL).
#
#   bash scripts/LightningStudio/run_pair.sh --A=<script> --B=<script> \
#        [A.KEY=VALUE ...] [B.KEY=VALUE ...] [KEY=VALUE ...]
#
# Examples:
#   # two different arms
#   bash scripts/LightningStudio/run_pair.sh --A=loss/l1_all.sh --B=loss/la0_all.sh
#   # seed replicates of one anchor arm
#   bash scripts/LightningStudio/run_pair.sh --A=loss/la0_all.sh --B=loss/la0_all.sh A.SEED=1 B.SEED=2
#   # a within-side sweep: side A runs its whole chain once per value, in sequence
#   bash scripts/LightningStudio/run_pair.sh --A=loss/l7_all.sh --B=loss/l8_all.sh \
#        BSTAR=bce A.CL_ALPHA=0.2,0.3,0.5 B.SR_W=0.5,1,2
#
# Config tokens:
#   --A= / --B=   inner scripts, resolved under LightningStudio/ (like run.sh)
#   A.KEY=VALUE   env override for side A only (B.KEY=VALUE for side B)
#   A.KEY=v1,v2   SWEEP: that side runs its whole chain once per value, in
#                 sequence (one swept key per side)
#   KEY=VALUE     shared override, exported to both sides
#
# Each side writes its own log: pair_<side>_<timestamp>.txt in the current dir
# (the engines additionally tee per-stage logs into the run dirs).
# =============================================================================
set -uo pipefail   # NB no -e: background sides report via `wait`, not ERR
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

SCRIPT_A="" ; SCRIPT_B=""
A_ENV=() ; B_ENV=()
for arg in "$@"; do
  case "$arg" in
    --A=*|--a=*)   SCRIPT_A="${arg#*=}" ;;
    --B=*|--b=*)   SCRIPT_B="${arg#*=}" ;;
    A.*=*)         A_ENV+=("${arg#A.}") ;;
    B.*=*)         B_ENV+=("${arg#B.}") ;;
    STAGE=*)       echo "WARN: STAGE is driven by run_pair.sh; ignoring '$arg'." >&2 ;;
    *=*)           export "${arg?}" ;;
    *)             echo "WARN: ignoring positional arg '$arg' (pair sides take none)." >&2 ;;
  esac
done

resolve() {  # name under LightningStudio/, or a literal path
  if [ -f "$LS_DIR/$1" ]; then echo "$LS_DIR/$1";
  elif [ -f "$1" ]; then echo "$1";
  else echo ""; fi
}
[ -n "$SCRIPT_A" ] && [ -n "$SCRIPT_B" ] || {
  echo "usage: bash scripts/LightningStudio/run_pair.sh --A=<script> --B=<script> [A.K=V] [B.K=V] [K=V]" >&2; exit 2; }
PATH_A=$(resolve "$SCRIPT_A"); PATH_B=$(resolve "$SCRIPT_B")
[ -n "$PATH_A" ] || { echo "ERROR: --A script not found: $SCRIPT_A" >&2; exit 2; }
[ -n "$PATH_B" ] || { echo "ERROR: --B script not found: $SCRIPT_B" >&2; exit 2; }

# How many GPUs can we see? Decides parallel-vs-sequential and thread splitting.
N_GPUS=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
NCPU=$(nproc 2>/dev/null || echo 4)
if [ "$N_GPUS" -ge 2 ]; then
  PARALLEL=1; THREADS=$(( NCPU / 2 ))
else
  PARALLEL=0; THREADS="$NCPU"
fi
[ "$THREADS" -lt 1 ] && THREADS=1

STAMP="$(date +%Y%m%d_%H%M%S)"
run_chain() {  # one tune->fit->bench pass: $1 script, $2 gpu ("" = unpinned), rest: env
  local script="$1" gpu="$2"; shift 2
  local pin=(); [ -n "$gpu" ] && pin=(CUDA_VISIBLE_DEVICES="$gpu")
  env ${1:+"$@"} ${pin[@]+"${pin[@]}"} \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      STAGE=tune SEARCH_GPUS=1 bash "$script" \
  && env ${1:+"$@"} ${pin[@]+"${pin[@]}"} \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      STAGE=fit bash "$script" \
  && env ${1:+"$@"} ${pin[@]+"${pin[@]}"} \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      STAGE=bench bash "$script"
}

run_side() {  # $1 side label, $2 script path, $3 gpu ("" = unpinned), rest: side env
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
  local log="pair_${side}_${STAMP}.txt"
  {
    echo "### side ${side}: $(basename "$script") on GPU ${gpu:-<shared>}  env=[${envs[*]:-}] ###"
    if [ -z "$sweep_key" ]; then
      run_chain "$script" "$gpu" ${envs[@]+"${envs[@]}"} || rc=1
    else
      local vals=() v
      IFS=',' read -r -a vals <<< "$sweep_vals"
      echo "### side ${side}: sweeping ${sweep_key} over [${sweep_vals}] (${#vals[@]} sequential chains) ###"
      for v in "${vals[@]}"; do
        envs[$sweep_idx]="${sweep_key}=${v}"
        echo "### side ${side}: ---- ${sweep_key}=${v} ---- ###"
        run_chain "$script" "$gpu" "${envs[@]}" || { rc=1; echo "### side ${side}: ${sweep_key}=${v} FAILED ###"; }
      done
    fi
    exit "$rc"
  } > "$log" 2>&1
}

echo "### run_pair ${STAMP}: A=$(basename "$PATH_A") [${A_ENV[*]:-}]  B=$(basename "$PATH_B") [${B_ENV[*]:-}] ###"
if [ "$PARALLEL" = 1 ]; then
  echo "### ${N_GPUS} GPUs visible -> sides run in PARALLEL (A=gpu0, B=gpu1) ###"
  run_side A "$PATH_A" 0 ${A_ENV[@]+"${A_ENV[@]}"} & PID_A=$!
  run_side B "$PATH_B" 1 ${B_ENV[@]+"${B_ENV[@]}"} & PID_B=$!
  wait "$PID_A"; RC_A=$?
  wait "$PID_B"; RC_B=$?
else
  echo "### ${N_GPUS} GPU(s) visible -> sides run SEQUENTIALLY (A, then B) on the shared GPU ###"
  run_side A "$PATH_A" "" ${A_ENV[@]+"${A_ENV[@]}"}; RC_A=$?
  run_side B "$PATH_B" "" ${B_ENV[@]+"${B_ENV[@]}"}; RC_B=$?
fi

echo "### side A exit=${RC_A} (pair_A_${STAMP}.txt)  side B exit=${RC_B} (pair_B_${STAMP}.txt) ###"
tail -n 3 "pair_A_${STAMP}.txt" "pair_B_${STAMP}.txt" 2>/dev/null || true
[ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ] || exit 1
echo "### run_pair DONE: both sides fit + bench ###"
