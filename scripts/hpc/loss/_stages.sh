#!/bin/bash
# Shared staging engine for the loss-ablation experiment scripts. NOT submitted
# directly — each arm script (l1_all.sh, l2_all.sh, ...) sets its config
# (EXP_TAG, ARM, and any loss hyperparameters) and sources this file.
#
# Experiment (arm) scripts must set:
#   EXP_TAG   arm-table id, e.g. l2_gap_ce (drives run dir + benchmark model_name)
#   ARM       the loss arm string passed to unet.train_ablation / build_loss:
#             bce | gap_ce | tl_ce | bce_dice | pstar_dice | pstar_tversky |
#             focal_tversky | <base>+cldice | <base>+skelrec
# and may override any loss hyperparameter (GAP_R, TL_ELL, PSTAR, CL_ALPHA, ...).
#
# The protocol fixes the screening LR (1e-3, valid across arms thanks to the
# §4.4 scale normalization) and screens the loss + its hyperparameters, not
# the optimiser. Stages:
#   STAGE=tune    Phase B ONLY (pstar_dice / pstar_tversky): Optuna search of
#                 the P*<->region mixing ratio mix_w (amendment 2026-07-21 —
#                 the compound's one genuinely free parameter). Short trials
#                 (TUNE_EPOCHS, fixed train seed, fixed LR), fanned across
#                 SEARCH_GPUS workers; writes best_loss_params.yaml, which
#                 STAGE=fit consumes when MIX_W is unset. Every other arm:
#                 no-op (nothing to search; chain compat with train_both).
#                 Phase C compounds are refused by unet.tune_loss (the §4.5
#                 warmup means short trials never see the skeleton term).
#   STAGE=fit     Train one arm at a fixed budget (unet.train_ablation): no early
#                 stopping, checkpoint on val F1 @ 0.5, closing val threshold
#                 sweep. Submit with --gres=gpu:1. (`train` is an alias.)
#   STAGE=bench   Score the fitted checkpoint into the loss benchmark store at
#                 the val-tuned θ* (--threshold), on the val split — protocol
#                 decisions are made on val; test stays held out. --gres=gpu:1.
#
# Replication contract: everything an arm needs lives in its script + this
# engine; the only knobs meant to vary at submit time are SEED, STAGE, and (for
# a within-arm hyperparameter grid) the relevant hp env var, e.g.
#   sbatch scripts/hpc/train.sbatch --SCRIPT=loss/l2_all.sh STAGE=fit GAP_R=3
#   sbatch scripts/hpc/train.sbatch --SCRIPT=loss/l2_all.sh STAGE=fit GAP_R=9
# each lands as a distinct model_name (…_r3, …_r9) so the report can compare them.
set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"

: "${EXP_TAG:?arm script must set EXP_TAG (e.g. l2_gap_ce)}"
: "${ARM:?arm script must set ARM (e.g. gap_ce, bce_dice, bce_dice+cldice)}"

STAGE="${STAGE:-fit}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"    # 0 = main process; GDAL/rasterio segfault in forks
PRECISION="${PRECISION:-bf16-mixed}"

# --- Data --------------------------------------------------------------------
# ROSA_all = the big "all" dataset (neighbour to ROSA_Dense_CDNGI). MASK_DIRNAME
# empty = the split CSVs' own masks_raster.
DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_all}"
MASK_DIRNAME="${MASK_DIRNAME:-}"
LABEL_SOURCE="${LABEL_SOURCE:-all}"   # comparability tag; uniform across arms

# --- Train budget (protocol Appendix B; fixed across arms) -------------------
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-}"          # empty = the base config's batch_size
ENCODER="${ENCODER:-}"                # empty = base config (resnet34, protocol-fixed)
AUGMENT="${AUGMENT:-true}"            # protocol-fixed D4 flip on the train crops

# --- Loss hyperparameters (protocol defaults; arm scripts/env override) ------
PSTAR="${PSTAR:-bce}"                 # pixel slot for pstar_* arms (set to P*)
POS_WEIGHT="${POS_WEIGHT:-5}"         # wbce road-class weight (legacy 5; ~30-50 = class balance)
MIX_W="${MIX_W:-}"                    # pstar_* mixing ratio; empty = tuned value
                                      # from best_loss_params.yaml (else 0.5)

# --- Phase B mix_w search (STAGE=tune; pstar arms only) ----------------------
N_TRIALS="${N_TRIALS:-30}"            # pre-registered budget (amendment 2026-07-21)
TUNE_EPOCHS="${TUNE_EPOCHS:-8}"       # short proxy; fixed LR; fixed train seed
SEARCH_GPUS="${SEARCH_GPUS:-2}"       # parallel workers sharing the sqlite study
MIX_MIN="${MIX_MIN:-0.2}"
MIX_MAX="${MIX_MAX:-0.8}"
SEARCH_TVERSKY="${SEARCH_TVERSKY:-false}"  # pstar_tversky: search alpha jointly (2D)
GAP_R="${GAP_R:-5}"                   # GapLoss buffer radius (Appendix B centre)
GAP_K="${GAP_K:-60.0}"               # GapLoss K (paper)
TL_ELL="${TL_ELL:-5}"                 # TL/T2/T4 filter length (paper centre; grid {3,5,7})
TL_THETA="${TL_THETA:-0.375}"         # TL/T2/T4/gap_tl binarization θ (papers + Appendix B:
                                      # 0.375; grid {0.375,0.5}; legacy l3 ran 0.5 pre-knob)
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
CL_ALPHA="${CL_ALPHA:-0.3}"
CL_ITERS="${CL_ITERS:-5}"
SR_W="${SR_W:-1.0}"
SR_RADIUS="${SR_RADIUS:-1}"
WARMUP_START="${WARMUP_START:-30}"    # from-scratch: ramp skeleton weight 30->40 of 100
WARMUP_RAMP="${WARMUP_RAMP:-10}"

# --- Bench -------------------------------------------------------------------
STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_loss}"  # loss-dedicated
BENCH_SPLIT="${BENCH_SPLIT:-val}"     # decisions on val; test held out
TILE_METRICS="${TILE_METRICS:-apls}"  # comma-separated tile-metric plugins
                                      # (benchmarking.tile_metrics); '' disables.
                                      # apls = the protocol's connectivity metric,
                                      # required for Decision A's composite.

WANDB_PROJECT="${WANDB_PROJECT:-instaroad-loss-ablation}"
WANDB_MODE="${WANDB_MODE:-online}"
# =============================================================================

BASE_CONFIG="$REPO_DIR/src/unet/configs/unet.yaml"
NORM_CONFIG="${NORM_CONFIG:-$REPO_DIR/src/unet/configs/norm_stats.yaml}"

# A short hp slug so a within-arm grid (r/ℓ/pstar/…) lands as distinct runs and
# distinct benchmark model_names instead of colliding under one EXP_TAG.
slug_for() {  # hp slug for one pixel-slot(-ish) name: $1 = ARM head or PSTAR
  case "$1" in
    *gap_tl_ce*)             echo "_r${GAP_R}_l${TL_ELL}_th${TL_THETA}" ;;
    *gap_ce*)                echo "_r${GAP_R}" ;;
    *tl_ce*|*t2_ce*|*t4_ce*) echo "_l${TL_ELL}_th${TL_THETA}" ;;
    *wbce*)                  echo "_w${POS_WEIGHT}" ;;
    *)                       echo "" ;;
  esac
}
case "$ARM" in
  pstar_dice*|pstar_tversky*)
    HP_SLUG="_p${PSTAR}$(slug_for "$PSTAR")"
    # explicit MIX_W -> part of the identity; tuned mix_w is an attribute
    # (recorded in config/train_meta/wandb), NOT the name — RUN_DIR must be
    # stable across the tune->fit->bench chain.
    [ -n "${MIX_W}" ] && HP_SLUG="${HP_SLUG}_mw${MIX_W}" ;;
  *)                          HP_SLUG="$(slug_for "$ARM")" ;;
esac
case "$ARM" in
  *+cldice)  HP_SLUG="${HP_SLUG}_ca${CL_ALPHA}" ;;
  *+skelrec) HP_SLUG="${HP_SLUG}_sw${SR_W}" ;;
esac

FULL_TAG="${EXP_TAG}${HP_SLUG}"
RUN_DIR="/scratch/${USER_NAME}/InstaRoad/runs/loss_${FULL_TAG}_seed${SEED}"
MODEL_NAME="${MODEL_NAME:-${FULL_TAG}}"   # what the stats pair/group on (seed = a column)
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"
echo "host=$(hostname)  exp=loss/${EXP_TAG}  arm=${ARM}  stage=${STAGE}  seed=${SEED}"
echo "full_tag=${FULL_TAG}  model_name=${MODEL_NAME}"
echo "DATASET_DIR=${DATASET_DIR}  mask_dirname=${MASK_DIRNAME:-<masks_raster>}"

# ============================== STAGE: tune ==================================
# Real search ONLY for the Phase B compounds (below, after the venv/data
# checks); every other arm keeps the no-op so train_both's chain runs.
if [ "$STAGE" = "tune" ]; then
  case "$ARM" in
    pstar_dice|pstar_tversky) : ;;   # falls through to the search below
    *)
      echo "=== TUNE: no-op (arm '${ARM}' has nothing to search; mix_w applies to pstar_* only) ==="
      echo "Next: STAGE=fit trains the arm at the fixed screening LR (${LR})."
      exit 0 ;;
  esac
fi

# --- Fail fast (shared by fit + bench) ---------------------------------------
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  exit 1
fi
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: ${NORM_CONFIG} missing. The frozen norm stats MUST match ${DATASET_DIR}." >&2
  echo "  Generate: python -m sentinel2data.cli norm-stats --dataset-dir ${DATASET_DIR} \\" >&2
  echo "            --out ${NORM_CONFIG}" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
echo "python=$(which python)"

MASK_ARGS=()
[ -n "${MASK_DIRNAME}" ] && MASK_ARGS=(--mask-dirname "${MASK_DIRNAME}")

# ============================== STAGE: tune (search) =========================
if [ "$STAGE" = "tune" ]; then
  STORAGE="${STORAGE:-sqlite:///${RUN_DIR}/study.db}"
  STUDY_NAME="loss_mix_${FULL_TAG}_seed${SEED}"
  TV_ARGS=()
  [ "${SEARCH_TVERSKY}" = "true" ] && [ "$ARM" = "pstar_tversky" ] && \
    TV_ARGS=(--search-tversky-alpha)

  run_tuner () {   # $1=gpu id (empty = no pin)  $2=n-trials  $3=sampler seed
    local gpu="$1" ntrials="$2" sseed="$3" pin=""
    [ -n "$gpu" ] && pin="CUDA_VISIBLE_DEVICES=$gpu"
    env $pin python -m unet.tune_loss \
      --base-config "$BASE_CONFIG" \
      --base-config "$NORM_CONFIG" \
      --dataset-dir "$DATASET_DIR" \
      ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} \
      --out "$RUN_DIR" \
      --arm "$ARM" --pstar "$PSTAR" \
      --pos-weight "$POS_WEIGHT" \
      --gap-r "$GAP_R" --gap-k "$GAP_K" \
      --tl-ell "$TL_ELL" --tl-theta "$TL_THETA" \
      --tversky-alpha "$TVERSKY_ALPHA" \
      --mix-min "$MIX_MIN" --mix-max "$MIX_MAX" \
      ${TV_ARGS[@]+"${TV_ARGS[@]}"} \
      --n-trials "$ntrials" --max-epochs "$TUNE_EPOCHS" \
      --lr "$LR" --num-workers "$NUM_WORKERS" --precision "$PRECISION" \
      --devices 1 \
      --seed "$sseed" --train-seed "$SEED" \
      --study-name "$STUDY_NAME" --storage "$STORAGE"
  }

  echo "=== TUNE mix_w [${MIX_MIN},${MIX_MAX}] (${N_TRIALS} trials x ${TUNE_EPOCHS}ep across ${SEARCH_GPUS} GPU(s); fixed LR=${LR}, train seed=${SEED}) ==="
  if [ "$SEARCH_GPUS" -le 1 ]; then
    run_tuner "" "$N_TRIALS" "$(( SEED * 1000 ))"
  else
    PER_WORKER=$(( (N_TRIALS + SEARCH_GPUS - 1) / SEARCH_GPUS ))
    echo "  fanning out ${SEARCH_GPUS} workers x ${PER_WORKER} trials each"
    PIDS=()
    for g in $(seq 0 $(( SEARCH_GPUS - 1 ))); do
      run_tuner "$g" "$PER_WORKER" "$(( SEED * 1000 + g ))" &
      PIDS+=($!)
    done
    RC=0
    for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
    [ "$RC" -eq 0 ] || { echo "ERROR: a tuner worker failed." >&2; exit 1; }
  fi
  echo "=== TUNE DONE ===  $(grep -m1 mix_w "${RUN_DIR}/best_loss_params.yaml" || true)"
  echo "Next: STAGE=fit refits the tuned mix_w at the full budget (fit reads best_loss_params.yaml)."
  exit 0
fi

# ============================== STAGE: bench =================================
# Score the fitted checkpoint into the loss benchmark store at the val-tuned θ*.
# Every row flows through benchmarking.confusion_matrix; the sharded store is
# safe under concurrent SLURM jobs, so all arms can bench into it in parallel.
if [ "$STAGE" = "bench" ]; then
  META="${RUN_DIR}/train_meta.json"
  if [ ! -f "$META" ]; then
    echo "ERROR: ${META} missing — run STAGE=fit first (it writes the checkpoint + θ*)." >&2
    exit 1
  fi
  CKPT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint'])" "$META")
  THETA=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['best_threshold'])" "$META")
  if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint ${CKPT} (from train_meta.json) not found." >&2
    exit 1
  fi

  CONFIG_ARGS=()
  [ -f "${RUN_DIR}/config.yaml" ] && CONFIG_ARGS=(--config-yaml "${RUN_DIR}/config.yaml")

  METRIC_ARGS=()
  if [ -n "${TILE_METRICS}" ]; then
    IFS=',' read -r -a _TMS <<< "${TILE_METRICS}"
    for _tm in "${_TMS[@]}"; do METRIC_ARGS+=(--tile-metric "${_tm}"); done
  fi

  # Push the bench metrics (incl. val APLS) onto the FIT stage's wandb run:
  # train_meta.json carries the wandb run id; eval resumes it and updates the
  # summary with bench_${BENCH_SPLIT}/* columns.
  WANDB_ARGS=()
  if [ "${WANDB_MODE}" != "disabled" ]; then
    export WANDB_MODE WANDB_PROJECT   # project = fallback for legacy-run id recovery
    WANDB_ARGS=(--wandb-meta "$META")
  fi

  echo "=== BENCH (ckpt=$(basename "$CKPT"), model_name=${MODEL_NAME}, seed=${SEED}, split=${BENCH_SPLIT}, θ*=${THETA}, tile_metrics=${TILE_METRICS:-none}) ==="
  python -m benchmarking.cli eval \
    --dataset-dir "$DATASET_DIR" \
    --checkpoint "$CKPT" \
    --model unet \
    --model-name "$MODEL_NAME" \
    --seed "$SEED" \
    --store-dir "$STORE_DIR" \
    --split "$BENCH_SPLIT" \
    --threshold "$THETA" \
    --exp-tag "loss_${EXP_TAG}" \
    --label-source "$LABEL_SOURCE" \
    ${METRIC_ARGS[@]+"${METRIC_ARGS[@]}"} \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    ${MASK_ARGS[@]+"${MASK_ARGS[@]}"}

  echo "=== BENCH DONE ===  store: ${STORE_DIR}"
  echo "Decision: python -m benchmarking.cli report --store-dir ${STORE_DIR} --metric f1 --metric iou --metric apls"
  echo "          python scripts/phase_a_report.py --runs /scratch/${USER_NAME}/InstaRoad/runs"
  exit 0
fi

# ============================== STAGE: fit ===================================
if [ "$STAGE" != "fit" ] && [ "$STAGE" != "train" ]; then
  echo "ERROR: STAGE must be tune, fit/train or bench, got '${STAGE}'." >&2
  exit 2
fi

# Optional Stage-0 gate: run the loss unit tests once before training (protocol:
# unit gates must pass before any run). Off by default so it doesn't re-run per
# arm; set STAGE0=true on the first arm you submit.
if [ "${STAGE0:-false}" = "true" ]; then
  echo "=== STAGE 0: pytest tests/test_losses.py ==="
  (cd "$REPO_DIR" && python -m pytest tests/test_losses.py -q) || {
    echo "ERROR: loss unit tests failed — aborting." >&2; exit 1; }
fi

# Assemble optional passthrough args (empty stays absent under set -u).
OPT_ARGS=()
[ -n "${BATCH_SIZE}" ] && OPT_ARGS+=(--batch-size "$BATCH_SIZE")
[ -n "${ENCODER}" ] && OPT_ARGS+=(--encoder "$ENCODER")
[ "${AUGMENT}" = "true" ] && OPT_ARGS+=(--augment) || OPT_ARGS+=(--no-augment)

# --- mix_w resolution (pstar arms): explicit MIX_W > tuned file > 0.5 --------
RESOLVED_MW="0.5"
RESOLVED_TVA="$TVERSKY_ALPHA"
case "$ARM" in pstar_dice*|pstar_tversky*)
  if [ -n "${MIX_W}" ]; then
    RESOLVED_MW="$MIX_W"
  elif [ -f "${RUN_DIR}/best_loss_params.yaml" ]; then
    RESOLVED_MW=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['mix_w'])" "${RUN_DIR}/best_loss_params.yaml")
    RESOLVED_TVA=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['tversky_alpha'])" "${RUN_DIR}/best_loss_params.yaml")
    echo "mix_w=${RESOLVED_MW} tversky_alpha=${RESOLVED_TVA} (tuned; best_loss_params.yaml)"
  else
    case "$ARM" in
      *+*) echo "WARN: no MIX_W set — fitting at 0.5. Phase C should INHERIT B*'s tuned ratio: pass MIX_W=<value from the Phase B best_loss_params.yaml>." >&2 ;;
      *)   echo "WARN: no MIX_W and no best_loss_params.yaml — fitting at the frozen 0.5 (run STAGE=tune first for the searched ratio)." >&2 ;;
    esac
  fi ;;
esac

echo "=== FIT ${FULL_TAG} (arm=${ARM}, ${EPOCHS} epochs, lr=${LR}, seed=${SEED}, mix_w=${RESOLVED_MW}) ==="
python -m unet.train_ablation \
  --base-config "$BASE_CONFIG" \
  --base-config "$NORM_CONFIG" \
  --dataset-dir "$DATASET_DIR" \
  ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} \
  --out "$RUN_DIR" \
  --arm "$ARM" \
  --pstar "$PSTAR" \
  --pos-weight "$POS_WEIGHT" \
  --gap-r "$GAP_R" --gap-k "$GAP_K" \
  --tl-ell "$TL_ELL" --tl-theta "$TL_THETA" \
  --tversky-alpha "$RESOLVED_TVA" --mix-w "$RESOLVED_MW" \
  --cl-alpha "$CL_ALPHA" --cl-iters "$CL_ITERS" \
  --sr-w "$SR_W" --sr-radius "$SR_RADIUS" \
  --warmup-start "$WARMUP_START" --warmup-ramp "$WARMUP_RAMP" \
  --epochs "$EPOCHS" --lr "$LR" \
  --num-workers "$NUM_WORKERS" --precision "$PRECISION" \
  --seed "$SEED" \
  --run-name "${FULL_TAG}_s${SEED}" \
  --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
  ${OPT_ARGS[@]+"${OPT_ARGS[@]}"}

echo "=== FIT DONE ===  ${RUN_DIR}"
echo "Next: re-run this arm with STAGE=bench (or use train_both.sbatch to chain fit->bench)."
