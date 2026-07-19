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
# Unlike the unet/sr engines there is NO Optuna search: the protocol fixes the
# screening LR (1e-3, valid across arms thanks to the §4.4 scale normalization)
# and screens the loss + its hyperparameters, not the optimiser. So the stages
# are:
#   STAGE=tune    NO-OP. Loss ablation has nothing to search; this stage exists
#                 only so train_both.sbatch's tune->fit->bench chain still works.
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
case "$ARM" in
  *gap_tl_ce*)             HP_SLUG="_r${GAP_R}_l${TL_ELL}_th${TL_THETA}" ;;
  *gap_ce*)                HP_SLUG="_r${GAP_R}" ;;
  *tl_ce*|*t2_ce*|*t4_ce*) HP_SLUG="_l${TL_ELL}_th${TL_THETA}" ;;
  pstar_dice|pstar_tversky) HP_SLUG="_p${PSTAR}" ;;
  *)                       HP_SLUG="" ;;
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
# No search for loss ablation — this stage is a deliberate no-op so the
# train_both.sbatch tune->fit->bench chain runs unchanged.
if [ "$STAGE" = "tune" ]; then
  echo "=== TUNE: no-op (loss ablation has no hyperparameter search) ==="
  echo "Next: STAGE=fit trains the arm at the fixed screening LR (${LR})."
  exit 0
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

echo "=== FIT ${FULL_TAG} (arm=${ARM}, ${EPOCHS} epochs, lr=${LR}, seed=${SEED}) ==="
python -m unet.train_ablation \
  --base-config "$BASE_CONFIG" \
  --base-config "$NORM_CONFIG" \
  --dataset-dir "$DATASET_DIR" \
  ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} \
  --out "$RUN_DIR" \
  --arm "$ARM" \
  --pstar "$PSTAR" \
  --gap-r "$GAP_R" --gap-k "$GAP_K" \
  --tl-ell "$TL_ELL" --tl-theta "$TL_THETA" \
  --tversky-alpha "$TVERSKY_ALPHA" \
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
