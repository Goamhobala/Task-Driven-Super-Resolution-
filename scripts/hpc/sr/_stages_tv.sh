#!/bin/bash
# Shared tune/fit/bench engine for the FINAL (train+val refit) SR series.
# NOT submitted directly — each r*_new.sh sets its config and sources this.
#
# This is the sibling of _stages.sh. Same arms, same recipe-v2 dynamics, same
# model code. Two deliberate differences, and nothing else:
#
#   1. DATASET   LABELS=new -> ROSA_New (the final curated dataset). The _all
#                series stays pointed at ROSA_all; run dirs, Optuna studies and
#                benchmark model_names are keyed on EXP_TAG (r0_new vs r0_all),
#                so old and new results can never mix in the store.
#
#   2. PROTOCOL  STAGE=tune is UNCHANGED — Optuna still trains on `train` and
#                scores on `val`, because that is what the holdout is for.
#                STAGE=fit then RE-FOLDS val into the training set
#                (data.train_splits = [train, val]) and reports on `test`
#                alone. Hyperparameters were already paid for out of val; once
#                chosen, withholding those tiles from the fit throws away ~17%
#                of the data for no inferential gain.
#
# Consequence of (2): during the refit there is NO honest holdout, so the
# val-driven machinery is removed rather than allowed to peek at data the model
# now trains on. src/sr/configs/joint_sr_trainval.yaml does this:
#   * limit_val_batches: 0     — no val loop at all
#   * EarlyStopping dropped    — FIXED, pre-registered epoch budget, identical
#                                across arms (same fairness rule as the loss
#                                ablation). Recipe v2's cosine has T_max =
#                                max_epochs, so the budget ends at LR 0: the
#                                schedule always completes.
#   * monitor: null            — the tested checkpoint is the END of the
#                                budget, saved as unet_s2rosa_jointsr_final.ckpt
#                                (NOT ..._best.ckpt, which by convention means
#                                "argmax over val" — the two names must never
#                                be confusable downstream).
#
# ---- Interface (identical to _stages.sh unless noted) -----------------------
# Experiment scripts must set:
#   EXP_TAG      e.g. r2a_new (drives the run dir + study + benchmark name)
#   LABELS       new | all | cdngi | overture | osm   (label/dataset source)
#   UPSAMPLER    sen2sr | sen2sr_full | sr4rs | bicubic
#   FREEZE_SR    true | false
#   SR_PAD       reflect-pad in native px (0 = off, 8 = border-artifact fix)
#
# Optional (submit-time or experiment-script) — see _stages.sh for the full
# prose on each; they behave identically here:
#   WARM_START_CKPT  stage-1 UNet init (r6/r7 staged protocol). _warm_tv.sh
#                derives it from the stage-1 arm's FINAL ckpt.
#   LOSS_ARM     any unet.losses.build_loss arm (+ its hps).
#   Recipe v2:   CLIP LR_SCHEDULE SR_WARMUP_EPOCHS L2SP_LAMBDA REG
#   SR_SNAPSHOT_EVERY   fit-stage SR-weights-only snapshots every N epochs.
#
# NEW here:
#   REFIT_EPOCHS     the pre-registered budget (default 100). This is now a
#                nothing-stops-it-early budget, so it is a BETWEEN-ARM CONSTANT
#                of the protocol: change it for one arm and the comparison is
#                void. Check the walltime — every arm now runs the full count.
#   TRAIN_SPLITS     "train val" (default). Set "train" to reproduce the old
#                holdout protocol on ROSA_New without switching engines; the
#                run dir / study / bench name then gain a _holdout tag so the
#                two protocols can never land in the same store row.
#   NORM_CONFIG      normalisation stats. DEFAULT = <DATASET_DIR>/norm_stats.yaml,
#                i.e. the file `sentinel2data.cli norm-stats` writes into the
#                dataset itself — the only copy guaranteed to have been computed
#                from THIS dataset's splits/train.csv. Unlike _stages.sh, this
#                engine does NOT hard-code the repo copy; it falls back to it
#                only if the dataset has none, and then refuses to run without
#                NORM_FALLBACK_OK=1.
#
# STAGE=tune   Optuna on train/val (unchanged). CHAIN_FIT=1 continues into fit
#              in the same allocation. Early stop: `touch <run dir>/STOP`.
#              Rescue a killed search: rerun with N_TRIALS=0.
# STAGE=fit    Refit on train+val for REFIT_EPOCHS on ONE GPU, then test.
#              RESUME_FIT=1 continues from last.ckpt.
# STAGE=bench  Score unet_s2rosa_jointsr_final.ckpt into the shared store.
#
# Replication contract: only SEED, STAGE and the loss block are meant to vary.
set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"

: "${EXP_TAG:?experiment script must set EXP_TAG}"
: "${LABELS:-new}"
: "${UPSAMPLER:?experiment script must set UPSAMPLER (sen2sr|sen2sr_full|sr4rs|bicubic)}"
: "${FREEZE_SR:?experiment script must set FREEZE_SR (true|false)}"
: "${SR_PAD:?experiment script must set SR_PAD (0 = off)}"

STAGE="${STAGE:-tune}"
SEED="${SEED:-0}"
# 0 was a DDP-era guard (GDAL handles + forked ranks). Search runs one
# single-GPU process per GPU and the refit is single-GPU, and the datasets
# open rasters lazily inside __getitem__, so forked loader workers are safe.
# Default is computed below, after SEARCH_GPUS is known.
NUM_WORKERS="${NUM_WORKERS:-}"
PRECISION="${PRECISION:-bf16-mixed}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN}"
WARM_START_CKPT="${WARM_START_CKPT:-}"

# --- The train+val protocol switch -------------------------------------------
# Space-separated split names for the REFIT's train loader. The default IS the
# protocol; "train" reverts to the classic holdout fit and tags itself so the
# two never mix.
TRAIN_SPLITS="${TRAIN_SPLITS:-train val}"
PROTO_TAG=""
MERGE_VAL=1
case " ${TRAIN_SPLITS} " in
  *" test "*)
    echo "ERROR: TRAIN_SPLITS must never contain 'test' — that is the held-out" >&2
    echo "  evaluation split. Got '${TRAIN_SPLITS}'." >&2
    exit 2 ;;
  *" val "*) : ;;
  *) PROTO_TAG="_holdout"; MERGE_VAL=0 ;;
esac

# --- Recipe v2 training dynamics (defaults = the agreed recipe) --------------
REG="${REG:-true}"
REG_TAG=""
if [ "$REG" = "false" ] || [ "$REG" = "0" ]; then
  REG_TAG="_noreg"
  CLIP="${CLIP:-0}"
  LR_SCHEDULE="${LR_SCHEDULE:-none}"
  SR_WARMUP_EPOCHS="${SR_WARMUP_EPOCHS:-0}"
fi
CLIP="${CLIP:-1.0}"                          # gradient clip (global L2; 0=off)
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"         # cosine | none
SR_WARMUP_EPOCHS="${SR_WARMUP_EPOCHS:-1.0}"  # SR-group ramp; model auto-off
                                             # for frozen/bicubic/warm-start
L2SP_LAMBDA="${L2SP_LAMBDA:-0.0}"            # 0 = dormant L2-SP anchor

SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"

# LABELS -> dataset dir + code-level mask_source (+ mask_dirname when raster).
# LABELS=new reads the ONCE-OFF pre-rasterised HR label COGs (the labels are
# frozen; per-crop graph rasterisation was the GPU-starving bottleneck).
# Generate them one time per dataset (standalone script, no PYTHONPATH/GPU;
# login node is fine):
#   $VENV_DIR/bin/python $REPO_DIR/src/sentinel2data/dataset/rasterize_hr_masks.py \
#     --dataset-dir <DATASET_DIR> --out-dirname mask_new_2pt5
# Submit-time MASK_SOURCE=graph reverts to on-the-fly rasterisation.
case "$LABELS" in
  new)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_New}"
    MASK_SOURCE="${MASK_SOURCE:-raster}"   # pre-rasterised graph labels
    MASK_DIRNAME="${MASK_DIRNAME:-mask_new_2pt5}" ;;
  all)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_all}"
    MASK_SOURCE="graph" ;;
  cdngi)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
    MASK_SOURCE="graph" ;;
  overture)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_Overture}"
    MASK_SOURCE="graph" ;;
  osm)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_New}"
    MASK_SOURCE="raster"
    MASK_DIRNAME="${MASK_DIRNAME:-mask_osm_2pt5}" ;;  # OSM HR rasters
  *)
    echo "ERROR: LABELS must be new|all|cdngi|overture|osm, got '${LABELS}'." >&2; exit 2 ;;
esac
MASK_DIRNAME="${MASK_DIRNAME:-}"   # empty for the graph (on-the-fly) sources

# --- Tune budget (train/val — UNCHANGED from _stages.sh) ---------------------
N_TRIALS="${N_TRIALS:-60}"
SEARCH_GPUS="${SEARCH_GPUS:-2}"
TUNE_EPOCHS="${TUNE_EPOCHS:-15}"
PATIENCE="${PATIENCE:-3}"
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-imagenet}"
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-1e-2}"
LR_SR_MIN="${LR_SR_MIN:-1e-7}"   # searched only when SR is learned & unfrozen
LR_SR_MAX="${LR_SR_MAX:-1e-3}"
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-3.352251180486363}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-3.352251180486363}"
ENCODERS="${ENCODERS:-resnet34}"      # NOT searched: encoder constancy is the control
BATCH_SIZES="${BATCH_SIZES:-2 4 8}"   # 512px UNet stage is memory-heavy

# Loader workers per training process: split the job's CPU allocation across
# the stage's processes (search fans out SEARCH_GPUS tuners; fit/test run one).
# Workers spend most time blocked on the prefetch queue, so no cores are
# reserved for the mains. With pre-rasterised masks (mask_new_2pt5) 1-2
# workers already keep the GPU fed; set NUM_WORKERS explicitly only if GPU
# utilisation sawtooths again.
if [ -z "${NUM_WORKERS}" ]; then
  JOB_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-4}}"
  if [ "${STAGE}" = "tune" ]; then
    NUM_WORKERS=$(( JOB_CPUS / SEARCH_GPUS ))
  else
    NUM_WORKERS=$(( JOB_CPUS - 1 ))
  fi
  [ "${NUM_WORKERS}" -lt 1 ] && NUM_WORKERS=1
fi

# --- Fit budget (train+val, FIXED — no early stopping) -----------------------
# Pre-registered and identical across arms. Nothing truncates it now, so budget
# the SLURM walltime for the full count on the SLOWEST arm (sr4rs).
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_joint_final}"

# --- Loss (unet.losses.build_loss; empty = legacy Dice + pos-weighted BCE) ---
LOSS_ARM="${LOSS_ARM:-gap_tl_ce}"
PSTAR="${PSTAR:-bce}"
GAP_R="${GAP_R:-4}";                 GAP_K="${GAP_K:-60.0}"
TL_ELL="${TL_ELL:-5}";               TL_THETA="${TL_THETA:-0.5409065645350193}"
GAP_THETA="${GAP_THETA:-0.38105240274638613}"        # official gap binarization
# R-SERIES RULE: the loss is a FROZEN CONTROL across R-arms. Pin the pilot
# winner's config at submit time: SEARCH_THETAS=false TL_THETA=<θ*>
# GAP_THETA=<θ*> POS_WEIGHT_MIN=<λ*> POS_WEIGHT_MAX=<λ*> (min==max = a
# constant). Leaving SEARCH_THETAS=true re-searches loss hps per R-arm and
# confounds the SR comparison.
SEARCH_THETAS="${SEARCH_THETAS:-true}"
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
CL_ALPHA="${CL_ALPHA:-0.3}";         CL_ITERS="${CL_ITERS:-5}"
SKEL_W="${SKEL_W:-1.0}";             SKEL_RADIUS="${SKEL_RADIUS:-1}"
WARMUP_START="${WARMUP_START:-30}";  WARMUP_RAMP="${WARMUP_RAMP:-10}"

LOSS_TAG=""
LOSS_ARGS_TUNE=()   # sr.tune flags (argparse)
LOSS_ARGS_FIT=()    # sr.cli fit/test flags (LightningCLI --model.*)
if [ -n "$LOSS_ARM" ]; then
  # '+' is not filesystem/wandb-friendly -> bce_dice+cldice => bce_dice-cldice
  LOSS_TAG="_$(echo "$LOSS_ARM" | tr '+' '-')"
  LOSS_ARGS_TUNE=(--loss-arm "$LOSS_ARM" --pstar "$PSTAR"
                  --gap-r "$GAP_R" --gap-k "$GAP_K"
                  --tl-ell "$TL_ELL" --tl-theta "$TL_THETA"
                  --gap-theta "$GAP_THETA" --search-thetas "$SEARCH_THETAS"
                  --tversky-alpha "$TVERSKY_ALPHA"
                  --cl-alpha "$CL_ALPHA" --cl-iters "$CL_ITERS"
                  --skel-w "$SKEL_W" --skel-radius "$SKEL_RADIUS"
                  --warmup-start "$WARMUP_START" --warmup-ramp "$WARMUP_RAMP")
  # NB tl_theta/gap_theta/pos_weight are NOT in the fit belt: the tune pins
  # them (searched or fixed) into best_params.yaml, and an explicit --model.*
  # here would override the pinned values with the env defaults. The overlay
  # is authoritative for those dims. (Ported from the LS twin, 2026-08-04.)
  LOSS_ARGS_FIT=(--model.loss_arm "$LOSS_ARM" --model.pstar "$PSTAR"
                 --model.gap_r "$GAP_R" --model.gap_k "$GAP_K"
                 --model.tl_ell "$TL_ELL"
                 --model.tversky_alpha "$TVERSKY_ALPHA"
                 --model.cl_alpha "$CL_ALPHA" --model.cl_iters "$CL_ITERS"
                 --model.sr_w "$SKEL_W" --model.sr_radius "$SKEL_RADIUS"
                 --model.warmup_start "$WARMUP_START" --model.warmup_ramp "$WARMUP_RAMP")
fi
# =============================================================================

BASE_CONFIG="$REPO_DIR/src/sr/configs/joint_sr.yaml"
TRAINVAL_CONFIG="$REPO_DIR/src/sr/configs/joint_sr_trainval.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"

# --- Norm stats: read them from the DATASET, not from the repo ---------------
# `sentinel2data.cli norm-stats` writes <dataset_dir>/norm_stats.yaml by
# default, so every dataset already ships the stats computed from ITS OWN
# splits/train.csv — which is the only file that can be correct for it.
# The repo copy at src/unet/configs/norm_stats.yaml is a hand-copy of one
# dataset's file (its header still names the dataset it came from); pointing
# every experiment at that single path means the stats silently stop matching
# the moment you switch datasets, and nothing in the run would tell you.
# So: prefer the dataset's own file, fall back to the repo copy only if the
# dataset has none, and say loudly which one is in use. NORM_CONFIG=<path>
# overrides both.
NORM_CONFIG_DATASET="${DATASET_DIR}/norm_stats.yaml"
NORM_CONFIG_REPO="$REPO_DIR/src/unet/configs/norm_stats.yaml"
if [ -n "${NORM_CONFIG:-}" ]; then
  NORM_SOURCE="explicit NORM_CONFIG override"
elif [ -f "$NORM_CONFIG_DATASET" ]; then
  NORM_CONFIG="$NORM_CONFIG_DATASET"
  NORM_SOURCE="dataset"
else
  NORM_CONFIG="$NORM_CONFIG_REPO"
  NORM_SOURCE="repo fallback"
fi
# RUNS_ROOT is shared with _warm_tv.sh, which reconstructs the STAGE-1 run dir
# from it — override one and you must override both, so they read the same var.
RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"
RUN_DIR="${RUNS_ROOT}/sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}${PROTO_TAG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# The refit's checkpoint. Named _final, never _best: under this protocol no
# checkpoint was ever selected on a holdout, and the filename says so.
FINAL_CKPT_NAME="unet_s2rosa_jointsr_final"

LOG_FILE="${RUN_DIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"
echo "host=$(hostname)  exp=sr/${EXP_TAG}  stage=${STAGE}  seed=${SEED}"
echo "labels=${LABELS} (mask_source=${MASK_SOURCE}${MASK_DIRNAME:+, mask_dirname=${MASK_DIRNAME}})  upsampler=${UPSAMPLER}  freeze_sr=${FREEZE_SR}  sr_pad=${SR_PAD}  loss_arm=${LOSS_ARM:-legacy}"
echo "recipe: reg=${REG}${REG_TAG:+ [${REG_TAG}]}  clip=${CLIP}  lr_schedule=${LR_SCHEDULE}  sr_warmup_epochs=${SR_WARMUP_EPOCHS}  l2sp_lambda=${L2SP_LAMBDA}  sr_snapshot_every=${SR_SNAPSHOT_EVERY}"
echo "protocol: tune on train/val -> refit on '${TRAIN_SPLITS}' (merge_val=${MERGE_VAL}) -> report on test"
echo "DATASET_DIR=${DATASET_DIR}  warm_start=${WARM_START_CKPT:-none}"
echo "norm_stats=${NORM_CONFIG}  [${NORM_SOURCE}]"

# --- Fail fast ---------------------------------------------------------------
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  echo "  (LABELS=${LABELS}. Upload the final dataset, or override DATASET_DIR.)" >&2
  exit 1
fi
for _s in splits/train.csv splits/val.csv splits/test.csv; do
  if [ ! -f "${DATASET_DIR}/${_s}" ]; then
    echo "ERROR: ${DATASET_DIR}/${_s} missing — the train+val protocol needs all" >&2
    echo "  three split CSVs (val is merged at fit; test is the only report set)." >&2
    exit 1
  fi
done
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: no norm stats for this dataset." >&2
  echo "  Looked for: ${NORM_CONFIG_DATASET}" >&2
  echo "  and:        ${NORM_CONFIG_REPO}" >&2
  echo "  Generate the dataset's own (this is the default output path):" >&2
  echo "    python -m sentinel2data.cli norm-stats --dataset-dir ${DATASET_DIR}" >&2
  exit 1
fi
# The repo fallback belongs to whichever dataset it was last copied from, so it
# is a coin flip on any other one. Refuse to guess silently.
if [ "${NORM_SOURCE}" = "repo fallback" ]; then
  echo "WARN: ${DATASET_DIR}/norm_stats.yaml does not exist; falling back to the" >&2
  echo "  repo copy ${NORM_CONFIG_REPO}, which was computed from a DIFFERENT" >&2
  echo "  dataset's splits/train.csv. Wrong mean/std shifts every input the model" >&2
  echo "  ever sees, and nothing downstream would flag it. Generate the real one:" >&2
  echo "    python -m sentinel2data.cli norm-stats --dataset-dir ${DATASET_DIR}" >&2
  echo "  (NORM_FALLBACK_OK=1 to proceed anyway.)" >&2
  if [ "${NORM_FALLBACK_OK:-0}" != "1" ]; then
    exit 1
  fi
  echo "  NORM_FALLBACK_OK=1 — proceeding on the repo copy." >&2
fi
if [ ! -f "${TRAINVAL_CONFIG}" ]; then
  echo "ERROR: ${TRAINVAL_CONFIG} missing — this engine needs the refit overlay." >&2
  exit 1
fi
case "${UPSAMPLER}" in
  sen2sr|sen2sr_full)
    if [ ! -f "${SEN2SR_DIR}/model.safetensor" ]; then
      echo "ERROR: SEN2SR weights not at ${SEN2SR_DIR} (upsampler=${UPSAMPLER})." >&2
      echo "  Lite: prefetch with sr.sen2sr_loader.download_sen2sr on a login node;" >&2
      echo "  full: download the SEN2SR (Mamba) mlstac dir there yourself." >&2
      exit 1
    fi ;;
  sr4rs)
    if [ ! -f "${SEN2SR_DIR}/gen_weights.safetensors" ]; then
      echo "ERROR: SR4RS extracted weights not at ${SEN2SR_DIR}/gen_weights.safetensors." >&2
      echo "  Run scripts/sr4rs/extract_sr4rs.py locally (TF venv), verify with" >&2
      echo "  'python -m sr.sr4rs_torch --model-dir ...', then upload the three" >&2
      echo "  gen_* files into ${SEN2SR_DIR}." >&2
      exit 1
    fi ;;
esac
if [ -n "${WARM_START_CKPT}" ] && [ ! -f "${WARM_START_CKPT}" ]; then
  echo "ERROR: WARM_START_CKPT=${WARM_START_CKPT} not found — run the stage-1" >&2
  echo "  (frozen-SR) arm's STAGE=fit first; its final ckpt seeds this arm's UNet." >&2
  exit 1
fi
if [ "${MASK_SOURCE}" = "raster" ]; then
  # -print -quit: no pipe to `head`, so `find` can't die of SIGPIPE and trip
  # `set -o pipefail`.
  first_mask=$(find "${DATASET_DIR}"/*/"${MASK_DIRNAME}" -maxdepth 1 -name '*.tif' -print -quit 2>/dev/null)
  if [ -z "${first_mask}" ]; then
    echo "ERROR: MASK_SOURCE=raster but no masks under <split>/${MASK_DIRNAME}/." >&2
    if [ "${LABELS}" = "osm" ]; then
      echo "  Generate with OpenStreetMapTest/dataset_hr_masks.py --scale 4" >&2
    else
      echo "  Generate ONCE with (standalone, login node is fine):" >&2
      echo "    ${VENV_DIR}/bin/python ${REPO_DIR}/src/sentinel2data/dataset/rasterize_hr_masks.py \\" >&2
      echo "      --dataset-dir ${DATASET_DIR} --out-dirname ${MASK_DIRNAME}" >&2
      echo "  (or MASK_SOURCE=graph to rasterise on the fly — slow.)" >&2
    fi
    exit 1
  fi
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "python=$(which python)"

if [ "${UPSAMPLER}" = "sen2sr_full" ] && ! python -c "import mamba_ssm" 2>/dev/null; then
  echo "ERROR: upsampler=sen2sr_full but mamba_ssm is not importable in ${VENV_DIR}." >&2
  echo "  Install on a GPU node with matching torch/CUDA:  uv pip install mamba-ssm" >&2
  exit 1
fi

# ============================== STAGE: tune ==================================
# IDENTICAL to _stages.sh: the search trains on `train` and scores on `val`.
# The holdout is spent here, deliberately and once.
if [ "$STAGE" = "tune" ]; then
  STORAGE="${STORAGE:-sqlite:///${RUN_DIR}/study.db}"
  SAMPLER_OFFSET="${SAMPLER_OFFSET:-0}"
  STUDY_NAME="sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}${PROTO_TAG}_seed${SEED}"

  run_tuner () {   # $1=gpu id (empty = no pin)  $2=n-trials  $3=seed
    local gpu="$1" ntrials="$2" seed="$3" pin=""
    [ -n "$gpu" ] && pin="CUDA_VISIBLE_DEVICES=$gpu"
    env $pin python -m sr.tune \
      --base-config "$BASE_CONFIG" \
      --base-config "$NORM_CONFIG" \
      --dataset-dir "$DATASET_DIR" \
      --sen2sr-dir "$SEN2SR_DIR" \
      --mask-source "$MASK_SOURCE" \
      ${MASK_DIRNAME:+--mask-dirname "$MASK_DIRNAME"} \
      --upsampler "$UPSAMPLER" \
      --freeze-sr "$FREEZE_SR" \
      --sr-pad "$SR_PAD" \
      ${WARM_START_CKPT:+--warm-start-unet "$WARM_START_CKPT"} \
      --out "$RUN_DIR" \
      --num-workers "$NUM_WORKERS" \
      --devices 1 \
      --n-trials "$ntrials" \
      --max-epochs "$TUNE_EPOCHS" \
      --patience "$PATIENCE" \
      --precision "$PRECISION" \
      --clip "$CLIP" \
      --lr-schedule "$LR_SCHEDULE" \
      --sr-warmup-epochs "$SR_WARMUP_EPOCHS" \
      --l2sp-lambda "$L2SP_LAMBDA" \
      --seed "$seed" \
      --train-seed "$SEED" \
      --study-name "$STUDY_NAME" \
      --storage "$STORAGE" \
      --encoder-weights "$ENCODER_WEIGHTS" \
      --lr-min "$LR_MIN" --lr-max "$LR_MAX" \
      --lr-sr-min "$LR_SR_MIN" --lr-sr-max "$LR_SR_MAX" \
      --pos-weight-min "$POS_WEIGHT_MIN" --pos-weight-max "$POS_WEIGHT_MAX" \
      --encoders $ENCODERS \
      --batch-sizes $BATCH_SIZES \
      ${LOSS_ARGS_TUNE[@]+"${LOSS_ARGS_TUNE[@]}"}
  }

  N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
  if [ "${N_GPUS}" -eq 0 ] && [ "${SEARCH_GPUS}" -gt 1 ]; then
    echo "WARN: no CUDA device visible — running ONE unpinned tuner on CPU (smoke-test mode)." >&2
    SEARCH_GPUS=1
  elif [ "${N_GPUS}" -gt 0 ] && [ "${SEARCH_GPUS}" -gt "${N_GPUS}" ]; then
    echo "WARN: SEARCH_GPUS=${SEARCH_GPUS} but only ${N_GPUS} GPU(s) visible — capping to ${N_GPUS}." >&2
    SEARCH_GPUS="${N_GPUS}"
  fi

  echo "=== OPTUNA SEARCH on train/val (n_trials=$N_TRIALS across ${SEARCH_GPUS} GPU(s), ${TUNE_EPOCHS} epochs/trial) ==="
  echo "    stop early (keeps study + writes overlay):  touch ${RUN_DIR}/STOP"
  if [ "$SEARCH_GPUS" -le 1 ]; then
    run_tuner "" "$N_TRIALS" "$(( SEED * 1000 + SAMPLER_OFFSET ))"
  else
    PER_WORKER=$(( (N_TRIALS + SEARCH_GPUS - 1) / SEARCH_GPUS ))
    echo "  fanning out ${SEARCH_GPUS} workers x ${PER_WORKER} trials each"
    pids=()
    for (( g=0; g<SEARCH_GPUS; g++ )); do
      run_tuner "$g" "$PER_WORKER" "$(( SEED * 1000 + SAMPLER_OFFSET + g ))" &
      pids+=($!)
      sleep 3   # stagger so worker 0 creates the study before the others attach
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    [ "$fail" -eq 0 ] || { echo "ERROR: an Optuna search worker failed (see log above)." >&2; exit 1; }
  fi
  echo "=== SEARCH DONE ===  best_params.yaml + study.db in $RUN_DIR"
  echo "Next (refit on train+val, then test):"
  echo "  bash scripts/hpc/submit.sh sr/${EXP_TAG}.sh STAGE=fit SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}"
  exit 0
fi

# ============================== STAGE: bench =================================
# Score the FINAL checkpoint into the shared benchmark store, at 2.5 m against
# the experiment's own GT, through the same joint_sr_dataset helpers training
# used. Default split is `test` — the only split this protocol reports.
if [ "$STAGE" = "bench" ]; then
  CKPT="${RUN_DIR}/checkpoints/${FINAL_CKPT_NAME}.ckpt"
  if [ ! -f "$CKPT" ]; then
    if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
      echo "WARN: ${FINAL_CKPT_NAME}.ckpt missing; benchmarking last.ckpt instead." >&2
      CKPT="${RUN_DIR}/checkpoints/last.ckpt"
    else
      echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — run STAGE=fit first." >&2
      exit 1
    fi
  fi

  STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks}"   # SHARED across experiments
  MODEL_NAME="${MODEL_NAME:-sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}${PROTO_TAG}}"
  LABEL_SOURCE="${LABEL_SOURCE:-${LABELS}}"
  BENCH_SPLIT="${BENCH_SPLIT:-test}"
  TILE_METRICS="${TILE_METRICS:-apls}"

  # val tiles are TRAINING tiles under this protocol — scoring on them would be
  # a train-set number sitting in the same store as honest test numbers.
  if [ "$MERGE_VAL" = "1" ] && [ "$BENCH_SPLIT" = "val" ]; then
    echo "ERROR: BENCH_SPLIT=val, but val was folded into training (TRAIN_SPLITS='${TRAIN_SPLITS}')." >&2
    echo "  That score would be a training score. Use BENCH_SPLIT=test." >&2
    exit 2
  fi

  CONFIG_ARGS=()
  [ -f "${RUN_DIR}/best_params.yaml" ] && CONFIG_ARGS=(--config-yaml "${RUN_DIR}/best_params.yaml")
  MASK_ARGS_BENCH=(--mask-source "$MASK_SOURCE")
  [ "$MASK_SOURCE" = "raster" ] && MASK_ARGS_BENCH+=(--mask-dirname "$MASK_DIRNAME")
  METRIC_ARGS=()
  if [ -n "${TILE_METRICS}" ]; then
    IFS=',' read -r -a _TMS <<< "${TILE_METRICS}"
    for _tm in "${_TMS[@]}"; do METRIC_ARGS+=(--tile-metric "${_tm}"); done
  fi

  echo "=== BENCH (ckpt=$(basename "$CKPT"), model_name=${MODEL_NAME}, seed=${SEED}, split=${BENCH_SPLIT}, gt=${MASK_SOURCE}, tile_metrics=${TILE_METRICS:-none}) ==="
  python -m benchmarking.cli eval \
    --dataset-dir "$DATASET_DIR" \
    --checkpoint "$CKPT" \
    --model sr \
    --model-name "$MODEL_NAME" \
    --seed "$SEED" \
    --store-dir "$STORE_DIR" \
    --split "$BENCH_SPLIT" \
    --sen2sr-dir "$SEN2SR_DIR" \
    --exp-tag "$EXP_TAG" \
    --label-source "$LABEL_SOURCE" \
    ${METRIC_ARGS[@]+"${METRIC_ARGS[@]}"} \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    "${MASK_ARGS_BENCH[@]}"

  echo "=== BENCH DONE ===  store: ${STORE_DIR}"
  echo "Report: python -m benchmarking.cli report --store-dir ${STORE_DIR}"
  exit 0
fi

# ============================== STAGE: fit ===================================
if [ "$STAGE" != "fit" ]; then
  echo "ERROR: STAGE must be tune, fit or bench, got '${STAGE}'." >&2
  exit 2
fi

BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/${FINAL_CKPT_NAME}.ckpt"
if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run STAGE=tune first." >&2
  exit 1
fi
echo "--- best hyperparameters (chosen on val, before the merge) ---"; cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"
RESUME_ARGS=()
if [ "${RESUME_FIT:-0}" = "1" ]; then
  if [ -f "$LAST_CKPT" ]; then
    echo "=== RESUME_FIT=1: continuing the refit from ${LAST_CKPT} ==="
    RESUME_ARGS=(--ckpt_path "$LAST_CKPT")
  else
    echo "WARN: RESUME_FIT=1 but ${LAST_CKPT} not found — refitting from scratch." >&2
  fi
fi

# The SR treatment (and loss arm) is passed explicitly (belt) even though the
# best_params overlay records it too (braces) — drift is impossible.
MODEL_ARGS=(--model.upsampler "$UPSAMPLER" --model.freeze_sr "$FREEZE_SR"
            --model.sr_pad "$SR_PAD" --model.sen2sr_dir "$SEN2SR_DIR"
            --model.lr_schedule "$LR_SCHEDULE"
            --model.sr_warmup_epochs "$SR_WARMUP_EPOCHS"
            --model.l2sp_lambda "$L2SP_LAMBDA"
            --model.sr_snapshot_every "$SR_SNAPSHOT_EVERY")
if [ -n "$WARM_START_CKPT" ]; then
  MODEL_ARGS+=(--model.warm_start_unet "$WARM_START_CKPT")
fi
if [ -n "$LOSS_ARM" ]; then
  MODEL_ARGS+=("${LOSS_ARGS_FIT[@]}")
fi

# joint_sr_trainval.yaml is layered LAST so its callback list and
# limit_val_batches win over joint_sr.yaml's val-monitored ones. train_splits is
# ALSO passed explicitly, so a holdout run (TRAIN_SPLITS=train) overrides the
# overlay's default rather than needing a second config file.
# shellcheck disable=SC2206
TRAIN_SPLITS_ARR=(${TRAIN_SPLITS})
SPLIT_ARGS=(--data.train_splits "[$(IFS=,; echo "${TRAIN_SPLITS_ARR[*]}")]")

echo "=== REFIT on '${TRAIN_SPLITS}' (best config, FIXED ${REFIT_EPOCHS} epochs, no early stopping, ${REFIT_GPUS} GPU) ==="
python -m sr.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --config "$TRAINVAL_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  ${MASK_DIRNAME:+--data.mask_dirname "$MASK_DIRNAME"} \
  "${SPLIT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --trainer.max_epochs "$REFIT_EPOCHS" \
  --trainer.devices "$REFIT_GPUS" \
  --trainer.precision "$PRECISION" \
  --trainer.gradient_clip_val "$CLIP" \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --seed_everything "$SEED" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

# Log the test metrics to the SAME wandb run the refit just created.
if LATEST_RUN=$(readlink -f "$RUN_DIR/wandb/latest-run" 2>/dev/null) && [ -n "$LATEST_RUN" ]; then
  export WANDB_RUN_ID="${LATEST_RUN##*-}"   # .../run-<timestamp>-<id> -> <id>
  export WANDB_RESUME=must
  echo "resuming wandb run ${WANDB_RUN_ID} for the test split"
else
  echo "WARN: could not locate the refit's wandb run; test will log to a fresh run" >&2
fi

if [ ! -f "$CKPT" ]; then
  if [ -f "$LAST_CKPT" ]; then
    echo "WARN: ${FINAL_CKPT_NAME}.ckpt missing; testing last.ckpt instead." >&2
    CKPT="$LAST_CKPT"
  else
    echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — refit produced none. Skipping test." >&2
    exit 1
  fi
fi

# The ONLY held-out evaluation in this protocol.
echo "=== TEST (held-out split, ckpt=$(basename "$CKPT")) ==="
python -m sr.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  ${MASK_DIRNAME:+--data.mask_dirname "$MASK_DIRNAME"} \
  "${MODEL_ARGS[@]}" \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
echo "Bench: bash scripts/hpc/submit.sh sr/${EXP_TAG}.sh STAGE=bench SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}"
