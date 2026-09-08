#!/bin/bash
# Shared tune/fit engine for the joint-SR experiment scripts. NOT submitted
# directly — each experiment script (r0_cdngi.sh, r2a_cdngi.sh, ...) sets its
# config and sources this file.
#
# Experiment scripts must set:
#   EXP_TAG      e.g. r2a_cdngi (drives the run dir + study name)
#   LABELS       cdngi | overture | osm  (label SOURCE naming — never "graph",
#                which collides with the graph-model thread; cdngi/overture map
#                to the pipeline's masks_graph parquets of the matching dataset,
#                osm maps to the pre-generated <split>/mask_osm_2pt5 rasters)
#   UPSAMPLER    sen2sr | bicubic
#   FREEZE_SR    true | false
#   SR_PAD       reflect-pad in native px (0 = off, 8 = border-artifact fix)
#
# Optional (submit-time or experiment-script):
#   WARM_START_CKPT  stage-1 (frozen-SR) JointSR ckpt whose UNet weights seed
#                every trial's / the refit's UNet (staged R6/R7 protocol —
#                the r6/r7 scripts derive it from their stage-1 run dir and
#                pin pos_weight/batch/encoder from that run's best_params;
#                the UNet lr is searched over a fine-tuning band anchored to
#                stage-1's best, alongside lr_sr — see _warm.sh, PIN_LR=1
#                for the exact pin). Empty (default) = cold ImageNet UNet
#                (R2/R4 protocol).
#   LOSS_ARM     any unet.losses.build_loss arm (bce | gap_ce | tl_ce |
#                gap_tl_ce | t2_ce | t4_ce | bce_dice | pstar_dice |
#                pstar_tversky | focal_tversky | <base>+cldice |
#                <base>+skelrec). Empty (default) = legacy Dice + pos-weighted
#                BCE. When set: pos_weight is not searched, and the run dir /
#                study / benchmark model_name gain a loss tag so arms never
#                collide. Loss hps: PSTAR GAP_R GAP_K TL_ELL TL_THETA
#                TVERSKY_ALPHA CL_ALPHA CL_ITERS SKEL_W SKEL_RADIUS
#                WARMUP_START WARMUP_RAMP.
#   Recipe v2 (defaults = the agreed cross-arm recipe; override only with
#   cause -- the recipe is a between-arm CONSTANT of the protocol):
#     CLIP              gradient clip, global L2 norm (1.0; 0 = off)
#     LR_SCHEDULE       cosine | none (cosine: per-step decay to 0, T_max =
#                       the stage's own epoch budget -- TUNE_EPOCHS for
#                       trials, REFIT_EPOCHS for the refit)
#     SR_WARMUP_EPOCHS  linear LR ramp on the SR group, absolute epochs
#                       (1.0). The MODEL auto-disables it for frozen/bicubic
#                       SR and warm-start arms, so it is safe to pass always.
#     L2SP_LAMBDA       L2-SP anchor toward the pretrained SR weights (0.0 =
#                       dormant). Escalate only on sr_drift_rel evidence.
#     REG=false         one-switch unregularised-GAN ablation (clip 0, no
#                       schedule, no warmup); auto-tags run/study/bench with
#                       _noreg so it never mixes with the v2 runs.
#   SR_SNAPSHOT_EVERY   fit-stage SR-weights-only snapshots every N epochs
#                       into <run dir>/sr_snapshots/ (+ an epoch-0 init
#                       frame) for replaying the SR output's task-driven
#                       evolution. 0 (default) = off; try 2-5.
#
# STAGE=tune   Optuna search, one INDEPENDENT tuner per GPU, shared sqlite study
#              (no DDP — that's the Optuna constraint). Default headers = gpu:2.
#              Stop EARLY without losing anything: `touch <run dir>/STOP` (or
#              `scancel -s USR1 <jobid>`) — the in-flight trial finishes,
#              best_params.yaml is written, and CHAIN_FIT still applies.
#              Rescue a KILLED search: rerun with N_TRIALS=0 — attaches to the
#              persisted study and writes best_params.yaml in seconds.
#   CHAIN_FIT=1  after the search, continue straight into STAGE=fit in the
#              SAME job/allocation (saves a queue round-trip; the refit uses
#              1 GPU, so any extra search GPUs idle during it).
# STAGE=fit    Refit best config on ONE GPU + wandb test. Submit with --gres=gpu:1.
#              Refits from scratch by default. RESUME_FIT=1 continues a
#              walltime-killed refit from <run dir>/checkpoints/last.ckpt
#              (restores epoch/optimizer/schedule + best-score tracking).
# STAGE=bench  Score the fitted checkpoint into the SHARED benchmark store
#              (per-chip confusion-matrix metrics at 2.5 m against the SAME GT
#              source the model trained on). Standalone on any existing
#              checkpoint; train_both.sbatch chains it. Submit with --gres=gpu:1.
#
# Replication contract: only SEED, STAGE and the loss block (LOSS_ARM + hps —
# tagged into the run dir/study/model_name, so arms never mix) are meant to
# vary at submit time.
set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"

: "${EXP_TAG:?experiment script must set EXP_TAG}"
: "${LABELS:-all}"
: "${UPSAMPLER:?experiment script must set UPSAMPLER (sen2sr|bicubic)}"
: "${FREEZE_SR:?experiment script must set FREEZE_SR (true|false)}"
: "${SR_PAD:?experiment script must set SR_PAD (0 = off)}"

STAGE="${STAGE:-tune}"
SEED="${SEED:-0}"
# 0 was a DDP-era guard; single-GPU-per-process search + lazy per-__getitem__
# raster opens make forked loader workers safe. See _stages_tv.sh.
# Default is computed below, after SEARCH_GPUS is known.
NUM_WORKERS="${NUM_WORKERS:-}"
PRECISION="${PRECISION:-bf16-mixed}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN}"
WARM_START_CKPT="${WARM_START_CKPT:-}"

# --- Recipe v2 training dynamics (defaults = the agreed recipe) --------------
# REG=false flips every stabiliser off in ONE switch (clip / cosine / warmup;
# l2sp is already dormant) -- the "unregularised GAN" ablation -- and tags the
# run dir / study / benchmark name with _noreg so it can NEVER mix with the v2
# runs. Individually-set vars still win either way; if you hand-roll a partial
# ablation instead of using REG=false, tag the run yourself.
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

# --- SR evolution snapshots (STAGE=fit only; demo/insight) -------------------
# Every N epochs save the SR net's WEIGHTS ONLY into <run dir>/sr_snapshots/
# (~45 MB/frame SR4RS, ~2 MB SEN2SR-Lite -- vs ~500 MB full Lightning ckpts),
# plus an epoch-0 "init" frame: replay how the task loss reshapes the SR
# output. 0 (default) = off. Storage at every-2 x 100 epochs: SR4RS ~2.3 GB,
# SEN2SR ~0.1 GB. Frozen/bicubic arms skip automatically.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-0}"

# LABELS -> dataset dir + code-level mask_source
case "$LABELS" in
  cdngi)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
    MASK_SOURCE="graph" ;;   # = the CDNGI dataset's own masks_graph parquets
  overture)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_Overture}"
    MASK_SOURCE="graph" ;;   # = the Overture dataset's own masks_graph parquets
  osm)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_Dense_CDNGI}"
    MASK_SOURCE="raster"
    MASK_DIRNAME="${MASK_DIRNAME:-mask_osm_2pt5}" ;;  # OSM HR rasters
  all)
    DATASET_DIR="${DATASET_DIR:-/scratch/${USER_NAME}/InstaRoad/ROSA_all}"
    MASK_SOURCE="graph" ;;   # = the all dataset's own masks_graph parquets
  *)
    echo "ERROR: LABELS must be cdngi|overture|osm|all, got '${LABELS}'." >&2; exit 2 ;;
esac
MASK_DIRNAME="${MASK_DIRNAME:-}"   # empty for the graph (on-the-fly) sources

# --- Tune budget -------------------------------------------------------------
N_TRIALS="${N_TRIALS:-60}"
SEARCH_GPUS="${SEARCH_GPUS:-2}"
TUNE_EPOCHS="${TUNE_EPOCHS:-15}"
PATIENCE="${PATIENCE:-3}"
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-imagenet}"
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-1e-2}"
LR_SR_MIN="${LR_SR_MIN:-1e-7}"   # searched only when UPSAMPLER=sen2sr && !FREEZE_SR
# Tightened from 1e-3 (2026-08-13): a sampled 3.1e-4 destroyed the SR front-end
# within 1,400 steps. See _stages_tv.sh for the full reasoning, incl. why lr and
# lr_sr stay independently sampled rather than reparametrised as a ratio.
LR_SR_MAX="${LR_SR_MAX:-1e-4}"
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-1.0}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-15.0}"
ENCODERS="${ENCODERS:-resnet34}"      # NOT searched: encoder constancy is the control
BATCH_SIZES="${BATCH_SIZES:-4}"       # PINNED, not searched (2026-08-12). `length` is
                                      # fixed per epoch, so a bs=1 trial takes 4x the
                                      # optimiser steps of a bs=4 trial and wins the
                                      # tune on step count alone -- batch size is a
                                      # confound, not a hyperparameter. 4 is a
                                      # between-arm constant for the whole SR series.
                                      # NB never change this on a RESUME_FIT: it
                                      # changes steps/epoch and breaks cosine T_max.

# Loader workers per training process: split the job's CPU allocation across
# the stage's processes (search fans out SEARCH_GPUS tuners; fit/test run
# one). See _stages_tv.sh for the rationale.
if [ -z "${NUM_WORKERS}" ]; then
  JOB_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-4}}"
  if [ "${STAGE}" = "tune" ]; then
    NUM_WORKERS=$(( JOB_CPUS / SEARCH_GPUS ))
  else
    NUM_WORKERS=$(( JOB_CPUS - 1 ))
  fi
  [ "${NUM_WORKERS}" -lt 1 ] && NUM_WORKERS=1
fi

# --- Fit budget --------------------------------------------------------------
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_joint}"

# --- Loss (unet.losses.build_loss; empty = legacy Dice + pos-weighted BCE) ---
LOSS_ARM="${LOSS_ARM:-wbce}"
PSTAR="${PSTAR:-bce}"
GAP_R="${GAP_R:-4}";                 GAP_K="${GAP_K:-60.0}"
TL_ELL="${TL_ELL:-5}";               TL_THETA="${TL_THETA:-0.375}"
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
                  --tversky-alpha "$TVERSKY_ALPHA"
                  --cl-alpha "$CL_ALPHA" --cl-iters "$CL_ITERS"
                  --skel-w "$SKEL_W" --skel-radius "$SKEL_RADIUS"
                  --warmup-start "$WARMUP_START" --warmup-ramp "$WARMUP_RAMP")
  LOSS_ARGS_FIT=(--model.loss_arm "$LOSS_ARM" --model.pstar "$PSTAR"
                 --model.gap_r "$GAP_R" --model.gap_k "$GAP_K"
                 --model.tl_ell "$TL_ELL" --model.tl_theta "$TL_THETA"
                 --model.tversky_alpha "$TVERSKY_ALPHA"
                 --model.cl_alpha "$CL_ALPHA" --model.cl_iters "$CL_ITERS"
                 --model.sr_w "$SKEL_W" --model.sr_radius "$SKEL_RADIUS"
                 --model.warmup_start "$WARMUP_START" --model.warmup_ramp "$WARMUP_RAMP")
fi
# =============================================================================

BASE_CONFIG="$REPO_DIR/src/sr/configs/joint_sr.yaml"
NORM_CONFIG="$REPO_DIR/src/unet/configs/norm_stats.yaml"
WANDB_CONFIG="$REPO_DIR/src/unet/configs/wandb.yaml"
RUN_DIR="/scratch/${USER_NAME}/InstaRoad/runs/sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}_seed${SEED}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"
echo "host=$(hostname)  exp=sr/${EXP_TAG}  stage=${STAGE}  seed=${SEED}"
echo "labels=${LABELS} (mask_source=${MASK_SOURCE})  upsampler=${UPSAMPLER}  freeze_sr=${FREEZE_SR}  sr_pad=${SR_PAD}  loss_arm=${LOSS_ARM:-legacy}"
echo "recipe: reg=${REG}${REG_TAG:+ [${REG_TAG}]}  clip=${CLIP}  lr_schedule=${LR_SCHEDULE}  sr_warmup_epochs=${SR_WARMUP_EPOCHS}  l2sp_lambda=${L2SP_LAMBDA}  sr_snapshot_every=${SR_SNAPSHOT_EVERY}"
echo "DATASET_DIR=${DATASET_DIR}  warm_start=${WARM_START_CKPT:-none}"

# --- Fail fast ---------------------------------------------------------------
if [ ! -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  exit 1
fi
if [ ! -f "${NORM_CONFIG}" ]; then
  echo "ERROR: ${NORM_CONFIG} missing — generate with sentinel2data.cli norm-stats." >&2
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
  echo "  (frozen-SR) arm's STAGE=fit first; its best ckpt seeds this arm's UNet." >&2
  exit 1
fi
if [ "${MASK_SOURCE}" = "raster" ]; then
  # -print -quit: no pipe to `head`, so `find` can't die of SIGPIPE and trip
  # `set -o pipefail` (that silently killed the unet osm.sh check).
  first_mask=$(find "${DATASET_DIR}"/*/"${MASK_DIRNAME}" -maxdepth 1 -name '*.tif' -print -quit 2>/dev/null)
  if [ -z "${first_mask}" ]; then
    echo "ERROR: MASK_SOURCE=raster but no masks under <split>/${MASK_DIRNAME}/." >&2
    if [ "${LABELS}" = "osm" ]; then
      echo "  Generate with OpenStreetMapTest/dataset_hr_masks.py --scale 4" >&2
    else
      echo "  Generate ONCE with (standalone, login node is fine):" >&2
      echo "    ${VENV_DIR}/bin/python ${REPO_DIR}/src/sentinel2data/dataset/rasterize_hr_masks.py \\" >&2
      echo "      --dataset-dir ${DATASET_DIR} --out-dirname ${MASK_DIRNAME}" >&2
    fi
    exit 1
  fi
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Long Optuna loops build/tear down models in ONE process; expandable segments
# let the allocator reclaim freed blocks of any size instead of fragmenting
# (r3's OOMs showed 100s of MiB "reserved but unallocated"). Pre-set the var
# to override.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "python=$(which python)"

# The full (Mamba) SEN2SR needs the CUDA-built mamba_ssm package.
if [ "${UPSAMPLER}" = "sen2sr_full" ] && ! python -c "import mamba_ssm" 2>/dev/null; then
  echo "ERROR: upsampler=sen2sr_full but mamba_ssm is not importable in ${VENV_DIR}." >&2
  echo "  Install on a GPU node with matching torch/CUDA:  uv pip install mamba-ssm" >&2
  exit 1
fi

# ============================== STAGE: tune ==================================
if [ "$STAGE" = "tune" ]; then
  # Default: sqlite (fine for ONE job/node). To run TWO sbatch jobs on the
  # same study concurrently (different nodes), BOTH must use the NFS-safe
  # journal backend and the 2nd job must offset its sampler seeds:
  #   job1: STORAGE=journal://<runs dir>/study.journal
  #   job2: STORAGE=journal://<same path> SAMPLER_OFFSET=500
  STORAGE="${STORAGE:-sqlite:///${RUN_DIR}/study.db}"
  SAMPLER_OFFSET="${SAMPLER_OFFSET:-0}"
  STUDY_NAME="sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}_seed${SEED}"

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

  # Cap the fan-out at the GPUs actually visible in THIS allocation. A worker
  # pinned to a nonexistent ordinal (CUDA_VISIBLE_DEVICES=1 on a 1-GPU job)
  # masks CUDA entirely: eager-CUDA loaders (sen2sr_full's mlstac card) die
  # with "No CUDA GPUs are available", everything else silently trains on CPU.
  # 0 GPUs is NOT an error: one unpinned CPU worker, so interactive CPU-only
  # smoke tests (run until epoch 1, then kill) keep working when the cluster
  # is busy. NB mamba_ssm's kernels are CUDA-only, so an r3/sen2sr_full smoke
  # test now loads fine on CPU but still dies at the first batch.
  N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
  if [ "${N_GPUS}" -eq 0 ] && [ "${SEARCH_GPUS}" -gt 1 ]; then
    echo "WARN: no CUDA device visible — running ONE unpinned tuner on CPU (smoke-test mode)." >&2
    SEARCH_GPUS=1
  elif [ "${N_GPUS}" -gt 0 ] && [ "${SEARCH_GPUS}" -gt "${N_GPUS}" ]; then
    echo "WARN: SEARCH_GPUS=${SEARCH_GPUS} but only ${N_GPUS} GPU(s) visible — capping to ${N_GPUS}." >&2
    SEARCH_GPUS="${N_GPUS}"
  fi

  echo "=== OPTUNA SEARCH (n_trials=$N_TRIALS across ${SEARCH_GPUS} GPU(s), ${TUNE_EPOCHS} epochs/trial) ==="
  echo "    stop early (keeps study + writes overlay):  touch ${RUN_DIR}/STOP"
  # Sampler seeds: SEED*1000+worker, so workers within a run differ (no
  # duplicate proposals) AND no sampler seed ever recurs across SEED runs
  # (SEED+g would make e.g. SEED=0/worker1 collide with SEED=1/worker0,
  # correlating the startup trials of nominally independent runs).
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
  echo "Next: sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/${EXP_TAG}.sh STAGE=fit SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}"
  exit 0
fi

# ============================== STAGE: bench =================================
# Score the fitted checkpoint into the shared benchmark store, evaluated at
# 2.5 m against the experiment's own GT (LABELS -> mask_source), through the
# same joint_sr_dataset helpers training used — eval GT cannot drift from
# train GT. --sen2sr-dir overrides the training node's path baked into hparams.
if [ "$STAGE" = "bench" ]; then
  CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_jointsr_best.ckpt"
  if [ ! -f "$CKPT" ]; then
    if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
      echo "WARN: best checkpoint missing; benchmarking last.ckpt instead." >&2
      CKPT="${RUN_DIR}/checkpoints/last.ckpt"
    else
      echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — run STAGE=fit first." >&2
      exit 1
    fi
  fi

  # benchmarks_newdata, NOT benchmarks: the ROSA_New TEST split was manually
  # relabelled in place on 2026-09-07 (7 tiles dropped, 181 -> 174, and 92 of
  # the survivors changed). Rows scored before and after are DIFFERENT
  # QUANTITIES, and nothing in the runs table separates them -- dataset_dir,
  # mask_dirname, mask_source, gt_res_m and cell_m are all identical because
  # the dataset path was reused, so `report` would average the two label sets
  # into one mean without complaint. The old store stays readable at
  # .../benchmarks; the pre-relabelling rows are still valid among themselves.
  # Train and val were untouched, so fits, tunes and theta* all still stand.
  STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_newdata}"   # SHARED across experiments
  MODEL_NAME="${MODEL_NAME:-sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}}"  # {family}_{exp}[_{loss}][_noreg]: what the stats pair/group on
  LABEL_SOURCE="${LABEL_SOURCE:-${LABELS}}"      # cdngi | overture | osm
  BENCH_SPLIT="${BENCH_SPLIT:-test}"
  TILE_METRICS="${TILE_METRICS:-apls,cldice}"    # comma-separated plugins; '' disables.
                                                 # apls is the resolution-robust
                                                 # cross-family comparison metric;
                                                 # cldice is its connectivity
                                                 # companion. Both are MACRO-only.
  BUFFER_PX="${BUFFER_PX:-1,2,3,4,5}"            # buffered P/R/F1 tolerance sweep; '' disables
  AP_BINS="${AP_BINS:-101}"                      # per-chip AP (AUPRC) bins; '' disables

  CONFIG_ARGS=()
  [ -f "${RUN_DIR}/best_params.yaml" ] && CONFIG_ARGS=(--config-yaml "${RUN_DIR}/best_params.yaml")
  MASK_ARGS_BENCH=(--mask-source "$MASK_SOURCE")
  [ "$MASK_SOURCE" = "raster" ] && MASK_ARGS_BENCH+=(--mask-dirname "$MASK_DIRNAME")
  METRIC_ARGS=()
  if [ -n "${TILE_METRICS}" ]; then
    IFS=',' read -r -a _TMS <<< "${TILE_METRICS}"
    for _tm in "${_TMS[@]}"; do METRIC_ARGS+=(--tile-metric "${_tm}"); done
  fi

  echo "=== BENCH (ckpt=$(basename "$CKPT"), model_name=${MODEL_NAME}, seed=${SEED}, gt=${MASK_SOURCE}, tile_metrics=${TILE_METRICS:-none}) ==="
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
    ${BUFFER_PX:+--buffer-px "$BUFFER_PX"} \
    ${AP_BINS:+--ap-bins "$AP_BINS"} \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    "${MASK_ARGS_BENCH[@]}"

  echo "=== BENCH DONE ===  store: ${STORE_DIR}"
  echo "Report: python -m benchmarking.cli report --store-dir ${STORE_DIR} \\"
  echo "          --metric f1 --metric iou --metric ap --metric cldice --metric apls --aggregation both"
  exit 0
fi

# ============================== STAGE: fit ===================================
if [ "$STAGE" != "fit" ]; then
  echo "ERROR: STAGE must be tune, fit or bench, got '${STAGE}'." >&2
  exit 2
fi

BEST_CONFIG="${RUN_DIR}/best_params.yaml"
CKPT="${RUN_DIR}/checkpoints/unet_s2rosa_jointsr_best.ckpt"
if [ ! -f "$BEST_CONFIG" ]; then
  echo "ERROR: ${BEST_CONFIG} not found — run STAGE=tune first." >&2
  exit 1
fi
echo "--- best hyperparameters ---"; cat "$BEST_CONFIG"

# Refit from inside RUN_DIR so the base config's relative `checkpoints/` lands here.
cd "$RUN_DIR"

# Refit from scratch by DEFAULT (a stale last.ckpt in the run dir is ignored,
# then overwritten). RESUME_FIT=1 instead continues a previous refit from its
# last.ckpt (e.g. a walltime-killed job): LightningCLI `fit --ckpt_path`
# restores the epoch, optimizer, LR schedule AND the ModelCheckpoint best-score
# state, so the run finishes the remaining epochs with tracking intact. (Same
# run dir = same exp/seed/loss/reg treatment, so a resumed checkpoint's config
# can't mismatch the overlay.)
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

# The experiment's SR treatment (and loss arm, if set) is passed explicitly
# (belt) even though the best_params overlay records it too (braces) — drift
# is impossible.
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

echo "=== REFIT (best config, ${REFIT_EPOCHS} epochs, ${REFIT_GPUS} GPU) ==="
python -m sr.cli fit \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
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

# Prefer the best checkpoint; fall back to last.ckpt; fail loudly otherwise.
if [ ! -f "$CKPT" ]; then
  if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
    echo "WARN: best checkpoint missing; testing last.ckpt instead." >&2
    CKPT="${RUN_DIR}/checkpoints/last.ckpt"
  else
    echo "ERROR: no checkpoint under ${RUN_DIR}/checkpoints/ — refit produced none. Skipping test." >&2
    exit 1
  fi
fi

echo "=== BENCHMARK (test split, ckpt=$(basename "$CKPT")) ==="
python -m sr.cli test \
  --config "$BASE_CONFIG" \
  --config "$NORM_CONFIG" \
  --config "$WANDB_CONFIG" \
  --config "$BEST_CONFIG" \
  --data.dataset_dir "$DATASET_DIR" \
  --data.num_workers "$NUM_WORKERS" \
  --data.mask_source "$MASK_SOURCE" \
  "${MODEL_ARGS[@]}" \
  --trainer.devices 1 \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --ckpt_path "$CKPT"

echo "=== DONE ===  outputs in $RUN_DIR"
