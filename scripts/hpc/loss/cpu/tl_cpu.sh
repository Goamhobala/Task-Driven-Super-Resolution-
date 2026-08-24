#!/bin/bash
# CPU-ONLY fit + bench for one (TL compound arm, seed). Runs the same engine the
# GPU refits use, so the resulting weights and rows are protocol-identical.
#
#   cd scripts/hpc
  # sbatch --job-name=tl_sdice_s1 --gres=gpu:0 --cpus-per-task=8 --time=48:00:00 \
  #        --export=ALL,ARM=pstar_sdice,SEED=1 \
  #        train.sbatch --SCRIPT=loss/cpu/tl_cpu.sh
#
#   ARM  = pstar_dice | pstar_sdice | pstar_lcdice     (default pstar_sdice)
#   SEED = 1 | 2 | 3                                   (default 1)
#
# WHAT IT DOES, IN ORDER
#   fit (resumable)  ->  bench val (writes theta*)  ->  bench test (same theta*)
# Every step is skipped if already done, so re-submitting continues rather than
# restarting. That matters here more than anywhere else -- see the wall clock
# note below.
#
# 48 h IS PROBABLY NOT ENOUGH FOR ONE FIT
# ---------------------------------------
# 50 epochs is ~4.2 h on an L40S. Training is GPU-bound, so on 8 cores expect
# 20-50x that: 3.5-8 DAYS. The fit stage checkpoints every epoch and RESUME_FIT
# picks up from last.ckpt, so the intended use is to submit this repeatedly
# until `ck['epoch'] == 49`. Check with:
#   python -c "import torch;print(torch.load('<run>/checkpoints/last.ckpt',
#              map_location='cpu',weights_only=False)['epoch'])"
# The BENCH stages, by contrast, are ~86% serial CPU anyway and finish in
# ordinary time on this hardware -- they are the part that genuinely belongs on
# CPU nodes.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
USER_NAME="${USER:-$(whoami)}"
RUNS_ROOT="${RUNS_ROOT:-/scratch/${USER_NAME}/InstaRoad/runs}"

ARM="${ARM:-pstar_sdice}"
SEED="${SEED:-1}"
EXP_TAG="r0_new_tl"

case "$ARM" in
  pstar_dice)   MODEL_NAME="sr_r0_new_tl_ce_dice_holdout" ;;
  pstar_sdice)  MODEL_NAME="sr_r0_new_tl_ce_sdice_holdout" ;;
  pstar_lcdice) MODEL_NAME="sr_r0_new_tl_ce_lcdice_holdout" ;;
  *) echo "ERROR: ARM must be pstar_dice|pstar_sdice|pstar_lcdice, got '$ARM'" >&2; exit 2 ;;
esac
RUN_DIR="${RUNS_ROOT}/sr_${EXP_TAG}_${ARM}_holdout_seed${SEED}"
mkdir -p "$RUN_DIR"

# --- baked best_params, one per arm -----------------------------------------
# Planted rather than looked up: a seed-N dir has never been tuned, and the
# tune lives on a Modal volume this cluster cannot reach. Values are each
# arm's own Optuna result at seed 0; pos_weight/tl_theta/gap_theta are the
# tl_ce parent's, pinned by Phase B and NOT re-searched.
plant () {
cat > "$RUN_DIR/best_params.yaml" <<YAML
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  upsampler: bicubic
  freeze_sr: false
  sr_pad: 0
  lr: ${LR}
  loss_arm: ${ARM}
  pstar: tl_ce
  gap_r: 4
  gap_k: 60.0
  tl_ell: 5
  tl_theta: 0.5411053504286575
  gap_theta: 0.5
  tversky_alpha: 0.7
  cl_alpha: 0.3
  cl_iters: 5
  sr_w: 1.0
  sr_radius: 1
  warmup_start: 15
  warmup_ramp: 5
  mix_w: ${MIX_W}
  pos_weight: 13.988665008276607
  lr_schedule: cosine
  sr_warmup_epochs: 1.0
  l2sp_lambda: 0.0
  adaptive_norm: false
  adaptive_norm_momentum: 0.01
  norm_recalibrate: 'off'
data:
  batch_size: 8
  mask_source: raster
  mask_dirname: mask_new_2pt5
trainer:
  precision: 32-true
YAML
}
case "$ARM" in
  pstar_dice)   LR=0.0003066390823976961 ; MIX_W=0.7147279908608887 ;;  # val_iou 0.3187
  # PROVISIONAL -- taken from the leading trial at 15/30, while both tunes were
  # still running. Good enough to exercise the pipeline; NOT the value to fit a
  # reported seed with. Both currently point at the same draw (trial #6), which
  # is what a half-finished TPE search looks like, not a real coincidence.
  # Re-bake from best_params.yaml once the searches finish.
  pstar_sdice)  LR=0.0005059803874660431 ; MIX_W=0.7127983191463305 ;;  # val_iou 0.3201 @15/30
  pstar_lcdice) LR=0.0005059803874660431 ; MIX_W=0.7127983191463305 ;;  # val_iou 0.2952 @15/30
esac
case "$LR" in __*) echo "ERROR: ${ARM} params not baked in yet" >&2; exit 3 ;; esac
# Loud, because a provisional-params run must not be mistaken for a reported one.
case "$ARM" in
  pstar_sdice|pstar_lcdice)
    echo "WARNING: ${ARM} is using PROVISIONAL params from a 15/30-trial tune." >&2
    echo "         Fine for a pipeline test; re-bake before fitting a reported seed." >&2 ;;
esac
plant

# --- CPU execution shape -----------------------------------------------------
# joint_sr.yaml sets accelerator:auto, so Lightning picks CPU with no GPU
# visible. Precision MUST leave bf16-mixed: CPU autocast is not the same code
# path and is slower than plain fp32 here.
export PRECISION="32-true"
export REFIT_GPUS=1 SEARCH_GPUS=1
JOB_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
export NUM_WORKERS="$(( JOB_CPUS > 1 ? JOB_CPUS - 1 : 1 ))"
export OMP_NUM_THREADS="$JOB_CPUS"

# --- pilot protocol, identical to the GPU refits -----------------------------
export TRAIN_SPLITS="train"       # val stays a genuine holdout
export REFIT_EPOCHS="${REFIT_EPOCHS:-50}"
export SKIP_TEST=1                # fit must not read test; bench does it explicitly
export TILE_METRICS="${TILE_METRICS:-apls,cldice}"
export BUFFER_PX="${BUFFER_PX:-1,2,3,4,5}"
export AP_BINS="${AP_BINS:-101}"
export SELECT_ON="${SELECT_ON:-iou_mean}"
export UPSAMPLER="bicubic" FREEZE_SR="false" SR_PAD=0
export ADAPTIVE_NORM=0 NORM_RECALIBRATE="off"
export LABELS="${LABELS:-new}"
# wandb: same treatment as the GPU refits (scripts/hpc/loss/refit/_refit_arm.sh).
# The cluster HAS a working key -- those refits log online -- so do NOT force
# offline here; that would leave these fits as the only ones with no dashboard.
# WANDB_NAME matters because without it every run lands as a random
# adjective-animal and telling seed 1 from seed 2 means opening each one.
# Set WANDB_MODE=offline explicitly if a node has no outbound network.
export WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_loss_pilot_seeds}"
export WANDB_NAME="${WANDB_NAME:-${EXP_TAG}_${ARM}_holdout_seed${SEED}_cpu}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${EXP_TAG}_${ARM}_holdout}"
export STORE_DIR="${STORE_DIR:-/scratch/${USER_NAME}/InstaRoad/benchmarks_test3}"
export MODEL_NAME EXP_TAG SEED
export LOSS_ARM="$ARM"

CKPT="$RUN_DIR/checkpoints/unet_s2rosa_jointsr_final.ckpt"
in_store () {  # split -> 0 if that row already exists
  python - "$STORE_DIR" "$MODEL_NAME" "$SEED" "$1" <<'PYEOF' 2>/dev/null
import sys
from pathlib import Path
store, name, seed, split = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
try:
    from benchmarking.store import load_runs
    runs = load_runs(Path(store))
except Exception:
    sys.exit(1)
if runs is None or getattr(runs, "empty", True) or "model_name" not in runs.columns:
    sys.exit(1)
hit = runs[(runs["model_name"] == name) & (runs["seed"] == seed)]
if "dataset_split" in runs.columns:
    hit = hit[hit["dataset_split"] == split]
sys.exit(0 if len(hit) else 1)
PYEOF
}

echo "=============================================================="
echo "  arm=${ARM}  seed=${SEED}  model=${MODEL_NAME}"
echo "  run_dir=${RUN_DIR}"
echo "  cpus=${JOB_CPUS}  workers=${NUM_WORKERS}  precision=${PRECISION}"
echo "=============================================================="

if [ -f "$CKPT" ] && [ -f "$RUN_DIR/sweep.json" ]; then
  echo "########## FIT already complete — skipping ##########"
else
  echo "########## FIT (resumable; re-submit until epoch 49) ##########"
  env STAGE=fit RESUME_FIT=1 bash "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh" \
    || { echo "fit did not finish in this job — re-submit to continue" >&2; exit 0; }
fi

for SPLIT in val test; do
  if in_store "$SPLIT"; then
    echo "########## BENCH ${SPLIT} already in store — skipping ##########"
  else
    echo "########## BENCH ${SPLIT} ##########"
    env STAGE=bench BENCH_SPLIT="$SPLIT" bash "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
  fi
done
echo "=== ${MODEL_NAME} seed${SEED} done ==="
