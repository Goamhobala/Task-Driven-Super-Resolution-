#!/bin/bash
# ONE CELL of the SR4RS lr_sr grid on Lightning: fit -> bench, resumable.
# Lightning twin of scripts/hpc/sr/grid/run_pool_r4.sh, for continuing a cell
# that started on the cluster, on an INTERRUPTIBLE machine.
#
#   bash scripts/LightningStudio/sr/grid/resume_r4grid.sh HC=on LRSR=1e-5
#
# Idempotent: rerun the same command after a preemption and it continues from
# the newest readable checkpoint; a finished cell costs seconds. The studio's
# .lightning_studio/on_start.sh calls this when <run dir>/AUTO_RESUME exists, so
# a restarted interruptible studio picks the fit back up on its own.
#
# TWO THINGS THE CLUSTER DRIVER NEVER HAD TO HANDLE
# -------------------------------------------------
# 1. Versioned last checkpoints. Lightning does not overwrite an existing
#    last.ckpt when a fit resumes — it writes last-v1.ckpt and leaves last.ckpt
#    frozen at the resume epoch. One resume is harmless; a SECOND resume from
#    last.ckpt would silently rewind to the first resume point. Every attempt
#    therefore promotes the newest readable checkpoint to last.ckpt first.
# 2. Torn writes. A preemption mid-save leaves a truncated file. "Newest" means
#    the highest epoch that torch can actually load, so a torn last-v1.ckpt
#    loses one epoch rather than the run.
set -euo pipefail
# Set BEFORE env.sh, whose defaults would otherwise win: NUM_WORKERS=0 (a
# DDP-era guard that starves the GPU; the cluster fit ran 7 loader workers) and
# WANDB_MODE=online (a machine without a wandb key dies at logger init). With
# no key, runs land offline under <run dir>/wandb; `wandb sync` them afterwards.
# The key comes from the teamspace secret WANDB_API_KEY.
export NUM_WORKERS="${NUM_WORKERS:-7}"
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"

for kv in "$@"; do case "$kv" in *=*) export "$kv" ;; esac; done
HC="${HC:?set HC=on|off}"
LRSR="${LRSR:?set LRSR=<lr_sr>}"
SEED="${SEED:-0}"
LR="${LR:-0.0002}"
FIT_EPOCHS="${FIT_EPOCHS:-${REFIT_EPOCHS:-100}}"
CELL_SCRIPT="$LS_DIR/sr/r4grid_new.sh"
HC_MASK="${HC_MASK_PATH:-${INSTAROAD_ROOT}/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"
# ckpt_epoch / fit_state / in_store: pure functions, shared with the cluster.
source "$REPO_DIR/scripts/hpc/sr/refit/_refit_lib.sh"
# _refit_lib's helpers call a bare `python`; they need the venv's torch.
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"

NAMES=$(env HC="$HC" LRSR="$LRSR" SEED="$SEED" PRINT_RUN_DIR=1 bash "$CELL_SCRIPT" 2>/dev/null)
RUN_DIR=$(printf '%s\n' "$NAMES" | sed -n 's/^RUN_DIR=//p')
MODEL_NAME=$(printf '%s\n' "$NAMES" | sed -n 's/^MODEL_NAME=//p')
if [ -z "$RUN_DIR" ] || [ -z "$MODEL_NAME" ]; then
  echo "ERROR: could not resolve names for HC=${HC} LRSR=${LRSR}" >&2; exit 2
fi
STORE_DIR="${STORE_DIR:-${INSTAROAD_ROOT}/benchmarks_corrected}"
mkdir -p "$RUN_DIR/checkpoints"

# One driver per run dir: on_start.sh and a manual launch must not race.
exec 9>"$RUN_DIR/.resume.lock"
if ! flock -n 9; then
  echo "another resume_r4grid.sh already holds ${RUN_DIR}/.resume.lock — exiting" >&2
  exit 0
fi

if [ "$HC" = "on" ]; then SR_PAD=8; HC_LINE="  sr_hc: 'on'
  hc_mask_path: ${HC_MASK}"; else SR_PAD=0; HC_LINE="  sr_hc: 'off'"; fi

# The overlay is COMPOSED from the cell's coordinates, verbatim the heredoc in
# scripts/hpc/sr/grid/run_pool_r4.sh (only hc_mask_path's root differs, and the
# fit passes --model.hc_mask_path explicitly anyway).
cat > "$RUN_DIR/best_params.yaml" <<YAML
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  upsampler: sr4rs
  freeze_sr: false
  sr_pad: ${SR_PAD}
${HC_LINE}
  lr: ${LR}
  lr_sr: ${LRSR}
  loss_arm: gap_ce
  pstar: gap_ce
  gap_r: 4
  gap_k: 60.0
  tl_ell: 5
  tl_theta: 0.375
  gap_theta: 0.55836
  tversky_alpha: 0.7
  cl_alpha: 0.3
  cl_iters: 5
  sr_w: 1.0
  sr_radius: 1
  warmup_start: 30
  warmup_ramp: 10
  mix_w: 0.6075946831862098
  pos_weight: 4.77222
  lr_schedule: cosine
  sr_warmup_epochs: 1.0
  l2sp_lambda: 0.0
  adaptive_norm: true
  adaptive_norm_momentum: 0.01
  norm_recalibrate: post
  std_band_raise_lo: 0.01
  std_band_raise_hi: 100.0
data:
  batch_size: 4
  mask_source: raster
  mask_dirname: mask_new_2pt5
trainer:
  precision: bf16-mixed
YAML

# --- promote the newest readable checkpoint to last.ckpt ---------------------
# On a Lightning JOB the studio tree is a snapshot: a job's own writes land in a
# per-job layer, synced live and readable from LATER jobs (read-only) at
# /teamspace/jobs/<job>/artifacts/<path under this_studio>. A job resubmitted
# after a preemption therefore starts from the studio's stale checkpoint unless
# it also looks there — so every earlier job's copy of this run dir is a
# candidate too. (Verified 2026-09-11: a 50 MB file written by a running job was
# visible in the teamspace within ~30 s.)
CK="$RUN_DIR/checkpoints"
REL="${RUN_DIR#"${INSTAROAD_ROOT}"/}"
best="" best_ep=-1
for f in "$CK"/last.ckpt "$CK"/last-v*.ckpt "$CK"/unet_s2rosa_jointsr_final.ckpt \
  /teamspace/jobs/*/artifacts/"$REL"/checkpoints/last.ckpt \
  /teamspace/jobs/*/artifacts/"$REL"/checkpoints/last-v*.ckpt \
  /teamspace/jobs/*/artifacts/"$REL"/checkpoints/unet_s2rosa_jointsr_final.ckpt; do
  [ -f "$f" ] || continue
  ep=$(ckpt_epoch "$f")
  echo "checkpoint ${f}: epoch ${ep}"
  if [ "$ep" -gt "$best_ep" ]; then best="$f"; best_ep="$ep"; fi
done
if [ -n "$best" ] && [ "$best" != "$CK/last.ckpt" ]; then
  echo "promoting $(basename "$best") (epoch ${best_ep}) -> last.ckpt"
  cp "$best" "$CK/.last.ckpt.tmp" && mv -f "$CK/.last.ckpt.tmp" "$CK/last.ckpt"
fi
rm -f "$CK"/last-v*.ckpt

# A resume restores the checkpoint's hparams OVER the command line, cluster
# paths included — repoint them at this machine's weights (see the script).
if [ -f "$CK/last.ckpt" ]; then
  python "$LS_DIR/sr/grid/patch_ckpt_paths.py" "$CK/last.ckpt" \
    sen2sr_dir="${INSTAROAD_ROOT}/models/SR4RS_RGBN" hc_mask_path="$HC_MASK"
fi

state=$(fit_state "$RUN_DIR" "$((FIT_EPOCHS - 1))")
echo "### ${RUN_DIR##*/}  lr_sr=${LRSR}  newest epoch=${best_ep}  state=${state}"
case "$state" in
  done) echo "### FIT complete — skipping" ;;
  sweep_only | resume) echo "### RESUMING from last.ckpt" ;;
  sweep_only_nolast) echo "### weights complete, sweep.json and last.ckpt both gone" >&2 ;;
  fresh) echo "### no usable checkpoint — refusing to refit a cluster cell from scratch" >&2; exit 2 ;;
esac

# Online: continue the cell's ORIGINAL wandb run (the oldest run-<ts>-<id> dir,
# i.e. the cluster fit) so the curves stay one run across the cluster and every
# preemption. The logger steps by global_step, which only moves forward on a
# resume, so wandb accepts the appended points.
if [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_RUN_ID:-}" ]; then
  first_run=$(ls -d "$RUN_DIR"/wandb/run-*-* 2>/dev/null | sort | head -1)
  if [ -n "$first_run" ]; then
    export WANDB_RUN_ID="${first_run##*-}" WANDB_RESUME=allow
    echo "wandb: continuing run ${WANDB_RUN_ID} online"
  fi
fi

if [ "$state" = "sweep_only" ] || [ "$state" = "resume" ]; then
  # CHAIN_BENCH=0: the bench is driven below behind in_store, so a resumed
  # attempt that already benched cannot append a duplicate shard.
  env HC="$HC" LRSR="$LRSR" SEED="$SEED" STAGE=fit RESUME_FIT=1 CHAIN_BENCH=0 \
    STORE_DIR="$STORE_DIR" REFIT_EPOCHS="$FIT_EPOCHS" bash "$CELL_SCRIPT"
fi

if in_store "$STORE_DIR" "$MODEL_NAME" "$SEED" test; then
  echo "### BENCH already in the store — skipping (append-only, no dedupe)"
elif [ ! -f "${RUN_DIR}/sweep.json" ]; then
  echo "### no sweep.json — bench has no theta*, skipping" >&2
else
  env HC="$HC" LRSR="$LRSR" SEED="$SEED" STAGE=bench MODEL_NAME="$MODEL_NAME" \
    STORE_DIR="$STORE_DIR" bash "$CELL_SCRIPT"
fi
rm -f "$RUN_DIR/AUTO_RESUME"
echo "=== ${RUN_DIR##*/}: fit + bench done ==="
