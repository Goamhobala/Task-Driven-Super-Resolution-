#!/bin/bash
# RL3 — FROZEN SR4RS x4 under a LINEAR PROBE read-out, BARE. HPC (l40s).
# Cluster twin of scripts/LightningStudio/sr/rl/rl3.sh: same campaign constants
# (_rl_common.sh), same engine (generated from this one), same store tags —
# EXCEPT that this arm early-stops (see below), which its _es5 tag records.
#
# The write-up ladder's r3 is the run-tag r5 (tags never change; the store is
# append-only and the thesis carries one mapping table). rl3 is its probe twin:
# rl3 - rl0 = the separability a frozen WGAN-GP generator adds.
#
# COST WARNING. "Frozen" does not mean cheap here. The probe is 5 parameters,
# but the arm still runs an 11.3 M-param SR4RS forward at 512 px on every batch,
# so rl3 costs roughly what r5 costs — replacing the decoder saves the U-Net's
# forward+backward, not the SR front-end, and the front-end dominates. It is
# also I/O-bound (five trainable parameters), so ask for CPUs: 8+.
#
# SR4RS ships no FFT hard constraint, so SR_HC=off is its native behaviour;
# it is forced explicitly anyway so that all five arms carry the same _nohc tag
# and no arm's constraint state is implicit.
#
# EARLY STOPPING — ON, patience 5 on val_ap (the campaign's own monitor).
# The Studio arms run the fixed 30-epoch budget; this one stops when the probe
# plateaus, which the campaign PRE-REGISTERED as happening before epoch 10
# ("frozen val-AP must plateau before epoch 10", plan §6.1). It is affordable
# here for the same reason it is safe: a frozen generator cannot drift, so
# nothing this arm measures is still changing after the plateau — the only
# thing the remaining epochs buy is queue time on the most expensive frozen arm
# in the series.
#
# Three consequences, all of them stated rather than discovered:
#   1. the budget is a CEILING for this arm, not the between-arm constant it is
#      everywhere else. `rl4 - rl3` therefore compares a 30-epoch joint run
#      against a stopped-at-plateau frozen one. Defensible only while the
#      plateau claim holds — CHECK IT on the logged val_ap curve, and if the
#      arm was still climbing when it stopped, rerun it with FIT_EARLY_STOP=0
#      rather than reporting the contrast.
#   2. the cosine does not complete: the run ends part-way down the schedule at
#      a non-zero lr, instead of at lr 0 like every other arm.
#   3. rows land under ..._es5_holdout_..., which is what keeps them out of the
#      Studio's fixed-budget rl3 rows. Deliberate: they are not the same
#      protocol and must not be averaged.
# FIT_EARLY_STOP=0 at submit time reverts all three and reproduces the Studio
# arm exactly (its rows then merge with the Studio's, which is the point).
#
# Prerequisite: gen_*.{safetensors,json,npz} in SEN2SR_DIR. Parity-verify the
# port locally first (`python -m sr.sr4rs_torch`) — there is no TF on the
# cluster.
#
# ONE SUBMISSION, START TO FINISH. This arm searches nothing, so it carries its
# own best_params.yaml (see THE OVERLAY below) and the engine plants it; the fit
# then chains straight into test -> θ* sweep -> bench (CHAIN_BENCH=1):
#
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=24:00:00 \
#          scripts/hpc/train.sbatch --SCRIPT=sr/rl/rl3.sh
#
# Budget the walltime for BOTH — the bench is minutes next to the fit's hours,
# but a job that dies at the wall clock after training loses the bench with it
# (re-run it alone with STAGE=bench; the checkpoint is on disk).
#
# The plan's §4 gate-1 measurement (h/epoch, peak VRAM at bs=4) now comes off
# the fit itself: read the first epoch out of the log and `scancel` if the
# walltime was wrong. For a throwaway probe that cannot touch this arm's run dir
# or its rows:
#   sbatch ... --SCRIPT=sr/rl/rl3.sh EXP_TAG=rl3_probe REFIT_EPOCHS=1 SKIP_TEST=1
# (SKIP_TEST=1 also stops the chain, so a probe never benches or touches test.)
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"

EXP_TAG="${EXP_TAG:-rl3_new}"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

# ONE JOB, NOT TWO: the fit is followed by the test split, the θ* sweep and the
# bench, in this allocation. Nothing between them needs a queue slot, and the
# bench is minutes against the fit's hours — two submissions only bought a
# second wait. STAGE defaults to `fit` for the same reason: there is no tune to
# be the natural first stage (see THE OVERLAY below).
#   CHAIN_BENCH=0  stop after the fit + sweep and bench separately later.
#   STAGE=bench    re-bench an existing run dir on its own (no chaining then).
STAGE="${STAGE:-fit}"
CHAIN_BENCH="${CHAIN_BENCH:-1}"
# There is no tune stage here, so a STAGE this arm cannot honour is refused
# LOUDLY instead of costing an epoch. Two ways one arrives without being typed
# on the sbatch line, both silent before this guard existed:
#   * SLURM exports the SUBMITTING shell's environment by default (--export=ALL),
#     so a leftover `export STAGE=tune` in the login shell reaches the job and
#     beats the ${STAGE:-fit} default above;
#   * an older checkout of this arm, where STAGE had no default and the engine's
#     own default (tune) applied — check `git log -1` on the cluster if you see
#     this after a pull.
# Either way the job would run a 1x1 Optuna pass, write an overlay this arm
# already carries, and exit — an epoch of SR4RS for nothing.
case "$STAGE" in
fit | bench) ;;
*)
  echo "ERROR: STAGE='${STAGE}' — this arm has no such stage." >&2
  echo "  It searches NOTHING: the head lr, the loss, λ, the batch size (and the" >&2
  echo "  rung's lr_sr) are pinned, and it carries its own best_params.yaml, so" >&2
  echo "  STAGE=tune would spend an epoch producing a file that already exists." >&2
  echo "  Submit with NO STAGE at all (fit -> test -> θ* sweep -> bench), or" >&2
  echo "  STAGE=bench to re-bench a finished run dir." >&2
  echo "  If you did not pass STAGE, it came from your shell: SLURM exports the" >&2
  echo "  submitting environment. Check with 'echo \$STAGE' on the login node," >&2
  echo "  then 'unset STAGE' (or submit with --export=NONE)." >&2
  exit 2
  ;;
esac
echo "[rl] stage=${STAGE}  chain_bench=${CHAIN_BENCH}"

# The one place this arm departs from its Studio twin (see EARLY STOPPING).
# Set for EVERY stage: ES_TAG is in the run dir, so a tune tagged one way and a
# fit the other would look for best_params.yaml in a directory that does not
# exist.

FIT_EARLY_STOP="${FIT_EARLY_STOP:-1}"
ES_PATIENCE="${ES_PATIENCE:-5}"
ES_MONITOR="${ES_MONITOR:-$MONITOR}"   # val_ap — never val_iou@0.5 (probe doc §5.1)
ES_MODE="${ES_MODE:-max}"

# --- THE OVERLAY, PLANTED (no tune stage) ------------------------------------
# This arm searches NOTHING: the head lr, the loss, λ, the batch size are
# pinned constants (_rl_common.sh). Its "tune" would have been one trial of
# one epoch whose only product was this file of values it was handed, so the
# file is written directly and STAGE=tune is skipped entirely. The engine plants
# it into RUN_DIR at the fit/bench stages.
#
# THIS MUST STAY BYTE-EQUIVALENT TO WHAT sr.tune WOULD HAVE WRITTEN.
# `sr.tune.write_best_overlay` owns the schema; this heredoc is a second author,
# which is a drift hazard — so it is pinned by a test rather than by eye:
# tests/test_rl_campaign_hpc.py::test_the_planted_overlay_is_what_a_tune_would_have_written
# rebuilds it through that function with these constants and compares. If you
# change a constant above, run that test; if it fails, the overlay is stale.
#
# What each key is doing here (the rest is belt — the engine re-passes it as
# --model.* AFTER the config layers, so those keys cannot drift):
#   lr, pos_weight, batch_size, precision   NOT in the fit belt. The overlay is
#       the only place they come from — drop one and the fit silently takes
#       joint_sr.yaml's default instead.
#   encoder_name/encoder_weights: null            head=linear builds no U-Net;
#       writing resnet34/imagenet would let this run be read back as an encoder
#       ablation of a network that was never constructed.
#   sr_hold_epochs is absent: a frozen arm has no SR group to hold.
read -r -d '' BEST_PARAMS <<'YAML' || true
# Planted by scripts/hpc/sr/rl/rl3.sh — this arm searches nothing (no tune stage).
# Equivalent to sr.tune.write_best_overlay's output for the pinned constants;
# pinned by tests/test_rl_campaign_hpc.py.
model:
  encoder_name: null
  encoder_weights: null
  upsampler: sr4rs
  freeze_sr: true
  sr_pad: 0
  lr: 0.003
  sr_hc: 'off'
  head: linear
  clip_sr: 1.0
  loss_arm: wbce
  pstar: bce
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
  pos_weight: 2.4789710497080004
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

source "$RL_DIR/../_stages_tv.sh"
