#!/bin/bash
# RL2 — JOINT task-driven fine-tuning of SEN2SR-Lite under a LINEAR PROBE,
# BARE, ONE RUNG of the lr_sr ladder. HPC (l40s).
#
# Cluster twin of scripts/LightningStudio/sr/rl/rl2.sh — same constants, same
# ladder, same tags, so a rung run here and a rung run in the Studio land in the
# same store row and the same W&B project. Nothing about the platform changes
# what this arm is; it is here because the SR4RS row already is, and running the
# SEN2SR row beside it keeps one queue and one set of logs.
#
# ONE SUBMISSION PER RUNG, START TO FINISH: the rung is a pin, not a search, so
# this arm carries its own best_params.yaml (see THE OVERLAY below) and the
# engine plants it; the fit then chains into test -> θ* sweep -> bench.
#
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=08:00:00 --job-name=rl2e-3 \
#          train.sbatch --SCRIPT=sr/rl/rl2.sh LRSR=1e-3
#
# ALL FOUR RUNGS SHARE THIS SCRIPT; LRSR is the coordinate and it goes into
# EXP_TAG, so run dirs and benchmark rows are disjoint per rung by construction.
# See _rl_rung.sh for the ladder, the dose argument, the run order (extremes
# first: 1e-3 then 1e-6) and the 1e-3 contingency.
#
# CHEAPEST OF THE THREE CLUSTER ARMS. SEN2SR-Lite is 187 K trainable parameters
# against SR4RS's 11.3 M, and the Studio measured ~2m45s/epoch for it on an L4 —
# so 30 epochs is a couple of hours here, not rl4's ten. 08:00:00 leaves room
# for the chained bench; check the first epoch in the log and adjust.
#
# THE SHAPE OF ONE RUN (plan §2): 30 epochs, single job, no stage pairing.
# Epochs 1-10 run with lr_sr held at EXACTLY 0 — an LR gate on the SR parameter
# group, so Adam's update is identically zero while its moments warm on the real
# gradients. Epochs 11-30 run the rung's lr_sr on its own cosine. The hold phase
# is therefore a frozen-arm run, and rl1's epochs 11-30 are the matched-budget
# control for the joint phase.
#
# WHAT IT MEASURES. rl2 - (its own hold-phase baseline) is how much linear road
# evidence 20 epochs of task gradient WRITE INTO the image. Registered
# prediction §6.3: that gain exceeds the frozen rl1 - rl0 gap — adaptation
# writes more decodable structure than the pretrained SR provides. Registered
# prediction §6.4: even the best rung stays far below the U-Net arms'
# segmentation quality, which is the thesis-relevant reading — the R-series gain
# is NOT mostly the SR acting as a segmenter.
#
# READ THIS ARM BEFORE rl4 (probe doc §10.6, Gate C). SEN2SR-Lite is the
# smaller, FFT-anchored generator, so if the mask-painting degeneracy of §7
# shows up here it will be larger in rl4; if it shows up nowhere, that is the
# finding. NB the constraint is OFF in this campaign (SR_HC=off, the bare lane),
# so "FFT-anchored" describes what the generator was TRAINED under, not what it
# runs under here — the HC question belongs to the R-series 2x2.
#
# NO EARLY STOPPING, deliberately: like rl4, this arm's measured quantity IS its
# trajectory, and a stopper reading a val metric would end the run exactly where
# the interesting part starts. The engine whitelists rl3 for FIT_EARLY_STOP and
# refuses it here, so a submit-time typo cannot truncate a rung.
#
# Prerequisite: SEN2SR-Lite's model dir on scratch (prefetch on a login node
# with sr.sen2sr_loader.download_sen2sr). The mask file it ships is unused —
# this is the bare lane.
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"
source "$RL_DIR/_rl_rung.sh"

EXP_TAG="${EXP_TAG:-rl2_new${RUNG_TAG}}"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"   # cold joint fine-tuning, gated by the hold
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN}"

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
# already carries, and exit — an epoch of SEN2SR for nothing.
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

# Explicit, not inherited: this arm must run the full budget (see NO EARLY
# STOPPING above). Belt only — the brace is in the engine, which whitelists rl3
# and REFUSES FIT_EARLY_STOP=1 on any other EXP_TAG, this one included.
FIT_EARLY_STOP="${FIT_EARLY_STOP:-0}"

# The SR-weights snapshot strips ARE the mechanism figure for §7 — a collapse
# with no frames is an observation that cannot be shown. _rl_common.sh already
# sets every epoch; restated nowhere, changed nowhere.

# --- THE OVERLAY, PLANTED (no tune stage) ------------------------------------
# This arm searches NOTHING: the head lr, the loss, λ, the batch size and this
# rung's lr_sr are pinned constants (_rl_common.sh + _rl_rung.sh). Its "tune"
# would have been one trial of one epoch whose only product was this file of
# values it was handed, so the file is written directly and STAGE=tune is
# skipped entirely. The engine plants it into RUN_DIR at the fit/bench stages.
#
# THIS MUST STAY BYTE-EQUIVALENT TO WHAT sr.tune WOULD HAVE WRITTEN.
# `sr.tune.write_best_overlay` owns the schema; this heredoc is a second author,
# which is a drift hazard — so it is pinned by a test rather than by eye:
# tests/test_rl_campaign_hpc.py::test_the_planted_overlay_is_what_a_tune_would_have_written
# rebuilds it through that function with these constants and compares, for every
# rung. If you change a constant above, run that test.
#
# It differs from rl4's overlay in ONE key — `upsampler` — which is the point of
# the row. Everything else is the campaign's shared control.
#
# What each key is doing here (the rest is belt — the engine re-passes it as
# --model.* AFTER the config layers, so those keys cannot drift):
#   lr, lr_sr, pos_weight, batch_size, precision   NOT in the fit belt. The
#       overlay is the only place they come from — drop one and the fit silently
#       takes joint_sr.yaml's default instead. lr_sr is THE treatment here.
#   sr_hc: 'off'                                   the bare lane. For SEN2SR
#       this is a REAL operator change (unlike the SR4RS row, where off and
#       native are the same thing): no positivity clamp and no FFT splice, so
#       the generator runs raw and nothing pins its low frequencies.
#   sr_hold_epochs: 10.0                           the hard hold. It IS in the
#       belt, but it is written anyway: an overlay replayed on its own (a bench,
#       a viz, a resumed run) must not be able to adapt the generator from step
#       0 when the run it describes held it for ten epochs.
#   encoder_name/encoder_weights: null             head=linear builds no U-Net.
#
# The rung is substituted in, not hard-coded: __LR_SR__ below is replaced with
# LRSR normalised to a YAML float. `1e-3` is NOT a float to PyYAML (its 1.1
# resolver needs a dot AND a signed exponent), so an un-normalised rung would
# reach the model as the STRING "1e-3" — hence %.10e, which is exact for every
# rung on the ladder and always parses.
LR_SR_YAML=$(awk -v v="$LRSR" 'BEGIN { printf "%.10e", v }' </dev/null)

read -r -d '' BEST_PARAMS <<'YAML' || true
# Planted by scripts/hpc/sr/rl/rl2.sh — this arm searches nothing (no tune stage).
# Equivalent to sr.tune.write_best_overlay's output for the pinned constants;
# pinned by tests/test_rl_campaign_hpc.py.
model:
  encoder_name: null
  encoder_weights: null
  upsampler: sen2sr
  freeze_sr: false
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
  lr_sr: __LR_SR__
  lr_schedule: cosine
  sr_warmup_epochs: 1.0
  sr_hold_epochs: 10.0
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
# Quoted heredoc + placeholder, as the refit scripts do: nothing in the block
# can be eaten by a stray $ in a later edit.
BEST_PARAMS="${BEST_PARAMS/__LR_SR__/$LR_SR_YAML}"

source "$RL_DIR/../_stages_tv.sh"
