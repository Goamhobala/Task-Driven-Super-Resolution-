#!/bin/bash
# RL4 — JOINT task-driven fine-tuning of SR4RS under a LINEAR PROBE, BARE,
# ONE RUNG of the lr_sr ladder. HPC (l40s). The heaviest arm in the series, and
# the reason the SR4RS row runs on the cluster at all: it OOMed a 24 GB L4 on
# 2026-08-29, and an l40s card has 48 GB.
#
# Cluster twin of scripts/LightningStudio/sr/rl/rl4.sh — same constants, same
# ladder, same tags. No early stopping here (see below), so its rows carry no
# _es tag and DO merge with the Studio's rl4 rows for the same rung.
#
# NO TUNE STAGE: the rung is a pin, not a search, so this arm carries its own
# best_params.yaml (see THE OVERLAY below) and the engine plants it.
#
#   S=sr/rl/rl4.sh
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=24:00:00 \
#          scripts/hpc/train.sbatch --SCRIPT=$S STAGE=fit   LRSR=1e-3
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=04:00:00 \
#          scripts/hpc/train.sbatch --SCRIPT=$S STAGE=bench LRSR=1e-3
#
# Same single-run hold-then-ramp shape as rl2 (10 held epochs, 20 joint), same
# ladder, same everything except the generator — see the Studio's rl2.sh for the
# protocol and _rl_rung.sh for the rungs. Run the extremes first (1e-3, 1e-6):
# they carry the effect and the probable collapses, and the middle rungs
# interpolate.
#
# WHY THIS ROW EXISTS. SR4RS is the larger generator and, unlike SEN2SR-Lite, it
# has NO low-frequency anchor of its own. If the mask-painting degeneracy of
# probe doc §7 is real anywhere, it should be largest here — and the campaign's
# top rung is where it would show. It is also the row where the R-series saw
# real drift: joint finetuning walks SR4RS off the reflectance scale, which
# SEN2SR's FFT constraint prevents. The adaptive-norm adapter is ON and tracking
# for exactly that reason, and the loosened rails are what let the drift be
# WATCHED instead of aborted.
#
# NO EARLY STOPPING — deliberately, and not merely by default. This arm's
# measured quantity IS its trajectory: how far task gradients walk the generator
# per unit dose, where the post-SR std leaves its nominal band, and whether the
# top rung dies numerically and at which step. A stopper reading a val metric
# would end the run exactly when that metric stops improving — i.e. it would
# truncate the record precisely where the interesting part starts, and a rung
# that ended at epoch 17 could not be compared with one that ran to 30. The
# fixed 30-epoch budget is what makes the ladder a dose-response curve.
# (Nothing else aborts it either: the rails are defanged to warn, and the
# trainval callback list has no stopper, so even a NaN loss runs to the end of
# the budget and its step is recorded. That is data, not a bug.)
#
# The §8 capacity caveat applies here with the most force: "how much of a
# segmenter the largest generator in the series becomes" is the honest reading
# of rl4 - rl3, not "the value of adaptation". Note also that this platform's
# rl3 early-stops, so the frozen comparator is a stopped-at-plateau run — check
# rl3's val_ap curve actually plateaued before reading the difference.
#
# BUDGET / VRAM. Gate 1 (plan §4) now reads off the fit's own first epoch —
# check h/epoch and peak VRAM there and `scancel` if the walltime was wrong. For
# a throwaway probe that cannot touch this rung's run dir or its rows:
#   sbatch ... --SCRIPT=$S STAGE=fit LRSR=1e-3 EXP_TAG=rl4_probe \
#              REFIT_EPOCHS=1 SR_HOLD_EPOCHS=0 SKIP_TEST=1
# (SR_HOLD_EPOCHS=0 so the probe's single epoch is a JOINT one — inside the hold
# the generator gets no gradients and the measurement would be rl3's.)
# Measured on an L4 (saved-for-backward set,
# bs=4, 128px LR -> 512px SR): ~18 GiB in the fp32 island, ~13 GiB of it res_4x
# alone — 256 channels at 512 px is ~1.07 GiB PER retained tensor. SR4RS ships
# no FFT hard constraint, so `_sr_forward` does not run it in the fp32 island:
# the generator inherits the Trainer's bf16-mixed autocast, halving the retained
# set to ~9 GiB while running the convolutions at ~2x throughput, with
# `pixel_norm` keeping its reduction in fp32 and `_sr_forward` still returning
# fp32 so the adaptive-norm EMA, the drift monitor and the probe are untouched.
# On a 48 GB l40s that leaves headroom the Studio did not have.
#
# If it OOMs anyway: SR4RS_GRAD_CKPT=1 composes with bf16 and is bit-exact — no
# dropout, no RNG in that generator, and tests/test_sr4rs_torch.py pins outputs
# AND every gradient identical with the flag on and off — so it voids no
# between-arm contrast, at the cost of one extra forward of the res_2x / res_4x
# stages (30-50% of the step). After that: grad-accum 2x2 (which the anorm EMA
# then sees as half-batches). NEVER change the batch size — it is a between-arm
# constant and `length` is fixed per epoch.
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"
source "$RL_DIR/_rl_rung.sh"

EXP_TAG="${EXP_TAG:-rl4_new${RUNG_TAG}}"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

# Explicit, not inherited: this arm must run the full budget (see NO EARLY
# STOPPING above), so the knob is pinned here where the reason is written down.
# Belt only — the brace is in the engine, which whitelists rl3 and REFUSES
# FIT_EARLY_STOP=1 on any other EXP_TAG, this one included. A submit-time typo
# cannot truncate this arm's budget or mint it an _es row.
FIT_EARLY_STOP="${FIT_EARLY_STOP:-0}"

# Activation checkpointing on the SR4RS upsample stages — OFF, superseded by
# bf16; see BUDGET / VRAM above. Read by `sr.sr4rs_torch.sr4rs_grad_ckpt_enabled`
# at model construction; an env flag rather than an hparam so it stays out of the
# checkpoint and out of a run's identity — which is exactly why turning it back
# on (SR4RS_GRAD_CKPT=1) if this OOMs costs no between-arm comparability.
export SR4RS_GRAD_CKPT="${SR4RS_GRAD_CKPT:-0}"

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
# What each key is doing here (the rest is belt — the engine re-passes it as
# --model.* AFTER the config layers, so those keys cannot drift):
#   lr, lr_sr, pos_weight, batch_size, precision   NOT in the fit belt. The
#       overlay is the only place they come from — drop one and the fit silently
#       takes joint_sr.yaml's default instead. lr_sr is THE treatment here.
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
# Planted by scripts/hpc/sr/rl/rl4.sh — this arm searches nothing (no tune stage).
# Equivalent to sr.tune.write_best_overlay's output for the pinned constants;
# pinned by tests/test_rl_campaign_hpc.py.
model:
  encoder_name: null
  encoder_weights: null
  upsampler: sr4rs
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
