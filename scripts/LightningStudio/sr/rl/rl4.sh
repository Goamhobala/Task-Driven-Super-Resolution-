#!/bin/bash
# RL4 — JOINT task-driven fine-tuning of SR4RS under a LINEAR PROBE, BARE,
# ONE RUNG of the lr_sr ladder. Lightning Studio. The heaviest arm in the
# series.
#
#   S=sr/rl/rl4.sh
#   bash scripts/LightningStudio/run.sh $S STAGE=tune  LRSR=1e-3
#   bash scripts/LightningStudio/run.sh $S STAGE=fit   LRSR=1e-3
#   bash scripts/LightningStudio/run.sh $S STAGE=bench LRSR=1e-3
#
# Same single-run hold-then-ramp shape as rl2 (10 held epochs, 20 joint), same
# ladder, same everything except the generator — see rl2.sh for the protocol and
# _rl_rung.sh for the rungs.
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
# The §8 capacity caveat applies here with the most force: "how much of a
# segmenter the largest generator in the series becomes" is the honest reading
# of rl4 - rl3, not "the value of adaptation".
#
# BUDGET / VRAM. Plan §4 gate 1: run STAGE=tune (1 trial x 1 epoch) first and
# read h/epoch and peak VRAM off it. The plan's original "~8–12 GB at bs=4"
# estimate was WRONG and this arm OOMed a 24 GB L4 on 2026-08-29. Measured
# (saved-for-backward set, bs=4, 128px LR -> 512px SR, in the fp32 island
# `_sr_forward` runs the generator in): ~18 GiB, ~13 GiB of it res_4x alone —
# 256 channels at 512px is ~1.07 GiB PER retained tensor. SR4RSGenerator is
# 11.3 M trainable params against TrainableSEN2SR's 187 K, and it is the grid,
# not the parameter count, that does the damage.
#
# So the plan's own first remedy is now ON here: SR4RS_GRAD_CKPT=1 activation-
# checkpoints the res_2x/res_4x stages, taking the retained set to ~6 GiB. It
# is bit-exact — no dropout, no RNG in that generator, and
# tests/test_sr4rs_torch.py pins outputs AND every gradient identical with the
# flag on and off — so it costs one extra forward of those stages and voids no
# between-arm contrast. If it STILL OOMs: grad-accum 2x2 (which the anorm EMA
# then sees as half-batches). NEVER change the batch size — it is a between-arm
# constant and `length` is fixed per epoch. (The flag is exported below.)
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"
source "$RL_DIR/_rl_rung.sh"
source "$RL_DIR/../../env.sh"

EXP_TAG="${EXP_TAG:-rl4_new${RUNG_TAG}}"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SR4RS_RGBN}"

# Activation checkpointing on the SR4RS upsample stages — see BUDGET / VRAM
# above. Read by `sr.sr4rs_torch.sr4rs_grad_ckpt_enabled` at model construction;
# an env flag rather than an hparam so it stays out of the checkpoint and out of
# a run's identity. Set SR4RS_GRAD_CKPT=0 to measure the unpatched footprint.
export SR4RS_GRAD_CKPT="${SR4RS_GRAD_CKPT:-1}"

source "$LS_DIR/sr/_stages_tv.sh"
