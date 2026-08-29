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
# read h/epoch and peak VRAM off it. Estimate ~8–12 GB at bs=4 — it was the
# U-Net, not SR4RS, that drove the old 44 GB OOM. If it OOMs: activation
# checkpointing on the SR4RS blocks first, then grad-accum 2x2 (which the anorm
# EMA then sees as half-batches). NEVER change the batch size — it is a
# between-arm constant and `length` is fixed per epoch.
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

source "$LS_DIR/sr/_stages_tv.sh"
