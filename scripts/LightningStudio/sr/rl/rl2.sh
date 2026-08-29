#!/bin/bash
# RL2 — JOINT task-driven fine-tuning of SEN2SR-Lite under a LINEAR PROBE,
# BARE, ONE RUNG of the lr_sr ladder. Lightning Studio.
#
#   S=sr/rl/rl2.sh
#   bash scripts/LightningStudio/run.sh $S STAGE=tune  LRSR=1e-3
#   bash scripts/LightningStudio/run.sh $S STAGE=fit   LRSR=1e-3
#   bash scripts/LightningStudio/run.sh $S STAGE=bench LRSR=1e-3
#
# ALL FOUR RUNGS SHARE THIS SCRIPT; LRSR is the coordinate and it goes into
# EXP_TAG, so run dirs, Optuna studies and benchmark rows are disjoint per rung
# by construction. See _rl_rung.sh for the ladder, the dose argument, the run
# order (extremes first) and the 1e-3 contingency.
#
# THE SHAPE OF ONE RUN (plan §2): 30 epochs, single job, no stage pairing.
# Epochs 1–10 run with lr_sr held at EXACTLY 0 — an LR gate on the SR parameter
# group, so Adam's update is identically zero while its moments warm on the real
# gradients. Epochs 11–30 run the rung's lr_sr on its own cosine. The hold phase
# is therefore a frozen-arm run, and rl1's epochs 11–30 are the matched-budget
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
# Read rl2 BEFORE rl4 (probe doc §10.6 Gate C): SEN2SR-Lite is the smaller,
# FFT-anchored generator, so if degeneracy shows up here it will be larger in
# rl4; if it shows up nowhere, that is the finding.
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"
source "$RL_DIR/_rl_rung.sh"
source "$RL_DIR/../../env.sh"

EXP_TAG="${EXP_TAG:-rl2_new${RUNG_TAG}}"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"   # cold joint fine-tuning, gated by the hold
SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SEN2SRLite_RGBN}"

source "$LS_DIR/sr/_stages_tv.sh"
