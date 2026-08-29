#!/bin/bash
# RL0 — BICUBIC x4 under a LINEAR PROBE read-out, ROSA_New. Lightning Studio.
# The floor of the probe ladder: how much road evidence a 5-parameter linear
# read-out can pull out of a deterministic upsampling with no learned generator
# at all. Every other arm's decodability is reported against this.
#
# Twin of the write-up ladder's r0. Nothing is trainable except the 5-parameter
# head, so there is no lr_sr, no hold and no adaptation to measure — 30 epochs
# head-only. It still carries the campaign's uniform SR_HC=off / SR_PAD=0 /
# anorm / rails settings so that its run dir and bench row are tagged
# identically to the arms it anchors (a tag mismatch is how an "identical
# recipe" claim quietly stops being true).
#
# Registered prediction it participates in (plan §6.1): this arm must plateau
# well before epoch 10, since a near-convex 5-parameter head on a fixed input
# distribution has nothing left to learn. If it has not, the hold is too short
# and must be lengthened UNIFORMLY across the series.
#
#   bash scripts/LightningStudio/run.sh sr/rl/rl0.sh STAGE=tune   # 1x1: pin + timing
#   bash scripts/LightningStudio/run.sh sr/rl/rl0.sh STAGE=fit    # the 30-epoch run
#   bash scripts/LightningStudio/run.sh sr/rl/rl0.sh STAGE=bench  # store the test row
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"
source "$RL_DIR/../../env.sh"

EXP_TAG="${EXP_TAG:-rl0_new}"
LABELS="new"
UPSAMPLER="bicubic"
FREEZE_SR="false"   # no SR parameters exist; matches r0_new's invocation

source "$LS_DIR/sr/_stages_tv.sh"
