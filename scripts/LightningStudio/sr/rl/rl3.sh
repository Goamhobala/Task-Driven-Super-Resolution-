#!/bin/bash
# RL3 — FROZEN SR4RS x4 under a LINEAR PROBE read-out, BARE. Lightning Studio.
#
# The write-up ladder's r3 is the run-tag r5 (tags never change; the store is
# append-only and the thesis carries one mapping table). rl3 is its probe twin:
# rl3 - rl0 = the separability a frozen WGAN-GP generator adds.
#
# COST WARNING. "Frozen" does not mean cheap here. The probe is 5 parameters,
# but the arm still runs an 11.3 M-param SR4RS forward at 512 px on every batch,
# so rl3 costs roughly what r5 costs — replacing the decoder saves the U-Net's
# forward+backward, not the SR front-end, and the front-end dominates. This is
# the arm the plan's §4 cost anchor was measured on.
#
# SR4RS ships no FFT hard constraint, so SR_HC=off is its native behaviour;
# it is forced explicitly anyway so that all five arms carry the same _nohc tag
# and no arm's constraint state is implicit.
#
# Prerequisite: gen_*.{safetensors,json,npz} in SEN2SR_DIR. Parity-verify the
# port locally first (`python -m sr.sr4rs_torch`) — there is no TF here.
#
#   bash scripts/LightningStudio/run.sh sr/rl/rl3.sh STAGE=tune   # 1x1: pin + timing
#   bash scripts/LightningStudio/run.sh sr/rl/rl3.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh sr/rl/rl3.sh STAGE=bench
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"
source "$RL_DIR/../../env.sh"

EXP_TAG="${EXP_TAG:-rl3_new}"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SR4RS_RGBN}"

source "$LS_DIR/sr/_stages_tv.sh"
