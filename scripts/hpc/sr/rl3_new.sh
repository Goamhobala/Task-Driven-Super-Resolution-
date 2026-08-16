#!/bin/bash
# RL3 — FROZEN SR4RS (pretrained WGAN-GP generator, torch port) + LINEAR PROBE
# read-out, ROSA_New. FINAL protocol (tune on train/val -> refit -> test).
#
# Twin: r5_new.  rl3 - rl0 = the linear separability added by the frozen
# unconstrained generator, the SR4RS counterpart of rl1 - rl0.
#
# Also stage 1 of the SR4RS LP-FT pair: rl4_new.sh warm-starts its probe from
# THIS arm's final ckpt, so this fit must COMPLETE before rl4's tune starts.
#
# No pad variant: SR4RS has no FFT hard constraint, so no Gibbs border ring, and
# padding only inflates the (already heavy) 512 px SR compute.
#
# NB SR4RS's output is UNANCHORED — that is the root cause the adaptive-norm
# work addressed. This arm is frozen, so `pre` recalibration would be the
# complete zero-risk fix for it; the engine default (`post`) is kept so all five
# rl arms share one normalisation policy. Changing it for this arm alone would
# put the nuisance variable back on the treatment.
#
# Prerequisites: gen_*.{safetensors,json,npz} in $SEN2SR_DIR. lr_sr auto-skipped.
#
#   bash scripts/hpc/submit.sh sr/rl3_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl3_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl3_new.sh STAGE=bench [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl3_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
