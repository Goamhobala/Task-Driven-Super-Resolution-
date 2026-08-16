#!/bin/bash
# RL1 — FROZEN SEN2SR-Lite + LINEAR PROBE read-out, WITH reflect-padding (8 px),
# ROSA_New. FINAL protocol (tune on train/val -> refit on train+val -> test).
#
# Twin: r1a_new.  rl1 - rl0 = the linear separability ADDED by frozen SEN2SR —
# a property of the image, cleanly attributable to the upsampler, because a
# per-pixel logistic regression has no spatial context and no capacity to
# compensate for a front-end that degrades it.
#
# Also stage 1 of the SEN2SR LP-FT pair: rl2_new.sh warm-starts its probe from
# THIS arm's final ckpt, so this fit must COMPLETE before rl2's tune starts.
#
# SR_PAD=8 mirrors the twin. Inferred (§2): a purely spectral probe is the worst
# possible consumer of an FFT border ring — it cannot contextually discount it —
# so this is where a padded/unpadded contrast would show up largest. Not in
# scope; rl1/rl2 stay padded and the write-up says so.
#
# lr_sr is auto-skipped (frozen SR).
#
#   bash scripts/hpc/submit.sh sr/rl1_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl1_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl1_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl1_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
SR_PAD=8

source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
