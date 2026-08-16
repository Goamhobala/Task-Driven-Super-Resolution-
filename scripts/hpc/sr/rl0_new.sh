#!/bin/bash
# RL0 — bicubic x4 + LINEAR PROBE read-out, ROSA_New. The rl-series anchor.
# FINAL protocol (tune on train/val -> refit on train+val -> report on test).
#
# Twin: r0_new. Measures the linear spectral separability of road vs non-road
# with no learned SR at all — the zero point that rl1-rl0 and rl3-rl0 are
# measured against. Because it appears in two of the four load-bearing
# contrasts, a defect here biases both, which is why §6.3 keys the fp32 island
# on head=="linear" rather than on the presence of an SR net: rl0 has NO SR
# stage, so under the old keying its logits alone would run in bf16, and bf16's
# 8 mantissa bits produce far more tied values than fp32 — AP is a ranking
# statistic, so ties degrade it. The preflight in _stages_tv.sh refuses to run
# until that keying exists.
#
# No SR params: lr_sr is not searched, SR_PAD is irrelevant.
#
#   bash scripts/hpc/submit.sh sr/rl0_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl0_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl0_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl0_new"
LABELS="new"
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
