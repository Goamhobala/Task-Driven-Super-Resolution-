#!/bin/bash
# RL4 — JOINT task-driven SR4RS fine-tuning under a LINEAR PROBE read-out,
# ROSA_New. FINAL protocol (tune on train/val -> refit on train+val -> test).
#
# Twin: r4b_new.  rl4 - rl3 is the unconstrained-generator counterpart of
# rl2 - rl1. SEN2SR carries an FFT hard constraint that pins its band means;
# SR4RS carries none, so if the mask-painting degeneracy of §7 is real anywhere,
# it should be largest here. Read rl2 first (Gate C, §10.6): if that arm has
# already collapsed to mask-painting, THAT is the headline result and this arm
# becomes a confirmation rather than an exploration — which may change how much
# budget it deserves before 4 September.
#
# Init: LP-FT from rl3's FINAL head, for the reasons set out in rl2_new.sh.
#
# The §8 capacity caveat applies with more force here than anywhere: SR4RS is
# the larger generator, so "how much of a segmenter the generator becomes" is
# the honest reading of rl4 - rl3, not "the value of adaptation".
#
# Prerequisites: gen_*.{safetensors,json,npz} in $SEN2SR_DIR (parity-verify
# locally with `python -m sr.sr4rs_torch`; no TF on the cluster).
#
#   bash scripts/hpc/submit.sh sr/rl3_new.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/rl4_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl4_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl4_new.sh STAGE=bench [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl4_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

# Evidence for the degeneracy diagnostics (§7), not a demo. Keep it on.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

STAGE1_TAG="rl3_new"
source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_warm_head_tv.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
