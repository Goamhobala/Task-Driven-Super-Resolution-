#!/bin/bash
# RL4a — JOINT task-driven fine-tuning of SR4RS WITH SEN2SR's FFT HARD
# CONSTRAINT mounted on top (pad 8), under a LINEAR PROBE read-out, ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> report on test).
#
# The hard-constraint variant of rl4_new; twin of r4a_new. See rl1b_new.sh for
# the 2x2 layout and rl3a_new.sh for the bundle and the mask provenance.
#
# This is the cell where the constraint is asked to do the thing it is claimed
# to do. rl4 - rl4a: SR4RS is the larger generator with no anchor of its own, so
# if the mask-painting degeneracy of §7 is real anywhere it should be largest in
# rl4 — and mounting the splice on top is the direct test of whether
# low-frequency pinning PREVENTS it. Under a linear read-out there is no decoder
# to absorb the difference, so the answer is attributable to the operator.
#
# Order of reading (Gate C, docs/sr_linear_probe.md §10.6): rl2, then rl4, then
# this arm. If the unconstrained joint arms have not degenerated, this cell is a
# control that costs the most walltime in the series and can be dropped first
# under deadline pressure. If they HAVE, this is the arm that turns "the
# generator became a segmenter" into "the constraint stops the generator
# becoming a segmenter", which is the stronger claim.
#
# Init: LP-FT from rl3a_new's FINAL head — the CONSTRAINED frozen twin, not rl3.
# _warm_head_tv.sh resolves the stage-1 run dir including the _hc tag, so
# rl3a_new's fit must COMPLETE (same SEED/LOSS_ARM/REG/SR_HC) before this tune.
#
# The §8 capacity caveat applies with more force here than anywhere: SR4RS is
# the largest generator in the series, so "how much of a segmenter the generator
# becomes" is the honest reading of rl4a - rl3a, not "the value of adaptation".
#
# Prerequisites:
#   * gen_*.{safetensors,json,npz} in $SEN2SR_DIR (parity-verify locally with
#     `python -m sr.sr4rs_torch`; no TF on the cluster).
#   * SEN2SR-Lite's model dir on scratch for HC_MASK_PATH.
#
# Budget: the heaviest arm in the series — SR4RS at 576 px with gradients. Allow
# ~1.3x rl4_new. On OOM use activation checkpointing, never a batch-size change
# (_rl_common.sh, §6.1).
#
#   bash scripts/hpc/submit.sh sr/rl3a_new.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/rl4a_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl4a_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl4a_new.sh STAGE=bench [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl4a_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=8
SR_HC="on"
HC_MASK_PATH="${HC_MASK_PATH:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"

SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

# Evidence for the degeneracy diagnostics (§7), not a demo. Keep it on: this arm
# and rl4 are the pair the constraint claim is read off.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

STAGE1_TAG="rl3a_new"
source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_warm_head_tv.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
