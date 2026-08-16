#!/bin/bash
# RL2 — JOINT task-driven SEN2SR fine-tuning under a LINEAR PROBE read-out,
# WITH padding (8 px), ROSA_New. FINAL protocol (tune on train/val -> refit on
# train+val -> test). The load-bearing arm of the series.
#
# Twin: r2a_new.  rl2 - rl1 = how much of a SEGMENTER the pretrained generator
# becomes when the only read-out is linear. This upper-bounds the share of the
# R-series joint gain (r2a - r1a) that is the SR net acting as a segmenter
# rather than as an image enhancer — the open question R2/R4 cannot answer from
# their own arms, because a 24 M-param U-Net can compensate for anything the
# generator does.
#
# DO NOT describe rl2 - rl1 as "the value of adaptation" (§8): it is adaptation
# PLUS ~240 k newly-trainable SEN2SR parameters against rl1's 5. Here the
# generator IS the model.
#
# Init: LP-FT. The probe is warm-started from rl1's FINAL head, so it is fully
# converged on the frozen-SR input distribution before any gradient reaches the
# generator, and SR trainability is the ONLY difference between the two arms.
# The cold alternative would rest on an unverified hope about relative
# timescales (lr_sr is non-zero from step 2 of the ramp), and would mix "SR
# unfrozen" with "the head followed a different early trajectory" — which
# matters because early head gradients are precisely what shape the SR's
# adaptation. Warm-starting removes the assumption rather than testing it.
#
# Gate C (§10.6): READ THIS ARM'S DIAGNOSTICS BEFORE LAUNCHING rl3/rl4.
# With a linear read-out and a segmentation-only loss, the generator's
# lowest-resistance solution is to paint the road mask into its output channels
# — the "SR image" becomes a road-probability map in reflectance coordinates.
# If that happened, rl2 - rl1 measures conv-net capacity, not image improvement.
# Signature to look for: the four bands collapsing toward mutual correlation ~1
# (one road channel replicated). Note sr_drift_rel CANNOT answer this — it is
# weight-space and cosine-confounded — so the snapshots below are the evidence,
# not a demo. No HR reference exists for these tiles, so this supports a DRIFT
# claim, never a fidelity one.
#
#   bash scripts/hpc/submit.sh sr/rl1_new.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/rl2_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl2_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl2_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl2_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=8

# Frame-by-frame replay of what the task loss writes into the generator. For
# this arm it is evidence for §7, so it is on by default and should stay on.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

STAGE1_TAG="rl1_new"
source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_warm_head_tv.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
