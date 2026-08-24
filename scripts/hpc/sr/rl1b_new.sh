#!/bin/bash
# RL1b — FROZEN SEN2SR-Lite with the FFT HARD CONSTRAINT OFF (bare generator,
# no pad) + LINEAR PROBE read-out, ROSA_New. FINAL protocol (tune on train/val
# -> refit on train+val -> report on test).
#
# The hard-constraint variant of rl1_new (docs/hc_2x2_plan.md x
# docs/sr_linear_probe.md). Crossing the two designs gives the rl 2x2
# (rows = generator, columns = constraint):
#
#                 bare (no HC, pad 0)          HC bundle (pad 8)
#   SEN2SR-Lite   rl1b (frozen) rl2b (joint)   rl1 (frozen)  rl2 (joint)
#   SR4RS         rl3  (frozen) rl4  (joint)   rl3a (frozen) rl4a (joint)
#
# WHY THIS ARM IS WORTH ITS GPU HOURS. Under a 24 M-param U-Net (r2a vs r2b)
# the constraint is measured through a decoder that can compensate for it. A
# 5-parameter per-pixel logistic regression cannot compensate for anything, so
# rl1 - rl1b is the cleanest read the whole project has on what the FFT splice
# does to the IMAGE's linear road/non-road separability — the constraint's own
# effect, with no capacity in the way. Sign is genuinely open: the splice pins
# the low frequencies to bicubic-upsampled LR (radiometric anchoring, good for
# a spectral probe) while also discarding whatever the generator put there.
#
# It is ALSO stage 1 of the bare-SEN2SR LP-FT pair: rl2b_new.sh warm-starts its
# probe from THIS arm's final ckpt, so this fit must COMPLETE before rl2b's
# tune starts. It cannot warm-start from rl1 instead — a probe converged on
# HC-spliced inputs is not converged on bare-generator inputs, and LP-FT's
# argument (docs/sr_linear_probe.md §2) needs SR trainability to be the ONLY
# difference between stage 1 and stage 2.
#
# The `b` lane is the RAW generator: no positivity clamp, no frequency splice,
# no pad (hc_2x2_plan §4, "Scheme B"). Every component of the bundle exists
# because of the constraint, so they travel together. SEN2SR-Lite was trained
# with the bundle in the loop, so its raw output was never a deployed product —
# that is the point of the cell, and it is also the direct test of the
# adaptive-norm chapter's root-cause claim that UNANCHOREDNESS, not the SR4RS
# architecture, drives post-SR moment drift. Watch the post-SR moments and
# sr_psnr_vs_init. adaptive_norm stays at the series setting either way: if
# this arm hits the documented drift failure mode, REPORT it, do not patch it
# mid-series.
#
# SR_HC=off puts _nohc into the run dir, the Optuna study and the bench
# model_name, so the constrained and bare senses of an arm can never meet in
# the append-only store.
#
# lr_sr is auto-skipped (frozen SR). Loss is frozen by _rl_common.sh — do not
# unpin it here, the whole series shares one loss control.
#
#   bash scripts/hpc/submit.sh sr/rl1b_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl1b_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl1b_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl1b_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
# Pad travels with the constraint: no FFT splice means no Gibbs ringing at the
# patch border, so there is no artifact for a pad to mitigate.
SR_PAD=0
SR_HC="off"

source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
