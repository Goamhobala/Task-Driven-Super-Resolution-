#!/bin/bash
# R2b — COLD joint SEN2SR-Lite fine-tuning with the FFT HARD CONSTRAINT OFF,
# ROSA_New. FINAL protocol (tune on train/val -> refit on train+val -> test).
#
# *** THIS ARM WAS REDEFINED (docs/hc_2x2_plan.md, 2026-08-16). ***
# r2b_new used to mean "SEN2SR + hard constraint, pad 0" — the padding-only
# contrast against r2a. It now means "bare SEN2SR generator": no positivity
# clamp, no frequency splice, no pad. SR_HC=off puts _nohc into the run dir,
# the Optuna study and the bench model_name, so the two senses of the name can
# never meet in the append-only store — but any LEGACY sr_r2b_new_* rows must
# still be flagged as retired in the analysis config (§9.5).
#
# The 2x2 this arm belongs to (rows = generator, columns = constraint):
#
#                 b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   SEN2SR-Lite   r2b_new  <- THIS ARM     r2a_new
#   SR4RS         r4b_new                  r4a_new
#
# The column treatment is ONE operator applied identically in both rows:
# clamp(., min=0) -> FFT splice with the shipped 512px sigma=35 mask -> reflect
# pad 8 with output crop. Every component exists because of the constraint, so
# the b cells run the RAW generator — including no clamp (docs/hc_2x2_plan.md
# §4, "Scheme B"). Note SEN2SR-Lite was trained with the bundle in the loop, so
# its raw output was never a deployed product; that is the point of the cell.
#
# Free secondary result: this is SEN2SR without its DC anchor, i.e. the direct
# test of the adaptive-norm chapter's root-cause claim that UNANCHOREDNESS, not
# the SR4RS architecture, drives post-SR moment drift. Watch sr_psnr_vs_init and
# the post-SR moments. adaptive_norm stays at the series setting either way — if
# this arm hits the documented drift failure mode, REPORT it, do not patch it
# mid-series.
#
#   bash scripts/hpc/submit.sh sr/r2b_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2b_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/r2b_new.sh STAGE=bench [SEED=n]
#
# Everything except the constraint is pinned identical to r2a_new: cold joint,
# same loss arm, same seeds, BATCH_SIZES=4, REFIT_EPOCHS=100, recipe-v2
# defaults. Pin the loss θ/pos_weight at submit time exactly as r2a_new did
# (SEARCH_THETAS=false ...) — the R-series rule; leaving the search on
# reconfounds the SR comparison.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r2b_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
# The bare lane: no FFT splice means no Gibbs ringing at the patch border, so
# there is no artifact for the pad to mitigate. Pad travels with the constraint.
SR_PAD=0
SR_HC="off"

REG="${REG:-true}"
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
