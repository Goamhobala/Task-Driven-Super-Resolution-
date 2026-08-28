#!/bin/bash
# R1b — FROZEN SEN2SR-Lite with the FFT HARD CONSTRAINT OFF, ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> test).
# lr_sr is auto-skipped (frozen SR).
#
# *** THIS ARM WAS REDEFINED (2026-08-28), following r2b's redefinition of
# *** 2026-08-16 (docs/hc_2x2_plan.md §4).
# r1b_new used to mean "frozen SEN2SR + hard constraint, pad 0" — the
# padding-only contrast against r1a. It now means "frozen BARE generator": no
# positivity clamp, no frequency splice, no pad, exactly the operator r2b runs.
# SR_HC=off puts _nohc into the run dir, the Optuna study and the bench
# model_name, so the two senses of the name can never meet in the append-only
# store — but any LEGACY sr_r1b_new_* rows (no _nohc) are the OLD arm and must
# be flagged as retired in the analysis config, exactly as r2b's were.
#
# *** THE OLD TUNE DOES NOT CARRY OVER. *** A frozen bare generator is a
# different treatment, so its lr must be re-searched: STAGE=tune first. The
# _nohc tag means the old overlay is not even on the path, so this fails loudly
# rather than silently refitting under the wrong operator.
#
# WHY: THE b COLUMN IS ONE OPERATOR, IN EVERY ROW
# -----------------------------------------------
#                 b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   frozen        r1b_new  <- THIS ARM     r1a_new
#   cold joint    r2b_new                  r2a_new
#
# The column treatment is a single operator applied identically down the
# column: clamp(., min=0) -> FFT splice with the shipped 512px sigma=35 mask ->
# reflect pad 8 with output crop. Every component exists because of the
# constraint, so the b cells run the RAW generator — including no clamp
# (docs/hc_2x2_plan.md §4, "Scheme B"). Padding travels with the constraint:
# no splice means no Gibbs ringing at the patch border for a pad to mitigate.
#
# What this buys, which the old r1b could not give:
#   r2b - r1b   what task-driven adaptation writes into the BARE generator —
#               the exact twin of r2a - r1a on the constrained lane, so the
#               two differences are comparable rather than measuring
#               "adaptation" against two different frozen baselines.
#   r1a - r1b   what the constraint does to a generator NOTHING is training —
#               the operator's own contribution, with no optimisation in the
#               path to absorb or exploit it.
# Note SEN2SR-Lite was trained with the bundle in the loop, so its raw output
# was never a deployed product; that is the point of the cell.
#
# Linear-probe twin: rl1b_new (same treatment, 1x1 probe instead of the UNet).
#
#   sbatch --gres=gpu:1 -J r1b_new_tune -o slurm-%x-%j.txt \
#     scripts/hpc/train.sbatch --SCRIPT=sr/r1b_new.sh STAGE=tune  [SEED=n]
#   ... STAGE=fit   ... STAGE=bench
#
# Everything except the constraint is pinned identical to r1a_new: frozen SR,
# same loss arm, same seeds, BATCH_SIZES=4, REFIT_EPOCHS=100. Pin the loss
# θ/pos_weight at submit time exactly as r1a_new did (SEARCH_THETAS=false ...)
# — the R-series rule; leaving the search on reconfounds the SR comparison.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r1b_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="true"
# The bare lane: no FFT splice means no Gibbs ringing at the patch border, so
# there is no artifact for the pad to mitigate. Pad travels with the constraint.
SR_PAD=0
SR_HC="off"

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
