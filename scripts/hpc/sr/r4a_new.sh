#!/bin/bash
# R4a — COLD joint fine-tuning of SR4RS (WGAN-GP-trained generator, PyTorch
# port) WITH SEN2SR's FFT HARD CONSTRAINT mounted on top, ROSA_New. FINAL
# protocol (tune on train/val -> refit on train+val -> test).
#
# New arm (docs/hc_2x2_plan.md, 2026-08-16). It completes the 2x2 that
# separates the constraint from the generator architecture:
#
#                 b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   SEN2SR-Lite   r2b_new                  r2a_new
#   SR4RS         r4b_new                  r4a_new  <- THIS ARM
#
# Primary contrasts: HC | SEN2SR = r2a - r2b, HC | SR4RS = r4a - r4b. The
# interaction is descriptive only — per-tile power for it is not established.
#
# NB the `a` suffix here means the HARD-CONSTRAINT BUNDLE, not "padded". (The
# cdngi-series r4a_cdngi is SR4RS + pad 8 with NO constraint; series tags keep
# the stores separate, but do not read the two names as the same treatment.)
# The bundle, applied identically in both rows: clamp(., min=0) -> FFT splice
# with the shipped mask -> reflect pad 8 with output crop. Each component
# exists because of the constraint — positivity and spectral consistency are
# the paper's two conditions, and the pad mitigates the splice's Gibbs ringing.
#
# The mask is SEN2SR-Lite's shipped hard_constraint.safetensor, reused
# BYTE-FOR-BYTE: a single 512x512 sigma=35 Gaussian, i.e. r = 0.14, the
# in-training optimum Table 4 reports for the Non-Reference RGBN x4 task. It was
# optimised for SEN2SR-Lite, NOT for SR4RS, so any gain here is a LOWER BOUND on
# what a tuned constraint could give this generator. Pre-registered; state it in
# the write-up rather than tuning r per arm. data.crop_size is 128 for every
# arm, so the 512px mask applies unchanged (pad 8 grows it to 576 by the same
# bilinear resize r2a uses — the cutoff stays at its fraction of Nyquist).
#
# Prerequisites:
#   * extract_sr4rs.py -> gen_*.{safetensors,json,npz} in $SEN2SR_DIR
#     (parity-verify with `python -m sr.sr4rs_torch` locally). No TF on the
#     cluster.
#   * SEN2SR-Lite's model dir on scratch, for HC_MASK_PATH below (the r2 arms
#     already need it; prefetch with sr.sen2sr_loader.download_sen2sr).
#
#   bash scripts/hpc/submit.sh sr/r4a_new.sh STAGE=tune  [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r4a_new.sh STAGE=fit   [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r4a_new.sh STAGE=bench [SEED=n] [LOSS_ARM=arm]
#
# Budget: the tune is the expensive one. SR4RS runs 256-channel convs (incl. a
# 9x9) and pad 8 grows the grid 512 -> 576 px, so allow ~1.3x r4b_new's
# per-trial walltime. If bs=4 OOMs at 576 px the fix is activation checkpointing
# on the SR4RS blocks (numerically identical) — NEVER a batch-size change, which
# is a pinned between-arm constant.
#
# Pin the loss θ/pos_weight at submit time exactly as r4b_new did
# (SEARCH_THETAS=false ...), and use the same seed set — the R-series rule.
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r4a_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
# The HC lane. Pad travels with the constraint: the FFT splice assumes a
# periodic patch, so edge discontinuities ring at the borders.
SR_PAD=8
SR_HC="on"
HC_MASK_PATH="${HC_MASK_PATH:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"

# NB the _all series ran the sr4rs arm at REG=false by default. Here the default
# is the recipe-v2 arm, matching every other _new arm — the unregularised run is
# an explicit REG=false ablation, not the headline number.
REG="${REG:-true}"

SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"
LOSS_ARM="${LOSS_ARM:-}"

SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-4}"      # PINNED, not searched (2026-08-12) -- see
                                     # _stages_tv.sh. 4 is the SR-series constant and
                                     # the largest that fits: 8 OOMs on 44GB (SR4RS
                                     # runs 256-ch convs, incl. a 9x9, at 512px).

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
