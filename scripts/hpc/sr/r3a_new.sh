#!/bin/bash
# R3a — FROZEN SR4RS (WGAN-GP-trained generator, PyTorch port) WITH SEN2SR's
# FFT HARD CONSTRAINT mounted on top (pad 8), ROSA_New. FINAL protocol (tune on
# train/val -> refit on train+val -> test). lr_sr is auto-skipped (frozen SR).
#
# *** THE r3 TAG WAS REDEFINED. *** It used to mean the full (Mamba) SEN2SR —
# see the retired r3a_cdngi.sh / r3b_cdngi.sh, which are a DIFFERENT series
# (cdngi labels, cdngi store) and stay on disk unchanged. The Mamba line is
# dropped; r3 now names the FROZEN SR4RS row. The _new suffix and the cdngi
# series tag keep the two senses apart in the append-only store, but do not
# read r3a_cdngi and r3a_new as the same treatment.
#
# WHAT THIS ARM IS: r1a with the backbone swapped to SR4RS. Same frozen
# read-out, same HC bundle, different generator. It fills the frozen row of the
# constraint x generator grid (rows = training regime, columns = constraint):
#
#                        b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   frozen  SEN2SR-Lite  r1b_new                  r1a_new
#   frozen  SR4RS        r3b_new                  r3a_new  <- THIS ARM
#   joint   SEN2SR-Lite  r2b_new                  r2a_new
#   joint   SR4RS        r4b_new                  r4a_new
#
# Contrasts this buys:
#   r3a - r3b   what the constraint does to a FROZEN generator that never saw
#               it in training — the operator's own contribution with no
#               optimisation in the path to absorb or exploit it, on the row
#               where the generator has no low-frequency anchor of its own.
#   r4a - r3a   task-driven adaptation measured on the constrained SR4RS lane,
#               the exact twin of r2a - r1a on the SEN2SR lane.
#
# The bundle, applied identically in every `a` cell: clamp(., min=0) -> FFT
# splice with SEN2SR-Lite's shipped 512px sigma=35 mask -> reflect pad 8 with
# output crop. The mask is reused BYTE-FOR-BYTE (see r4a_new.sh): it was
# optimised for SEN2SR-Lite, not SR4RS, so any gain here is a LOWER BOUND on
# what a tuned cutoff could give this generator. Pre-registered — state it in
# the write-up rather than tuning r per arm.
#
# Relation to r5_new: r5_new is frozen SR4RS bare-and-native (no HC, pad 0),
# i.e. the same treatment r3b_new runs under an explicit _nohc tag. r3a_new has
# no r5 equivalent — the constraint is what is new here.
#
# Prerequisites:
#   * extract_sr4rs.py -> gen_*.{safetensors,json,npz} in $SEN2SR_DIR
#     (parity-verify with `python -m sr.sr4rs_torch` locally). No TF on the
#     cluster.
#   * SEN2SR-Lite's model dir on scratch, for HC_MASK_PATH below.
#
#   bash scripts/hpc/submit.sh sr/r3a_new.sh STAGE=tune  [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r3a_new.sh STAGE=fit   [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r3a_new.sh STAGE=bench [SEED=n] [LOSS_ARM=arm]
#
# Budget: frozen is NOT cheap here — the arm still runs an 11.3 M-param SR4RS
# forward at 512 px on every batch, and pad 8 grows the grid to 576, so allow
# ~1.3x r3b_new's per-trial walltime. If bs=4 OOMs at 576 px the fix is
# activation checkpointing on the SR4RS blocks (numerically identical) — NEVER
# a batch-size change, which is a pinned between-arm constant.
#
# Pin the loss θ/pos_weight at submit time exactly as r3b_new did
# (SEARCH_THETAS=false ...), and use the same seed set — the R-series rule;
# leaving the search on reconfounds the SR comparison.
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r3a_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
# The HC lane. Pad travels with the constraint: the FFT splice assumes a
# periodic patch, so edge discontinuities ring at the borders.
SR_PAD=8
SR_HC="on"
HC_MASK_PATH="${HC_MASK_PATH:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"

REG="${REG:-true}"
LOSS_ARM="${LOSS_ARM:-}"

SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-4}"      # PINNED, not searched (2026-08-12) -- see
                                     # _stages_tv.sh. 4 is the SR-series constant and
                                     # the largest that fits: 8 OOMs on 44GB (SR4RS
                                     # runs 256-ch convs, incl. a 9x9, at 512px).

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
