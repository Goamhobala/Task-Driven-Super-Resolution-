#!/bin/bash
# R3b — FROZEN SR4RS (WGAN-GP-trained generator, PyTorch port), BARE: no
# positivity clamp, no FFT splice, no pad, ROSA_New. FINAL protocol (tune on
# train/val -> refit on train+val -> test). lr_sr is auto-skipped (frozen SR).
#
# *** THE r3 TAG WAS REDEFINED. *** It used to mean the full (Mamba) SEN2SR —
# see the retired r3b_cdngi.sh, which is a DIFFERENT series (cdngi labels,
# cdngi store) and stays on disk unchanged. The Mamba line is dropped; r3 now
# names the FROZEN SR4RS row. Do not read r3b_cdngi and r3b_new as the same
# treatment.
#
# WHAT THIS ARM IS: r1b with the backbone swapped from SEN2SR-Lite to SR4RS.
# Same frozen read-out, same bare operator, different generator. It fills the
# frozen row of the constraint x generator grid:
#
#                        b: bare (no HC, pad 0)   a: HC bundle (pad 8)
#   frozen  SEN2SR-Lite  r1b_new                  r1a_new
#   frozen  SR4RS        r3b_new  <- THIS ARM     r3a_new
#   joint   SEN2SR-Lite  r2b_new                  r2a_new
#   joint   SR4RS        r4b_new                  r4a_new
#
# SR4RS ships no FFT hard constraint, so SR_HC=off is its NATIVE behaviour.
# It is forced explicitly anyway — the same discipline rl3 follows — so that
# every b cell carries the identical _nohc tag and no arm's constraint state is
# implicit. Consequence to know before submitting:
#
#   *** THIS IS r5_new's TREATMENT UNDER A DIFFERENT TAG. *** r5_new is frozen
#   SR4RS, pad 0, SR_HC=native — the same operator this arm runs. The _nohc tag
#   goes into the run dir, the Optuna study AND the bench model_name, so these
#   rows will NOT merge with the existing sr_r5_new_* ones and this arm needs
#   its own STAGE=tune. That is deliberate: the grid above only reads as a grid
#   if the b column is one tag down every row. If you would rather reuse r5's
#   budget, drop SR_HC below and report r5_new in the b cell instead — but then
#   say so in the write-up's mapping table, do not do it silently.
#
# Contrasts this buys:
#   r3a - r3b   the constraint's own contribution on a FROZEN generator that
#               never saw it in training, with no optimisation in the path to
#               absorb or exploit it.
#   r4b - r3b   what task-driven adaptation writes into the BARE SR4RS
#               generator — the twin of r2b - r1b on the SEN2SR lane.
#
# Free secondary result: frozen SR4RS is the unanchored generator with nothing
# training it, i.e. the cleanest read on the adaptive-norm chapter's claim that
# UNANCHOREDNESS drives post-SR moment drift. Watch the post-SR moments;
# adaptive_norm stays at the series setting — if this arm hits the documented
# drift failure mode, REPORT it, do not patch it mid-series.
#
# Linear-probe twin: rl3 (same treatment, 1x1 probe instead of the UNet).
#
# Prerequisites: extract_sr4rs.py -> gen_*.{safetensors,json,npz} in $SEN2SR_DIR
# (parity-verify with `python -m sr.sr4rs_torch` locally). No TF on the cluster.
#
#   bash scripts/hpc/submit.sh sr/r3b_new.sh STAGE=tune  [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r3b_new.sh STAGE=fit   [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r3b_new.sh STAGE=bench [SEED=n] [LOSS_ARM=arm]
#
# Budget: frozen is NOT cheap here — the arm still runs an 11.3 M-param SR4RS
# forward at 512 px on every batch, so it costs roughly what r5_new costs.
# Freezing saves the U-Net's forward+backward, not the SR front-end, and the
# front-end dominates.
#
# Pin the loss θ/pos_weight at submit time exactly as r3a_new does
# (SEARCH_THETAS=false ...), and use the same seed set — the R-series rule.
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r3b_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
# The bare lane: no FFT splice means no Gibbs ringing at the patch border, so
# there is no artifact for the pad to mitigate. Pad travels with the constraint.
SR_PAD=0
SR_HC="off"

REG="${REG:-true}"
LOSS_ARM="${LOSS_ARM:-}"

SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-4}"      # PINNED, not searched (2026-08-12) -- see
                                     # _stages_tv.sh. 4 is the SR-series constant and
                                     # the largest that fits: 8 OOMs on 44GB (SR4RS
                                     # runs 256-ch convs, incl. a 9x9, at 512px).

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
