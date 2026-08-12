#!/bin/bash
# R4b — COLD joint task-driven fine-tuning of SR4RS (WGAN-GP-trained generator,
# PyTorch port) WITHOUT padding, ROSA_all. The unconstrained (GAN) contrast to
# SEN2SR: same joint protocol, LR pair searched. vs r4a: padding on/off isolates
# GAN border effects (no FFT ring here, but the GAN has its own edge artifact).
#
# Prerequisites: extract_sr4rs.py -> gen_*.{safetensors,json,npz} in $SEN2SR_DIR
# (see r4a_cdngi.sh). No TF anywhere near the cluster.
#
#   bash scripts/hpc/submit.sh sr/r4b_all.sh STAGE=tune [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r4b_all.sh STAGE=fit  [SEED=n] [LOSS_ARM=arm]
#
# Toggles (all overridable at submit time):
#   REG=false           unregularised-GAN ablation — one switch turns off clip /
#                       cosine schedule / SR warmup, and tags the run/study/
#                       bench name with _noreg so it never mixes with the v2
#                       (regularised) runs. Default true = the recipe-v2 arm.
#   SR_SNAPSHOT_EVERY=N fit-stage: save the fine-tuned SR net's WEIGHTS ONLY
#                       into <run dir>/sr_snapshots/ every N epochs (+ an
#                       epoch-0 init frame) to replay how the task loss reshapes
#                       the SR output. Default here = 2; set 0 to disable.
#   LOSS_ARM=arm        any unet.losses.build_loss arm (empty = legacy loss).
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r4b_all"
LABELS="all"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0

# Recipe-v2 regularisation toggle (consumed by _stages.sh). Exposed here so the
# unregularised-GAN ablation is a one-flag submit: REG=false.
REG="${REG:-false}"

# Save the fine-tuned SR4RS generator every 2 epochs during the refit (weights
# only, ~45 MB/frame). Optional: override SR_SNAPSHOT_EVERY=0 to switch off.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

# Any unet.losses.build_loss arm (see r4a_cdngi.sh); empty = legacy loss.
LOSS_ARM="${LOSS_ARM:-}"

SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-4}"      # PINNED, not searched (2026-08-12) -- see
                                     # _stages.sh. 4 is the SR-series constant and
                                     # the largest that fits: 8 OOMs on 44GB (SR4RS
                                     # runs 256-ch convs, incl. a 9x9, at 512px).

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
