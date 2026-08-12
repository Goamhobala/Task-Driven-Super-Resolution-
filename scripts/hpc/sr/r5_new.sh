#!/bin/bash
# R5 — FROZEN SR4RS preprocessing (pretrained WGAN-GP generator, torch port),
# ROSA_New. FINAL protocol (tune on train/val -> refit on train+val -> test).
# The SR4RS analogue of R1: R5 - R0 = value of the frozen pretrained GAN SR on
# SA data. Stage 1 of the staged SR4RS protocol: r6_new.sh warm-starts its UNet
# from THIS arm's final ckpt.
# No pad variant: SR4RS has no FFT hard constraint, so no Gibbs border ring,
# and padding only inflates the (already heavy) 512 px SR compute.
#
# Prerequisites: gen_*.safetensors/json/npz in $SEN2SR_DIR.
#
#   bash scripts/hpc/submit.sh sr/r5_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/r5_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/r5_new.sh STAGE=bench [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r5_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-4}"      # PINNED, not searched (2026-08-12) -- see
                                     # _stages_tv.sh. 4 is the SR-series constant and
                                     # the largest that fits: 8 OOMs on 44GB (SR4RS
                                     # runs 256-ch convs, incl. a 9x9, at 512px).

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
