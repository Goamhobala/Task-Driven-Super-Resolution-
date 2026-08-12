#!/bin/bash
# R4a — joint task-driven fine-tuning of SR4RS (WGAN-GP-trained generator,
# PyTorch port) WITH reflect-padding (8 px), CDNGI labels. The unconstrained
# contrast to SEN2SR's hard-constraint: same joint protocol, LR pair searched.
# vs r4b: padding on/off isolates GAN edge effects (no FFT ring here, but GAN
# borders are their own artifact).
#
# Prerequisites: run scripts/sr4rs/extract_sr4rs.py locally (TF venv), PASS
# `python -m sr.sr4rs_torch --model-dir ...` parity, upload gen_*.safetensors/
# json/npz into $SEN2SR_DIR. No TF anywhere near the cluster.
#
#   sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r4a_cdngi.sh STAGE=tune [SEED=n] [LOSS_ARM=arm]
#   sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r4a_cdngi.sh STAGE=fit [SEED=n] [LOSS_ARM=arm]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r4a_cdngi"
LABELS="cdngi"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=8
# Any unet.losses.build_loss arm (bce | gap_ce | tl_ce | gap_tl_ce | t2_ce |
# t4_ce | bce_dice | pstar_dice | pstar_tversky | focal_tversky |
# <base>+cldice | <base>+skelrec); empty = legacy Dice + pos-weighted BCE.
# Hps (GAP_R, TL_ELL, CL_ALPHA, SKEL_W, ...) pass through _stages.sh too.
LOSS_ARM="${LOSS_ARM:-}"
SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-4}"      # PINNED, not searched (2026-08-12) -- see
                                     # _stages.sh. 4 is the SR-series constant and
                                     # the largest that fits: 8 OOMs on 44GB (SR4RS
                                     # runs 256-ch convs, incl. a 9x9, at 512px).

source "$REPO_DIR/scripts/hpc/sr/_stages.sh"
