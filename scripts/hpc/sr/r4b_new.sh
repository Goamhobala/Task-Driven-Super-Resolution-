#!/bin/bash
# R4b — COLD joint task-driven fine-tuning of SR4RS (WGAN-GP-trained generator,
# PyTorch port) WITHOUT padding, ROSA_New. FINAL protocol (tune on train/val ->
# refit on train+val -> test). The unconstrained (GAN) contrast to SEN2SR: same
# joint protocol, LR pair searched.
#
# Prerequisites: extract_sr4rs.py -> gen_*.{safetensors,json,npz} in $SEN2SR_DIR
# (parity-verify with `python -m sr.sr4rs_torch` locally). No TF on the cluster.
#
#   bash scripts/hpc/submit.sh sr/r4b_new.sh STAGE=tune  [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r4b_new.sh STAGE=fit   [SEED=n] [LOSS_ARM=arm]
#   bash scripts/hpc/submit.sh sr/r4b_new.sh STAGE=bench [SEED=n] [LOSS_ARM=arm]
#
# Toggles (all overridable at submit time):
#   REG=false           unregularised-GAN ablation — one switch turns off clip /
#                       cosine schedule / SR warmup, and tags the run/study/
#                       bench name with _noreg so it never mixes with the v2
#                       (regularised) runs.
#   SR_SNAPSHOT_EVERY=N fit-stage: save the fine-tuned SR net's WEIGHTS ONLY
#                       into <run dir>/sr_snapshots/ every N epochs (+ an
#                       epoch-0 init frame). Default here = 2; 0 disables.
#   LOSS_ARM=arm        any unet.losses.build_loss arm (empty = legacy loss).
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r4b_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="false"
SR_PAD=0

# NB the _all series ran this arm at REG=false by default. Here the default is
# the recipe-v2 arm, matching every other _new arm — the unregularised run is
# an explicit REG=false ablation, not the headline number.
REG="${REG:-true}"

SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"
LOSS_ARM="${LOSS_ARM:-}"

SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"  # 8 OOMs on 44GB: SR4RS runs 256-ch convs
                                     # (incl. a 9x9) at the full 512px grid

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
