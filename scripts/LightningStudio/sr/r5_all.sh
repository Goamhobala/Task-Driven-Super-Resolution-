#!/bin/bash
# R5 — FROZEN SR4RS preprocessing (pretrained WGAN-GP generator, torch port),
# ROSA_all. The SR4RS analogue of R1: R5 - R0 = value of the frozen pretrained
# GAN SR on SA data. Stage 1 of the staged SR4RS protocol: r6_all.sh
# warm-starts its UNet from this arm's fitted ckpt.
# No pad variant: SR4RS has no FFT hard constraint, so no Gibbs border ring,
# and padding only inflates the (already heavy) 512 px SR compute.
#
# Prerequisites: gen_*.safetensors/json/npz in $SEN2SR_DIR (extract_sr4rs.py
# locally + parity via `python -m sr.sr4rs_torch`; no TF on the cluster).
#
#   bash scripts/LightningStudio/run.sh sr/r5_all.sh STAGE=tune [SEED=n]
#   bash scripts/LightningStudio/run.sh sr/r5_all.sh STAGE=fit  [SEED=n]
set -euo pipefail
# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="r5_all"
LABELS="all"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
SR_PAD=0
SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SR4RS_RGBN}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"  # 8 OOMs on 44GB: SR4RS runs 256-ch convs
                                     # (incl. a 9x9) at the full 512px grid

source "$LS_DIR/sr/_stages.sh"
