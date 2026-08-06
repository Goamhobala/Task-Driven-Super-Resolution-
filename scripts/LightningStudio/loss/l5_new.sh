#!/bin/bash
# Pilot arm 5 — pstar_dice (Phase B): (1−mw)·P* + mw·Dice, mw fixed 0.5 on
# the SR path (mix_w not exposed on JointSRUNetLightning; thread it through
# sr/model.py if Phase B ever searches it). P* = Phase A winner: gap_tl_ce.
# NB do NOT rename l5_all.sh for this — the *_all loss scripts drive the OLD
# 10 m unet engine (different pipeline, dataset, store).
#
#   bash scripts/LightningStudio/run.sh loss/l5_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l5_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l5_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="pstar_dice"
PSTAR="${PSTAR:-gap_tl_ce}"   # Phase A winner (benched at θ*, val)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
