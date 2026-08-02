#!/bin/bash
# Pilot arm 12 — lcdice STANDALONE (Phase B). Log-cosh Dice (Jadon 2020),
# the form Xu et al. 2023 advertise as best F1 on Massachusetts. Region-only
# arm, λ* does not apply. Standalone first; compose with P* (l14) only if
# this beats or ties P*.
#
#   bash scripts/LightningStudio/run.sh loss/l12_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l12_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="lcdice"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
