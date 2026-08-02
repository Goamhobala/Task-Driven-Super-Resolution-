#!/bin/bash
# Pilot arm 11 — sdice STANDALONE (Phase B). Squared-denominator Dice
# (Milletari et al. 2016), exactly the form Xu et al. 2023 advertise as best
# F1 on DeepGlobe. Region-only arm (no pixel slot), so λ* does not apply.
# Phase B rule: standalone first; compose with P* (l13) only if this beats
# or ties P*.
#
#   bash scripts/LightningStudio/run.sh loss/l11_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l11_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="sdice"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
