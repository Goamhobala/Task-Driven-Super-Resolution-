#!/bin/bash
# Pilot arm 16 — dice STANDALONE (Phase B reference). Plain Dice, the
# region-family baseline the sdice (l11) / lcdice (l12) variants are judged
# against (Xu et al. 2023 test all three standalone). Region-only arm: the
# tune searches lr+batch only (no λ, no θ — gated out automatically).
#
#   bash scripts/LightningStudio/run.sh loss/l16_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l16_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l16_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="dice"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
