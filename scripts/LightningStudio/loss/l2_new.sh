#!/bin/bash
# Pilot arm 2 — gap_ce at matched λ* (Yuan & Xu 2022 GapLoss, r=4 = the
# paper's 9x9 window; at 2.5 m that is a 22.5 m buffer, near the VHR regime
# the paper tuned for, so the hp grid is dropped). λ* arrives via the shared
# best_params.yaml overlay (model.pos_weight) and composes into the gap map —
# the Phase A question is spatial weighting ON TOP OF class weighting.
#
#   bash scripts/LightningStudio/run.sh loss/l2_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l2_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="gap_ce"
GAP_R="${GAP_R:-4}"       # paper default (9x9 => r=4)
GAP_K="${GAP_K:-60.0}"    # paper's tuned K

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
