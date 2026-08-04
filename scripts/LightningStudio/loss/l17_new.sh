#!/bin/bash
# Pilot arm 17 — gap_t2_ce (combination series): Gap + TL-line + T2
# quarter-circle curvature maps in ONE normalized CE. Single-model
# counterpart of the sum-rule ENSEMBLES in Nanni/Giannini (verified: their
# GL+TL(+DI) fuses separately trained networks). Fully tuned like every
# other arm (2026-08-03c: symmetric per-arm tune budget is the fairness
# rule; the earlier fit-only inheritance was a compute concession, dropped).
# Searches lr + λ + tl_theta + gap_theta.
#
#   bash scripts/LightningStudio/run.sh loss/l17_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l17_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l17_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="gap_t2_ce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
