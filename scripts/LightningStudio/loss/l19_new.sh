#!/bin/bash
# Pilot arm 19 — gap_t2t4_ce (combination series): the full blend — Gap +
# TL-line + T2 + T4 curvature (12 kernels on the TL side; the base-reset-
# before-cap fix in tl_weight_map exists for exactly this arm). Fully tuned
# (see l17_new.sh, 2026-08-03c). Searches lr + λ + tl_theta + gap_theta.
#
#   bash scripts/LightningStudio/run.sh loss/l19_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l19_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l19_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="gap_t2t4_ce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
