#!/bin/bash
# Pilot arm 18 — gap_t4_ce (combination series): Gap + TL-line + T4
# semicircle (tight-turn) curvature maps in ONE normalized CE. Fully tuned
# (see l17_new.sh, 2026-08-03c). NB the 10 m run found t4 significantly
# WORSE than tl standalone; this tests whether that survives blending +
# 2.5 m. Searches lr + λ + tl_theta + gap_theta.
#
#   bash scripts/LightningStudio/run.sh loss/l18_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l18_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l18_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="gap_t4_ce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
