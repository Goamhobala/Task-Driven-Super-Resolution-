#!/bin/bash
# Pilot arm 4a — t2_ce STANDALONE (attribution axis, 2026-08-03c): TL-line +
# T2 quarter-circle curvature, no Gap map. Decision-irrelevant given the
# gap-blends dominate, but fills the {tl,t2,t4} x {alone, +gap} attribution
# matrix — "does curvature help alone, or only in composition?" — and feeds
# the weight-map visualisation figure. Searches lr + λ + tl_theta.
#
#   bash scripts/LightningStudio/run.sh loss/l4a_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l4a_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l4a_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="t2_ce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
