#!/bin/bash
# Pilot arm 4b — t4_ce STANDALONE (attribution axis, 2026-08-03c): TL-line +
# T4 semicircle curvature, no Gap map. See l4a_new.sh. Prior: the 10 m run
# had t4 significantly below tl — the interesting outcome is whether 2.5 m
# (where tight curves are finally resolvable) reverses that.
# Searches lr + λ + tl_theta.
#
#   bash scripts/LightningStudio/run.sh loss/l4b_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l4b_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l4b_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="t4_ce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
