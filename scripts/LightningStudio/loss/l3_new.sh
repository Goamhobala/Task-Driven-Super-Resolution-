#!/bin/bash
# Pilot arm 3 — tl_ce at matched λ* (Nanni et al. 2024 Topological Loss,
# ℓ=5 / θ=0.375 = paper values; at 2.5 m ℓ=5 px is a 12.5 m lookahead, near
# the VHR regime — hp grid dropped). λ* via the shared overlay, composed
# into the TL map.
#
#   bash scripts/LightningStudio/run.sh loss/l3_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l3_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="tl_ce"
TL_ELL="${TL_ELL:-5}"
TL_THETA="${TL_THETA:-0.375}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
