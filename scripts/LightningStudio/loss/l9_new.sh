#!/bin/bash
# Pilot arm 9 — gap_tl_ce at matched λ*: the GL+TL blend (Nanni et al. 2024
# Table 2 — their best compound on 3 of 4 datasets). One pixel-slot arm:
# mean-normalized gap + TL maps summed inside a single weighted CE (see
# make_gap_tl_ce; the exact 0.5·gap+0.5·tl identity holds at λ=1, at λ*>1 it
# is the prescribed single-map composition).
#
#   bash scripts/LightningStudio/run.sh loss/l9_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l9_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="gap_tl_ce"
GAP_R="${GAP_R:-4}"
GAP_K="${GAP_K:-60.0}"
TL_ELL="${TL_ELL:-5}"
TL_THETA="${TL_THETA:-0.375}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
