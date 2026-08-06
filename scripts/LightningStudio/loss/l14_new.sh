#!/bin/bash
# Pilot arm 14 — pstar_lcdice (Phase B compound, CONDITIONAL): (1−mw)·P* +
# mw·lcDice. Run only if l12 (lcdice standalone) beat or tied P*. Set PSTAR
# to the Phase A winner:
#
#   bash scripts/LightningStudio/run.sh loss/l14_new.sh STAGE=fit PSTAR=wbce
set -euo pipefail
LOSS_ARM="pstar_lcdice"
PSTAR="${PSTAR:-gap_tl_ce}"   # Phase A winner (benched at θ*, val)
# mix_w is FIXED at build_loss's 0.5 for the pilot (JointSRUNetLightning does
# not expose it; thread it through sr/model.py if Phase B ever searches it).

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
