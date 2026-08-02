#!/bin/bash
# Pilot arm 7 — B*+cldice (Phase C): (1−α)·B* + α·clDice, α=0.3. Set BSTAR to
# the Phase B winner (a base arm name; PSTAR too if B* is a pstar_* compound).
# CL_ITERS at 2.5 m: soft-skeleton iterations must cover the widest road
# half-width in px under the HR buffers — default 8 here (VERIFY once against
# the mask_new_2pt5 buffer widths; the 10 m value was 5).
# Skeleton warmup is pinned by the pilot harness (start 15, ramp 5, of E=50).
#
#   bash scripts/LightningStudio/run.sh loss/l7_new.sh STAGE=fit BSTAR=wbce
#   bash scripts/LightningStudio/run.sh loss/l7_new.sh STAGE=fit BSTAR=pstar_sdice PSTAR=wbce
set -euo pipefail
BSTAR="${BSTAR:-wbce}"      # <- Phase B winner
LOSS_ARM="${BSTAR}+cldice"
CL_ALPHA="${CL_ALPHA:-0.3}"
CL_ITERS="${CL_ITERS:-8}"   # 2.5 m: k >= max half-width px (10 m used 5)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
