#!/bin/bash
# Pilot arm 8 — B*+skelrec (Phase C): B* + w·SkeletonRecall (Kirchhoff et al.
# 2024), additive as in the paper (w=1, tube r=1). NB the effective anchor
# weight differs from l7's convex mix — the known asymmetry, flagged in the
# protocol. GT-side skeletons only, so near-free at train time.
# Skeleton warmup pinned by the pilot harness (start 15, ramp 5, of E=50).
#
#   bash scripts/LightningStudio/run.sh loss/l8_new.sh STAGE=fit BSTAR=wbce
#   bash scripts/LightningStudio/run.sh loss/l8_new.sh STAGE=fit BSTAR=pstar_sdice PSTAR=wbce
set -euo pipefail
BSTAR="${BSTAR:-wbce}"      # <- Phase B winner
LOSS_ARM="${BSTAR}+skelrec"
SKEL_W="${SKEL_W:-1.0}"
SKEL_RADIUS="${SKEL_RADIUS:-1}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
