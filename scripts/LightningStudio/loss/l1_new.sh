#!/bin/bash
# Pilot arm 1 — bce (OPTIONAL literature floor). STRICTLY plain CE: the only
# arm that ignores the shared λ*. NB plain BCE has never actually run — the
# 10 m table's "bce" was wbce(5) via the pre-07-21 default. Cheapest run in
# the menu; skip it if the compute budget is tight and the report does not
# need the floor.
#
#   bash scripts/LightningStudio/run.sh loss/l1_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l1_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="bce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
