#!/bin/bash
# Pilot arm 10 — wbce (THE INCUMBENT, and the pilot's shared-tune arm).
# Uniform CE at class balance λ: the winner of the (confounded) 10 m Phase A
# and the loss the SR prelims run on. Post-2026-07-30b code: λ passes through
# the §4.4 normalizer (the old kwarg form inflated the loss scale ~mean(W)).
#
# This arm's STAGE=tune IS the pilot's one shared Optuna search: its
# best_params.yaml (lr, batch, pos_weight λ*) is the frozen screening config
# every other l*_new.sh arm reuses (see _pilot_new.sh). Run it first:
#
#   bash scripts/LightningStudio/run.sh loss/l10_new.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh loss/l10_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l10_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="wbce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
