#!/bin/bash
# RL1 — FROZEN SEN2SR-Lite x4 under a LINEAR PROBE read-out, BARE. Lightning.
#
# rl1 - rl0 = the linear spectral separability the PRETRAINED generator adds,
# with no task gradient having ever touched it. That difference is the honest
# "what does SR give you for free" number, and it is the baseline the joint arm
# rl2 is measured against.
#
# BARE, so this twins r1b and NOT r1a: SR_HC=off strips SEN2SR-Lite's shipped
# FFT hard constraint (and SR_PAD=0 with it — there is no splice to ring at the
# patch border, so there is nothing for a pad to mitigate). That is the campaign
# rule, not an oversight: the joint arms measure the UNCONSTRAINED upper bound
# on degeneracy, and a frozen control carrying a constraint its joint twin lacks
# would confound the contrast. The frozen splice contrast (r1a - r1b
# separability) comes free from LDA on cached outputs; a SEN2SR HC-on pair is
# the cheap add-back if it is ever demanded.
#
# SEEDS: this arm gets the +1 second seed (plan §2) along with the extreme
# rungs — it is the reference point for the whole SEN2SR row.
#
#   bash scripts/LightningStudio/run.sh sr/rl/rl1.sh STAGE=tune
#   bash scripts/LightningStudio/run.sh sr/rl/rl1.sh STAGE=fit   [SEED=1]
#   bash scripts/LightningStudio/run.sh sr/rl/rl1.sh STAGE=bench [SEED=1]
set -euo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RL_DIR/_rl_common.sh"
source "$RL_DIR/../../env.sh"

EXP_TAG="${EXP_TAG:-rl1_new}"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="true"    # preprocessing only: no gradient reaches the generator
SEN2SR_DIR="${SEN2SR_DIR:-${INSTAROAD_ROOT}/models/SEN2SRLite_RGBN}"

source "$LS_DIR/sr/_stages_tv.sh"
