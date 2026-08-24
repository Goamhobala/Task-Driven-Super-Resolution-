#!/bin/bash
# RL2b — JOINT task-driven fine-tuning of the BARE SEN2SR-Lite generator (FFT
# hard constraint OFF, no pad) under a LINEAR PROBE read-out, ROSA_New. FINAL
# protocol (tune on train/val -> refit on train+val -> report on test).
#
# The hard-constraint variant of rl2_new; twin of r2b_new. See rl1b_new.sh for
# the 2x2 layout and the bundle definition.
#
# The contrast this arm exists for: rl2 - rl2b, i.e. does the constraint change
# how far the task loss can repurpose the generator? Mechanistically it should:
# the splice pins the low frequencies of the output to the bicubic-upsampled
# input, so the mask-painting degeneracy of §7 (the "SR image" collapsing into a
# road-probability map in reflectance coordinates) can only be written into the
# HIGH frequencies while the constraint is on. rl2b removes that ceiling with
# the architecture held fixed — the same 240 k parameters, the same loss, the
# same probe. If mask-painting is a real failure mode, this is the SEN2SR arm
# where it is free to happen, and rl2b - rl2 measures the headroom the
# constraint was taking away.
#
# Read alongside rl4 (bare SR4RS): rl2b and rl4 are the two unconstrained joint
# cells, so their diagnostics answer "is it the constraint or the architecture"
# without the r-series' decoder in the way.
#
# Init: LP-FT from rl1b_new's FINAL head — the BARE frozen twin, not rl1. A
# probe converged on HC-spliced inputs is not converged on bare-generator
# inputs, so warm-starting from rl1 would mix "SR unfrozen" with "the head
# started somewhere else", which is precisely what LP-FT is here to remove.
# _warm_head_tv.sh resolves the stage-1 run dir including the _nohc tag, so
# rl1b_new's fit must COMPLETE (same SEED/LOSS_ARM/REG/SR_HC) before this tune.
#
# DO NOT describe rl2b - rl1b as "the value of adaptation" (§8): it is
# adaptation PLUS ~240 k newly-trainable generator parameters against rl1b's 5.
# Here the generator IS the model.
#
# Gate C (docs/sr_linear_probe.md §10.6) applies to this arm as it does to rl2 —
# read the snapshots before drawing an image-quality conclusion. Bands
# collapsing toward mutual correlation ~1 means conv-net capacity was measured,
# not image improvement. No HR reference exists for these tiles, so this
# supports a DRIFT claim, never a fidelity one — and with the DC anchor gone,
# expect the drift to be larger here than in rl2 by construction.
#
#   bash scripts/hpc/submit.sh sr/rl1b_new.sh STAGE=tune ; ... STAGE=fit
#   bash scripts/hpc/submit.sh sr/rl2b_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl2b_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl2b_new.sh STAGE=bench [SEED=n]
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl2b_new"
LABELS="new"
UPSAMPLER="sen2sr"
FREEZE_SR="false"
SR_PAD=0
SR_HC="off"

# Frame-by-frame replay of what the task loss writes into the generator. This is
# the evidence for §7, not a demo — and this arm is the one with nothing pinning
# its low frequencies, so it should stay on.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

STAGE1_TAG="rl1b_new"
source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_warm_head_tv.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
