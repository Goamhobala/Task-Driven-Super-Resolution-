#!/bin/bash
# RL3a — FROZEN SR4RS (pretrained WGAN-GP generator, torch port) WITH SEN2SR's
# FFT HARD CONSTRAINT mounted on top (pad 8) + LINEAR PROBE read-out, ROSA_New.
# FINAL protocol (tune on train/val -> refit on train+val -> report on test).
#
# The hard-constraint variant of rl3_new (docs/hc_2x2_plan.md x
# docs/sr_linear_probe.md); see rl1b_new.sh for the 2x2 layout.
#
# rl3a - rl3 is the constraint's effect on an UNANCHORED generator's image, read
# by a probe with no capacity to compensate. This is where the effect should be
# largest anywhere in the project: SR4RS's output is unanchored (the root cause
# the adaptive-norm work addressed), and the splice pins its low frequencies to
# the bicubic-upsampled input, which is exactly the missing anchor. Note the
# consequence for interpretation: part of any gain here is radiometric
# re-anchoring that `pre` recalibration would also deliver, so this arm is
# evidence about the MECHANISM (low-frequency pinning), not an argument that the
# constraint is the only way to get it.
#
# It is ALSO stage 1 of the constrained-SR4RS LP-FT pair: rl4a_new.sh
# warm-starts its probe from THIS arm's final ckpt, so this fit must COMPLETE
# before rl4a's tune starts. It cannot warm-start from rl3 — different input
# distribution (docs/sr_linear_probe.md §2).
#
# The bundle, applied identically to both rows of the 2x2: clamp(., min=0) ->
# FFT splice with the shipped mask -> reflect pad 8 with output crop. The mask
# is SEN2SR-Lite's shipped hard_constraint.safetensor reused BYTE-FOR-BYTE (a
# single 512x512 sigma=35 Gaussian, r = 0.14, the in-training optimum Table 4
# reports for the Non-Reference RGBN x4 task). It was optimised for SEN2SR-Lite,
# NOT for SR4RS, so any gain here is a LOWER BOUND on what a tuned constraint
# could give this generator. Pre-registered: state it in the write-up rather
# than tuning r per arm. data.crop_size is 128 for every arm, so the 512px mask
# applies unchanged (pad 8 grows it to 576 by the same resize r2a uses — the
# cutoff stays at its fraction of Nyquist).
#
# Normalisation: the engine default (`post`) is kept, as in rl3 — all rl arms
# share one normalisation policy, and changing it for one arm would put the
# nuisance variable back on the treatment. That matters more here than usual,
# because the constraint ITSELF re-anchors the output; if `pre` were switched on
# for this arm alone, the two effects would be inseparable.
#
# Prerequisites:
#   * gen_*.{safetensors,json,npz} in $SEN2SR_DIR (extract_sr4rs.py locally,
#     parity-verify with `python -m sr.sr4rs_torch`; no TF on the cluster).
#   * SEN2SR-Lite's model dir on scratch for HC_MASK_PATH below — SEN2SR_DIR
#     points at SR4RS_RGBN here, which ships no mask. Prefetch with
#     sr.sen2sr_loader.download_sen2sr on a login node.
#
# Budget: pad 8 grows the SR grid 512 -> 576 px on top of SR4RS's 256-channel
# convs (incl. a 9x9), so allow ~1.3x rl3_new's per-trial walltime. If bs=4 OOMs
# at 576 px the fix is activation checkpointing on the SR4RS blocks (numerically
# identical) — NEVER a batch-size change, which is a pinned between-arm constant
# (_rl_common.sh, §6.1).
#
# lr_sr is auto-skipped (frozen SR).
#
#   bash scripts/hpc/submit.sh sr/rl3a_new.sh STAGE=tune  [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl3a_new.sh STAGE=fit   [SEED=n]
#   bash scripts/hpc/submit.sh sr/rl3a_new.sh STAGE=bench [SEED=n]
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

EXP_TAG="rl3a_new"
LABELS="new"
UPSAMPLER="sr4rs"
FREEZE_SR="true"
# The HC lane. Pad travels with the constraint: the FFT splice assumes a
# periodic patch, so edge discontinuities ring at the borders — and a purely
# spectral probe is the worst possible consumer of a border ring, since it
# cannot contextually discount one.
SR_PAD=8
SR_HC="on"
HC_MASK_PATH="${HC_MASK_PATH:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"

SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"

source "$REPO_DIR/scripts/hpc/sr/_rl_common.sh"
source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
