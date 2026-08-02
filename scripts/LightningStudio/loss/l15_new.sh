#!/bin/bash
# Pilot arm 15 — balance_ce (Phase A, the ADAPTIVE-λ cell). BalanCE (Xie & Tu
# 2015 HED; Xu et al. 2023's recall-topper): per-batch β = negative-pixel
# fraction, i.e. inverse-frequency class weighting recomputed every batch
# (λ_t = neg/pos ≈ 34 at ROSA_New's 2.84% train road density). Completes the
# class-balance axis: l1 bce (λ=1 floor) · l10 wbce (fixed tuned λ*) · this
# (adaptive λ_t). Deliberately IGNORES the shared λ* overlay.
# NB fixed-β BalanCE would be redundant — under §4.4 normalization it is
# exactly wbce(λ=β/(1−β)) (unit-tested) — only the adaptive form earns a slot.
#
#   bash scripts/LightningStudio/run.sh loss/l15_new.sh STAGE=fit
#   bash scripts/LightningStudio/run.sh loss/l15_new.sh STAGE=bench
set -euo pipefail
LOSS_ARM="balance_ce"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_pilot_new.sh"
