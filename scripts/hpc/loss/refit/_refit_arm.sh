#!/bin/bash
# ONE stage of ONE seed of a pilot refit — the inner runner every per-arm script
# in this directory invokes as a SUBPROCESS.
#
# WHY A SUBPROCESS AND NOT `source`
# --------------------------------
# The arm scripts in loss/ and sr/ end by SOURCING the engine, because they run
# exactly one stage and the job is then over. A seed refit runs six stages
# (fit + bench val + bench test, twice), and `sr/_stages_tv.sh` calls `exit` on
# several paths — the SKIP_TEST guard, the bench's completion, its error
# branches. Sourced, the first `exit` would end the whole chain and every later
# stage would silently never run. So each stage gets its own shell.
#
# Everything the engine needs arrives through the environment; this file supplies
# only the pilot-protocol constants shared by every refit, then hands over.
#
#   env EXP_TAG=r0_new LOSS_ARM=gap_tl_ce SEED=1 STAGE=fit bash _refit_arm.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

: "${LOSS_ARM:?_refit_arm.sh needs LOSS_ARM}"
: "${SEED:?_refit_arm.sh needs SEED}"
: "${STAGE:?_refit_arm.sh needs STAGE (fit|bench)}"

EXP_TAG="${EXP_TAG:-r0_new}"
LABELS="${LABELS:-new}"

# --- the pilot arm's treatment (identical to LightningStudio/loss/_pilot_new.sh)
# r0 = bicubic upsampling, so no SR net and no SR weights are involved.
UPSAMPLER="bicubic"
FREEZE_SR="false"
SR_PAD=0

# --- pilot protocol, pinned here so no per-arm script can drift off it -------
# TRAIN_SPLITS=train keeps val a genuine holdout: it is what the theta sweep and
# the val bench read, so folding it in would make both a training score. It also
# gives the run dir / model_name their `_holdout` tag.
export TRAIN_SPLITS="train"
export REFIT_EPOCHS="${REFIT_EPOCHS:-50}"
export SKIP_TEST="${SKIP_TEST:-1}"          # fit does not touch test; the bench
                                            # stage reads it explicitly instead
export BENCH_SPLIT="${BENCH_SPLIT:-val}"
export TILE_METRICS="${TILE_METRICS:-apls,cldice}"
export BUFFER_PX="${BUFFER_PX:-1,2,3,4,5}"
export AP_BINS="${AP_BINS:-101}"
export SELECT_ON="${SELECT_ON:-iou_mean}"   # theta* criterion, as for seed 0

# --- compute shape: one L40S, eight cores ------------------------------------
export REFIT_GPUS="${REFIT_GPUS:-1}"
export SEARCH_GPUS="${SEARCH_GPUS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-7}"      # 8 cores minus the main process
export PRECISION="${PRECISION:-bf16-mixed}" # L40S is Ada: native bf16

# --- wandb: one run per (arm, seed), named so the grouping is obvious --------
# Without WANDB_NAME every refit lands as a random adjective-animal and telling
# seed 1 from seed 2 means opening each run.
export WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_loss_pilot_seeds}"
export WANDB_NAME="${WANDB_NAME:-${EXP_TAG}_${LOSS_ARM}_holdout_seed${SEED}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${EXP_TAG}_${LOSS_ARM}_holdout}"

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
