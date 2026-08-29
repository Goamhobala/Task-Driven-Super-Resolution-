#!/bin/bash
# =============================================================================
# RL-SERIES CAMPAIGN CONSTANTS — HPC (UCT l40s).
#
# CLUSTER TWIN of scripts/LightningStudio/sr/rl/_rl_common.sh. Every campaign
# constant below is byte-identical to that file's, and must stay so:
# tests/test_rl_campaign_hpc.py compares the two and fails on any divergence.
# The Studio file carries the full rationale for each constant — read it there
# and change it there first. docs/rl_lightning_campaign_plan.md (rev 3,
# 2026-08-29) is the authority; docs/sr_linear_probe.md remains the authority on
# the head itself and the θ*/AP protocol.
#
# WHY A CLUSTER COPY EXISTS. rl3/rl4 are the SR4RS arms: an 11.3 M-param
# generator forward at 512 px on every batch, with gradients in rl4. That arm
# OOMed a 24 GB L4 on 2026-08-29 (~18 GiB retained in the fp32 island; ~9 GiB
# with bf16), and the Studio tier runs two jobs at a time. The l40s partition
# has 48 GB cards and a queue, so the two heavy arms are cheaper to run here
# while the Studio lanes carry rl0/rl1/rl2.
#
# THE ONLY DELIBERATE DIFFERENCES FROM THE STUDIO FILE (all platform, none
# scientific):
#   1. no `env.sh` — the engine derives REPO_DIR/VENV_DIR from $HOME and
#      /scratch/$USER itself, and each arm sets its own SEN2SR_DIR;
#   2. NUM_WORKERS is left to the engine, which splits the SLURM allocation
#      (`--cpus-per-task`) across the stage's processes. The Studio file has to
#      override it because env.sh exports 0 there; here 0 never arrives, and the
#      engine's derivation is better than a hard 4. Budget --cpus-per-task 8+:
#      an rl arm backprops through FIVE parameters, so it is I/O-bound;
#   3. FIT_EARLY_STOP is a per-arm setting on this platform (rl3 on, rl4 off) —
#      see the block at the bottom.
#
# Everything is overridable at submit time (train.sbatch exports KEY=VALUE), but
# perturbing ONE arm voids every between-arm contrast in the series.
# =============================================================================

# --- The read-out ------------------------------------------------------------
HEAD="${HEAD:-linear}"
MONITOR="${MONITOR:-val_ap}"

# --- NO TUNING ANYWHERE (plan §1) -------------------------------------------
# One trial of one epoch: it writes best_params.yaml (which STAGE=fit requires)
# and it IS the plan's §4 gate-1 timing/VRAM pass. NOT N_TRIALS=0 — that is the
# rescue idiom and needs an existing COMPLETE trial to re-emit.
N_TRIALS="${N_TRIALS:-1}"
TUNE_EPOCHS="${TUNE_EPOCHS:-1}"
PATIENCE="${PATIENCE:-5}"   # the TUNE stage's stopper; inert at 1 epoch

# --- The pinned head lr (plan §1, §5) ---------------------------------------
# 3e-3 for all five arms. Licence to pin rather than search: the cluster rl3
# study found val_ap flat in lr across [~2.4e-3, ~7.9e-3] (≥30 trials, 0.043–
# 0.049 spread). A per-arm head lr voids every between-arm contrast.
HEAD_LR="${HEAD_LR:-3e-3}"
if ! awk -v v="$HEAD_LR" 'BEGIN { exit !(v + 0 == v && v >= 1e-3 && v <= 1e-2) }' \
     </dev/null 2>/dev/null; then
  echo "ERROR: HEAD_LR='${HEAD_LR}' is outside [1e-3, 1e-2]." >&2
  echo "  The rl3 study sampled lr in roughly [2.4e-3, 7.9e-3] and found the" >&2
  echo "  objective flat there — that flatness is the ONLY reason this campaign" >&2
  echo "  is allowed to pin the head lr instead of searching it. Outside that" >&2
  echo "  region there is no evidence, and a pinned guess would be worse than a" >&2
  echo "  search. (RL_LOOSE_OK=1 to override, for ALL FIVE arms or none.)" >&2
  if [ "${RL_LOOSE_OK:-0}" != "1" ]; then exit 2; fi
fi
LR_MIN="${LR_MIN:-$HEAD_LR}"
LR_MAX="${LR_MAX:-$HEAD_LR}"

# --- The budget: ONE 30-epoch run per arm-seed (plan §2) --------------------
# Frozen arms: 30 epochs head-only. Joint arms: 10 epochs with lr_sr held at
# EXACTLY 0, then 20 joint epochs on the rung's lr_sr. With lr_sr = 0 the joint
# arm's first 10 epochs ARE a frozen-arm run, so the branches diverge only at
# epoch 11 and the frozen arm's epochs 11–30 ARE the matched-budget control.
REFIT_EPOCHS="${REFIT_EPOCHS:-30}"
# SR_HOLD_EPOCHS is deliberately NOT set here — _rl_rung.sh owns it for the
# joint arms, behind its own ${VAR:-10} guard.
SR_WARMUP_EPOCHS="${SR_WARMUP_EPOCHS:-1.0}"

BATCH_SIZES="${BATCH_SIZES:-4}"
PRECISION="${PRECISION:-bf16-mixed}"   # l40s is Ada: native bf16

# --- Loss: wbce, PINNED, uniform across all five arms (plan §1) -------------
# λ copied from runs/sr_r0_new_wbce_holdout_seed0/best_params.yaml (a wbce tune
# at batch_size=4 under this same holdout protocol). Never re-searched.
LOSS_ARM="${LOSS_ARM:-wbce}"
PSTAR="${PSTAR:-bce}"                 # inert: `wbce` is a base arm, not a pstar_*
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-2.4789710497080004}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-2.4789710497080004}"
SEARCH_THETAS="${SEARCH_THETAS:-false}"
SEARCH_MIX_W="${SEARCH_MIX_W:-false}"

# --- Protocol: holdout, so the val curves exist (plan §2, §6) ---------------
# The campaign's primary evidence is within-run trajectories, which need a val
# loop every epoch; the R-series' merged protocol sets limit_val_batches: 0.
# Nothing is tuned on val here, so nothing is lost by holding it out.
TRAIN_SPLITS="${TRAIN_SPLITS:-train}"
FIT_VAL_LOOP="${FIT_VAL_LOOP:-1}"
BENCH_SPLIT="${BENCH_SPLIT:-test}"
SWEEP_SPLIT="${SWEEP_SPLIT:-val}"
SWEEP_CRITERION="${SWEEP_CRITERION:-iou}"
AP_BINS="${AP_BINS:-101}"
TILE_METRICS="${TILE_METRICS:-apls}"

# --- Recipe v2 dynamics ------------------------------------------------------
# CLIP is deliberately left at the engine's 1.0 and NOT restated: under
# HEAD=linear the engine routes it to --model.clip_sr and switches the
# Trainer-level global clip off (probe doc §6.2).
REG="${REG:-true}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"  # the hold gate lives inside this cosine
L2SP_LAMBDA="${L2SP_LAMBDA:-0.0}"

# --- Adaptive post-SR normalisation, ON and TRACKING ------------------------
ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
ADAPTIVE_NORM_M="${ADAPTIVE_NORM_M:-0.01}"
NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"

# --- Band guard: DEFANGED, NOT DELETED (plan §2, grid plan §3) --------------
# Loosened rails mean the raise can never fire while the warn stream and the
# variance-floor diagnostic keep printing; the stability envelope is read POST
# HOC from the logged adapt_std_b* curves at the nominal 0.5x / 4.0x crossings.
STD_BAND_RAISE_LO="${STD_BAND_RAISE_LO:-0.01}"
STD_BAND_RAISE_HI="${STD_BAND_RAISE_HI:-100}"
STD_BAND_ACTION="${STD_BAND_ACTION:-warn}"

# --- Snapshots: EVERY epoch --------------------------------------------------
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-1}"

# --- Bare-only: no hard constraint anywhere (plan §1) -----------------------
# Forced `off` (not `native`) so every arm carries the same _nohc tag and no
# arm's constraint state is implicit. On the SR4RS row the two resolve to the
# same operator, so rl3/rl4 stay behaviourally identical to r5/r4b.
SR_HC="${SR_HC:-off}"
SR_PAD="${SR_PAD:-0}"

# --- Compute -----------------------------------------------------------------
# One GPU per job at every stage: the "tune" is a 1x1 pass, so the engine's
# 2-GPU search fan-out would allocate a second card to do nothing.
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=<see the arm> \
#          scripts/hpc/train.sbatch --SCRIPT=sr/rl/rl3.sh STAGE=tune
# NUM_WORKERS is left to the engine (see the header): it splits
# --cpus-per-task across the stage's processes.
SEARCH_GPUS="${SEARCH_GPUS:-1}"
REFIT_GPUS="${REFIT_GPUS:-1}"

SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_rl_campaign}"

# --- Early stopping: PER ARM on this platform, and only here ----------------
# The Studio lanes run the fixed 30-epoch budget on all five arms. On the
# cluster rl3 opts into EarlyStopping and rl4 does not, so the knob is set in
# the ARM scripts, not here — the two arms genuinely differ in it, and a default
# in this file would hide which one chose what.
#
# Read the engine's FIT_EARLY_STOP block before changing either: an
# early-stopped arm's budget is a ceiling rather than a constant, its cosine
# does not complete, and its rows carry an _es<patience> tag that keeps them out
# of the fixed-budget rows of the same arm (including the Studio's).
#
# The engine WHITELISTS rl3 and refuses FIT_EARLY_STOP=1 on every other EXP_TAG,
# so this is not a knob the campaign can leak: rl4, the r-series and any future
# arm run the fixed budget whatever a submit line says.
#
# Nothing is assigned here on purpose: `: "${FIT_EARLY_STOP:=0}"` would BIND the
# variable, and an arm's own `FIT_EARLY_STOP="${FIT_EARLY_STOP:-1}"` (which runs
# after this file is sourced) would then see 0 and keep it. The engine defaults
# it to 0 for every arm that says nothing.

# --- Sanity: the controls must actually be constant -------------------------
if [ "$SEARCH_THETAS" != "false" ] || [ "$SEARCH_MIX_W" != "false" ] \
   || [ "$POS_WEIGHT_MIN" != "$POS_WEIGHT_MAX" ] || [ "$LR_MIN" != "$LR_MAX" ]; then
  echo "WARN: the rl-series controls are NOT frozen for this submit:" >&2
  echo "  lr=[${LR_MIN}, ${LR_MAX}]  pos_weight=[${POS_WEIGHT_MIN}, ${POS_WEIGHT_MAX}]" >&2
  echo "  SEARCH_THETAS=${SEARCH_THETAS} SEARCH_MIX_W=${SEARCH_MIX_W}" >&2
  echo "  Every rl contrast assumes the head, the loss and the budget are" >&2
  echo "  identical across arms. If this is deliberate it must be done to ALL" >&2
  echo "  FIVE arms, or the series is void. (RL_LOOSE_OK=1 to silence.)" >&2
  if [ "${RL_LOOSE_OK:-0}" != "1" ]; then exit 2; fi
fi
if [ "$TUNE_EPOCHS" != "1" ] || [ "$N_TRIALS" != "1" ]; then
  echo "NOTE: N_TRIALS=${N_TRIALS} TUNE_EPOCHS=${TUNE_EPOCHS} — the campaign pins" >&2
  echo "  1x1 (a config-writing + timing pass, not a search). Anything else is" >&2
  echo "  tuning, which plan §1 rules out for every arm." >&2
fi

echo "[rl] head=${HEAD}  monitor=${MONITOR}  loss=${LOSS_ARM}(λ=${POS_WEIGHT_MIN})  bs=${BATCH_SIZES}"
echo "[rl] head lr PINNED at ${HEAD_LR}  |  budget ${REFIT_EPOCHS} ep (joint arms hold the first 10)"
echo "[rl] bare-only: sr_hc=${SR_HC} sr_pad=${SR_PAD}  |  rails=[${STD_BAND_RAISE_LO}x, ${STD_BAND_RAISE_HI}x] action=${STD_BAND_ACTION}"
echo "[rl] protocol: train_splits='${TRAIN_SPLITS}' (val curves kept), sweep on ${SWEEP_SPLIT}, bench on ${BENCH_SPLIT}"
