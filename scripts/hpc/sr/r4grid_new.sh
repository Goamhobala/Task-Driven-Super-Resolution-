#!/bin/bash
# R4GRID — the lr_sr x hard-constraint MECHANISM ablation on the SR4RS row.
# ONE parameterised script for all 8 cells: the SR4RS mirror of r2grid_new.sh.
#
# WHAT IS DIFFERENT FROM r2grid (and it is only three things)
# ----------------------------------------------------------
#   1. The generator is SR4RS (11.3 M trainable) instead of SEN2SR-Lite
#      (187 K), so SEN2SR_DIR points at SR4RS_RGBN and the cells are dearer.
#   2. SR4RS SHIPS NO HARD CONSTRAINT, so the HC=on lane mounts SEN2SR-Lite's
#      hard_constraint.safetensor on it — byte-for-byte the shipped file, as
#      the formal r4a arm does. Its cutoff is the value the SEN2SR paper
#      optimised; re-deriving one would make the two lanes different operators.
#   3. THERE IS NO TUNE. lr is PINNED at 2e-4 and lr_sr is the cell coordinate,
#      so both axes are fixed and there is nothing left to search. The overlay
#      is planted by grid/run_pool_r4.sh; STAGE=fit consumes it directly.
#      r2grid searched lr per cell because lr and lr_sr interact — pinning it
#      here trades that control for eight fewer searches, so BETWEEN-CELL
#      differences now carry lr mis-specification as well as the lr_sr dose.
#      Read the trajectories, not the deltas.
#
# ONE SEED (0). No replicates: same pre-commitment as r2grid — these cells are
# DESCRIPTIVE, and at n=1 a between-cell metric difference is not interpreted
# against seed noise.
#
#   S=sr/r4grid_new.sh; C="HC=off LRSR=1e-4"        # one cell's coordinates
#   sbatch --gres=gpu:1 -J r4grid_off_ls1e-4_fit   -o slurm-%x-%j.txt \
#     scripts/hpc/train.sbatch --SCRIPT=$S STAGE=fit   $C
#   sbatch --gres=gpu:1 -J r4grid_off_ls1e-4_bench -o slurm-%x-%j.txt \
#     scripts/hpc/train.sbatch --SCRIPT=$S STAGE=bench $C
#
# -J/-o are not decoration here: ALL EIGHT CELLS SHARE THIS SCRIPT, so without
# a per-cell job name `squeue` shows eight identical rows and train.sbatch's
# default log (slurm-%j.txt) names them by jobid alone. --gres=gpu:1 because
# SEARCH_GPUS=1 below: the header's gpu:2 would hold a second idle card for the
# tune as well as the fit.
#
# Run order (§8) — extremes first, they carry the interaction and the probable
# collapses; the 1e-6/1e-7 cells interpolate toward the formal arms and are the
# least informative:
#   1. HC=off LRSR=1e-4   2. HC=on LRSR=1e-4
#   3. HC=off LRSR=1e-5   4. HC=on LRSR=1e-5
#   5. the 1e-6 / 1e-7 cells as time permits
#
# WHAT THIS ANSWERS THAT THE TUNED 2x2 CANNOT
# -------------------------------------------
# r2a/r2b search (lr, lr_sr) jointly, and their searches select lr_sr near the
# floor — at which point the SR barely moves and the constraint has nothing
# visible to act on. Here lr_sr is PINNED per cell (LR_SR_MIN == LR_SR_MAX, the
# pos_weight pattern: the tuned constant lands in best_params.yaml, so the fit
# stage needs no change), so the dose-response of task-driven adaptation is
# observed BY CONSTRUCTION rather than reached only if a 30-trial TPE search
# happens to visit it. It is also immune to pruner herding: MedianPruner ranks
# from epoch 2 and plausibly disfavours slow-recovering adaptation rates, but
# with lr_sr pinned that mechanism cannot select what gets explored.
#
# ONLY lr is searched (1-D, 15 trials). Re-tuning lr per cell is REQUIRED, not
# optional: lr and lr_sr interact, and reusing a foreign lr would confound the
# grid with lr mis-specification.
#
# DISJOINTNESS
# ------------
# EXP_TAG is derived from the cell (r2grid_<hc>_ls<mantissa>e<exp>), so run
# dirs, Optuna studies and benchmark rows are disjoint from the formal arms —
# and from each other — by construction. The loosened rails add _rails on top
# (see below). Nothing here can ever land in an r2a_new / r2b_new row.
#
# PRE-COMMITMENT (§4), made before the first grid run:
#   Grid cells are DESCRIPTIVE. Test is reported to complete the record, and no
#   cell is ever promoted to replace or restate a formal arm's result, whatever
#   its test score. At one seed, between-cell metric differences are not
#   interpreted against seed noise; trajectories, snapshots and crossings are
#   the primary evidence. AP leads the figure (threshold-free, smooth at n=1).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

# --- the two cell coordinates ------------------------------------------------
HC="${HC:?set HC=on|off — the hard-constraint lane (r2a lane vs r2b lane)}"
LRSR="${LRSR:?set LRSR=<lr_sr> — the pinned adaptation rate, e.g. 1e-4}"

case "$HC" in
on | off) : ;;
*)
  echo "ERROR: HC must be on|off, got '${HC}'." >&2
  exit 2
  ;;
esac

# lr_sr -> tag. Normalised to 1-2 significant figures in scientific notation so
# 1e-4, 1E-04 and 0.0001 all name the SAME cell, and a typo'd 1.4e-5 names a
# different one instead of silently reusing a neighbour's run dir and study.
LS_TAG=$(awk -v v="$LRSR" 'BEGIN {
  if (v + 0 != v || v <= 0) exit 1
  split(sprintf("%.1e", v), a, "e")
  m = a[1]; sub(/\.0$/, "", m)
  printf "%se%d", m, a[2] + 0
}' </dev/null) || {
  echo "ERROR: LRSR must be a positive number, got '${LRSR}'." >&2
  exit 2
}

EXP_TAG="r4grid_${HC}_ls${LS_TAG}"

# --- the lane: HC on = r2a's configuration, HC off = r2b's -------------------
# The pad travels with the constraint. The bare lane has no FFT splice, so
# there is no Gibbs ringing at the patch border for a pad to mitigate — this is
# the same pairing the formal 2x2 uses (docs/hc_2x2_plan.md §4, "Scheme B").
USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
if [ "$HC" = "on" ]; then
  # `on`, not `native`: SR4RS has no native constraint to fall back on, so the
  # bundle is MOUNTED explicitly and the engine hard-errors without the mask.
  SR_HC="on"
  SR_PAD=8
  export HC_MASK_PATH="${HC_MASK_PATH:-/scratch/${USER_NAME}/InstaRoad/models/SEN2SRLite_RGBN/hard_constraint.safetensor}"
else
  SR_HC="off" # raw generator: no positivity clamp, no frequency splice
  SR_PAD=0
fi

# --- everything else = the formal r2a/r2b configuration ----------------------
LABELS="new" # ROSA_New, pre-rasterised 2.5 m masks
UPSAMPLER="sr4rs"
# SR4RS weights live in their OWN dir; the engine default (SEN2SRLite_RGBN)
# holds no gen_weights.safetensors and the fit dies at the pre-flight check.
export SEN2SR_DIR="${SEN2SR_DIR:-/scratch/${USER_NAME}/InstaRoad/models/SR4RS_RGBN}"
FREEZE_SR="false" # cold joint fine-tuning
REG="${REG:-true}"

# §7.3: EVERY epoch. A cell that dies at step ~2k must still leave its snapshot
# sequence behind — the snapshot strips ARE the mechanism figure, and a collapse
# with no frames is an observation that cannot be shown. SEN2SR-Lite snapshots
# are ~2 MB, so 100 of them is a rounding error next to the checkpoints.
SR_SNAPSHOT_EVERY="${SR_SNAPSHOT_EVERY:-2}"

# --- the pinned axis ---------------------------------------------------------
# min == max makes Optuna's log-uniform suggest a CONSTANT, which then lands in
# best_params.yaml exactly like a searched value. This is the only reason the
# fit stage needs no special-casing.
export LR_SR_MIN="$LRSR"
export LR_SR_MAX="$LRSR"

# --- the OTHER pinned axis: lr ----------------------------------------------
# 2e-4 is where every tuned arm of this family landed (r2a 2.05e-4, r2b 2.61e-4,
# r4b 1.91e-4, r3b 1.92e-4), so it is the family's consensus rate rather than an
# arbitrary constant. Pinned the same way lr_sr is, so a tune -- if anyone runs
# one -- suggests the constant and lands it in best_params.yaml unchanged.
export LR_MIN="${LR_MIN:-2e-4}"
export LR_MAX="${LR_MAX:-2e-4}"
# Kept only so a stray STAGE=tune is cheap and degenerate rather than a 15x10
# search of a one-point space. The pool never runs it.
N_TRIALS="${N_TRIALS:-1}"
TUNE_EPOCHS="${TUNE_EPOCHS:-1}"
PATIENCE="${PATIENCE:-5}"
SEARCH_GPUS="${SEARCH_GPUS:-1}"
BATCH_SIZES="${BATCH_SIZES:-4}" # between-arm constant of the whole SR series

# --- fit: the pre-registered budget, ONE seed, no replicates -----------------
REFIT_EPOCHS="${REFIT_EPOCHS:-100}"
REFIT_GPUS="${REFIT_GPUS:-1}"

# --- band-guard policy: LOOSENED RAILS, NOT DISABLED (§3) --------------------
# The grid's purpose includes observing the collapse dynamics the production
# guard exists to kill, so let the runs develop naturally: with these rails the
# raise can never fire, while the warn stream and the variance-floor diagnostic
# keep printing. Do NOT reach for adaptive_norm_check_every=0 instead — that
# silences both, i.e. deletes the measurement.
#
# They apply to the TUNE stage as well, and must: at pinned 1e-4 in the bare
# lane the production band would prune every trial and the study would produce
# no best_params at all.
#
# IDENTICAL across all 8 cells — the grid's internal pair rule. The stability
# envelope is read POST HOC from the logged adapt_std_b* curves (the step at
# which any band crosses 0.5x / 4.0x of its starting std), which is strictly
# more information than the raise gave: the trajectory after the crossing is
# retained. Expected endpoints in the extreme cells are std -> variance floor,
# rs/band_std gain divergence, NaN loss, adapt_skipped_batches climbing. A run
# that ends in numerical death at step X is a VALID OBSERVATION — record X.
export STD_BAND_RAISE_LO="${STD_BAND_RAISE_LO:-0.01}"
export STD_BAND_RAISE_HI="${STD_BAND_RAISE_HI:-100}"

# ...and nothing in the grid ABORTS on a band exit, at either stage.
#
# The rails alone do not guarantee that. They make the exit unlikely, but a
# cell that genuinely collapses to the variance floor can still cross 0.01x,
# and in the TUNE that would prune the trial (sr.tune's own default is
# `raise`). With lr_sr pinned, pruning is the wrong response twice over: the
# trial's lr vanishes from the ranking, and if every trial of a cell prunes the
# study writes no best_params at all — the guard deciding which cells exist,
# which is exactly what §3 rules out. So the search runs `warn` as well.
#
# What is NOT given up: MedianPruner still prunes on the objective, so a
# hopeless trial is still cut short — on its SCORE, which is the criterion that
# belongs in a search, rather than on the adapter's moments. And a trial that
# goes numerically dead returns a non-finite objective, which Optuna records as
# a failed trial and steps past; the study survives either way.
#
# The refit was never at risk (joint_sr.yaml pins `warn`), but it is set
# explicitly here so both stages of a cell read one policy from one place.
export STD_BAND_ACTION="${STD_BAND_ACTION:-warn}"

# --- the loss: a frozen control, copied from the formal arm ------------------
# Pinned verbatim from scripts/hpc/sr/refit/r2a_new_gap_ce.sh (seed 66's tune,
# best val_ap=0.5950); r2b_new_nohc_gap_ce.sh carries the identical block, so
# the grid sits on the same loss surface as both arms it anchors against. If
# the final r2a/r2b arms ever move off this block, MOVE THIS TOO — the
# anchoring in §5 is only meaningful while they match.
#
# NOTE (2026-08-27): docs/lrsr_grid_ablation_plan.md §2 names the final loss
# `pstar_sdice`/`gap_ce`, while BOTH refit scripts on disk run gap_ce/gap_ce
# (θ*=0.55836, λ*=4.77222, mix_w unconsumed by a non-pstar arm). This script
# follows the SCRIPTS, per the doc's own rule — "matched verbatim to whatever
# the final r2a/r2b runs use". If the arms are in fact pstar_sdice, submit with
#   LOSS_ARM=pstar_sdice PSTAR=gap_ce MIX_W=<that overlay's> GAP_THETA=<its θ*> \
#   POS_WEIGHT_MIN=<λ*> POS_WEIGHT_MAX=<λ*>
# and change the defaults here — but change them for ALL EIGHT cells, before
# the first one runs.
# SEARCH_THETAS/SEARCH_MIX_W stay false: the R-series rule is that the loss is
# never re-searched per arm, and re-searching it here would confound lr_sr with
# a different loss per cell.
LOSS_ARM="${LOSS_ARM:-gap_ce}"
export PSTAR="${PSTAR:-gap_ce}"
export GAP_R="${GAP_R:-4}"
export GAP_K="${GAP_K:-60.0}"
export TL_ELL="${TL_ELL:-5}"
export TL_THETA="${TL_THETA:-0.375}"
export GAP_THETA="${GAP_THETA:-0.55836}"
export MIX_W="${MIX_W:-0.6075946831862098}"
export TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
export CL_ALPHA="${CL_ALPHA:-0.3}"
export CL_ITERS="${CL_ITERS:-5}"
export SKEL_W="${SKEL_W:-1.0}"
export SKEL_RADIUS="${SKEL_RADIUS:-1}"
export WARMUP_START="${WARMUP_START:-30}"
export WARMUP_RAMP="${WARMUP_RAMP:-10}"
export SEARCH_THETAS="${SEARCH_THETAS:-false}"
export SEARCH_MIX_W="${SEARCH_MIX_W:-false}"
# λ* pinned as a constant (min == max), same pattern as lr_sr.
export POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-4.77222}"
export POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-4.77222}"

# --- protocol: the final one, unchanged --------------------------------------
export TRAIN_SPLITS="${TRAIN_SPLITS:-train val}" # merge_val=1, no _holdout tag
export MONITOR="${MONITOR:-val_ap}"
export ADAPTIVE_NORM="${ADAPTIVE_NORM:-1}"
export ADAPTIVE_NORM_M="${ADAPTIVE_NORM_M:-0.01}"
export NORM_RECALIBRATE="${NORM_RECALIBRATE:-post}"
export BENCH_SPLIT="${BENCH_SPLIT:-test}"

# θ* on GLOBAL pooled counts, deliberately NOT the R-series refit's f1_macro.
# Macro means the mean over per-chip values, and the per-chip value of a chip
# with no predicted road is convention-dependent (NaN-on-silence vs
# 0.0-on-hallucination) — so the macro denominator moves with how badly a cell
# collapsed, which is precisely the axis this grid varies. Micro is immune.
# Constant across all 8 cells either way; all four criteria land in sweep.json,
# so re-argmaxing later costs no inference.
export SWEEP_CRITERION="${SWEEP_CRITERION:-iou}"

# AP leads the grid figure, so the bench row must carry it.
export AP_BINS="${AP_BINS:-101}"
export TILE_METRICS="${TILE_METRICS:-apls,cldice}"

# --- wandb: one project for the whole grid -----------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-sr_s2rosa_lrsr_grid}"
export WANDB_NAME="${WANDB_NAME:-${EXP_TAG}_seed${SEED:-0}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-r4grid_${HC}}"

echo "=== SR4RS lr_sr GRID CELL: HC=${HC} (sr_hc=${SR_HC}, sr_pad=${SR_PAD})  lr_sr=${LRSR} (pinned) ==="
echo "    exp_tag=${EXP_TAG}  rails=[${STD_BAND_RAISE_LO}x, ${STD_BAND_RAISE_HI}x]  lr=2e-4 pinned  descriptive-only"

source "$REPO_DIR/scripts/hpc/sr/_stages_tv.sh"
