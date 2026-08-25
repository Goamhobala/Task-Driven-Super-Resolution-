#!/bin/bash
# Resume guards shared by every per-arm refit script here. Sourced, not run.

# --- is this seed's FIT actually finished? -----------------------------------
# `unet_s2rosa_jointsr_final.ckpt` is written EVERY EPOCH, so its existence
# means "a fit started", not "a fit finished". A chain that dies at the wall
# clock leaves a perfectly readable checkpoint from whatever epoch it reached,
# and the next submit skips the fit and benches THAT.
#
# This is not hypothetical: sr_r2b_new_nohc_pstar_sdice_anorm_recalpost_seed42
# shipped a "final" checkpoint from EPOCH 14 of a 100-epoch refit, and its
# test row + θ* sweep were both computed on it. The completed 100-epoch weights
# sat beside it as *-v1.ckpt, unused.
#
# So the guard reads the epoch out of the checkpoint and requires it to have
# reached REFIT_EPOCHS. sweep.json must exist too — the bench needs the
# operating point, and a run interrupted between the two would otherwise never
# get one.
fit_complete () {  # run_dir want_epochs -> 0 if finished
  local run_dir="$1" want="$2"
  [ -f "$run_dir/sweep.json" ] || return 1
  python - "$run_dir/checkpoints/unet_s2rosa_jointsr_final.ckpt" "$want" <<'PYEOF' 2>/dev/null
import sys
from pathlib import Path
ck, want = Path(sys.argv[1]), int(sys.argv[2])
if not ck.is_file():
    sys.exit(1)
import torch
try:
    epoch = int(torch.load(ck, map_location="cpu", weights_only=False,
                           mmap=True).get("epoch", -1))
except Exception:
    sys.exit(1)                      # unreadable/truncated -> refit it
# Lightning writes `epoch` as the count of completed epochs at save time, which
# lands on REFIT_EPOCHS for a run that used its whole budget.
print(f"    final ckpt epoch={epoch} (want {want})", file=sys.stderr)
sys.exit(0 if epoch >= want else 1)
PYEOF
}

# --- already benched? --------------------------------------------------------
# The store is append-only with uuid run_ids and the bench stage has no
# duplicate check, so re-benching a (model, seed, split) adds a SECOND shard and
# every mean silently averages those chips twice.
in_store () {  # store model_name seed split -> 0 if already present
  python - "$1" "$2" "$3" "$4" <<'PYEOF' 2>/dev/null
import sys
from pathlib import Path
store, name, seed, split = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
try:
    from benchmarking.store import load_runs
    runs = load_runs(Path(store))
except Exception:
    sys.exit(1)
if runs is None or getattr(runs, "empty", True) or "model_name" not in runs.columns:
    sys.exit(1)
hit = runs[(runs["model_name"] == name) & (runs["seed"] == seed)]
if "dataset_split" in runs.columns:
    hit = hit[hit["dataset_split"] == split]
sys.exit(0 if len(hit) else 1)
PYEOF
}

# --- the per-seed chain ------------------------------------------------------
# Every arm script's body is identical once its treatment and overlay are set,
# so it lives here: plant the overlay, fit, bench test, tidy.
#
# Requires, from the caller: EXP_TAG LOSS_ARM UPSAMPLER SR_PAD SR_HC
# FREEZE_SR SR_SNAPSHOT_EVERY MODEL_NAME RUN_TAG BEST_PARAMS SEEDS
run_seeds () {
  local user_name="${USER:-$(whoami)}"
  local runs_root="${RUNS_ROOT:-/scratch/${user_name}/InstaRoad/runs}"
  local store_dir="${STORE_DIR:-/scratch/${user_name}/InstaRoad/benchmarks}"
  local epochs="${REFIT_EPOCHS:-100}"
  local seed run_dir

  for seed in $SEEDS; do
    run_dir="${runs_root}/${RUN_TAG}_seed${seed}"
    mkdir -p "$run_dir"
    # STAGE=fit refuses to start without this file, and RUN_DIR carries the
    # SEED — a seed-N dir has never been tuned and never will be. So the
    # overlay is PLANTED rather than looked up: the cluster needs nothing from
    # runslightning/, and the exact config a refit used is readable in the file
    # that ran it.
    printf '%s\n' "$BEST_PARAMS" > "$run_dir/best_params.yaml"

    if [ "${FORCE_FIT:-0}" != "1" ] && fit_complete "$run_dir" "$epochs"; then
      echo "########## ${RUN_TAG}  SEED=${seed}  FIT already complete — skipping ##########"
    else
      echo "########## ${RUN_TAG}  SEED=${seed}  FIT (${epochs} epochs, train+val) ##########"
      env EXP_TAG="$EXP_TAG" LOSS_ARM="$LOSS_ARM" SEED="$seed" STAGE=fit \
          UPSAMPLER="$UPSAMPLER" FREEZE_SR="$FREEZE_SR" SR_PAD="$SR_PAD" \
          SR_HC="$SR_HC" SR_SNAPSHOT_EVERY="$SR_SNAPSHOT_EVERY" \
          REFIT_EPOCHS="$epochs" \
          bash "$REPO_DIR/scripts/hpc/sr/refit/_refit_arm.sh"
    fi

    # test only — val was folded into training, so there is no val row to write.
    if in_store "$store_dir" "$MODEL_NAME" "$seed" test; then
      echo "########## ${RUN_TAG}  SEED=${seed}  BENCH test already in store — skipping ##########"
    else
      echo "########## ${RUN_TAG}  SEED=${seed}  BENCH test (at the refit's θ*) ##########"
      env EXP_TAG="$EXP_TAG" LOSS_ARM="$LOSS_ARM" SEED="$seed" STAGE=bench \
          UPSAMPLER="$UPSAMPLER" FREEZE_SR="$FREEZE_SR" SR_PAD="$SR_PAD" \
          SR_HC="$SR_HC" SR_SNAPSHOT_EVERY="$SR_SNAPSHOT_EVERY" \
          BENCH_SPLIT=test STORE_DIR="$store_dir" MODEL_NAME="$MODEL_NAME" \
          bash "$REPO_DIR/scripts/hpc/sr/refit/_refit_arm.sh"
    fi

    # last.ckpt holds the same weights as the final ckpt once the fit completed.
    [ "${KEEP_LAST:-0}" = "1" ] || rm -f "${run_dir}/checkpoints/last.ckpt"
  done

  echo "=== ${RUN_TAG}: seeds ${SEEDS} done ==="
}
