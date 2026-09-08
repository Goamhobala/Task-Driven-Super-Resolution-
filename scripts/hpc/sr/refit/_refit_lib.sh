#!/bin/bash
# Resume guards shared by every per-arm refit script here. Sourced, not run.

# --- how far did this seed's FIT actually get? -------------------------------
# Prints one of:
#   done        weights AND sweep.json present -> skip the stage entirely
#   sweep_only  training finished but sweep.json is missing -> RESUME, which
#               reaches the sweep WITHOUT retraining (see below)
#   resume      a partial checkpoint exists -> continue from it
#   fresh       nothing usable -> train from scratch
#
# WHY `unet_s2rosa_jointsr_final.ckpt` IS NOT A DONE-MARKER
# ---------------------------------------------------------
# It is rewritten EVERY EPOCH, so its existence means "a fit started". A job
# killed at the wall clock leaves a perfectly readable checkpoint from whatever
# epoch it reached. sr_r2b_new_nohc_pstar_sdice_..._seed42 shipped a "final"
# checkpoint from EPOCH 14 of a 100-epoch refit, and its test row and theta*
# sweep were both computed on it. Hence the epoch check.
#
# WHY `sweep_only` MATTERS MORE THAN IT LOOKS
# -------------------------------------------
# STAGE=fit is refit -> test -> theta* sweep as ONE unit under `set -e`. If the
# sweep dies (an unstaged benchmarking change, a bad --criterion, an OOM in the
# scorer) the 100 epochs are already on disk but sweep.json is not — and a guard
# that simply demands sweep.json would throw those epochs away and retrain.
# RESUME_FIT=1 hands Lightning `--ckpt_path last.ckpt`; with the checkpoint
# already at max_epochs there is nothing left to train, so it returns at once
# and the stage walks on to the test + sweep it died at. Minutes, not hours.
ckpt_epoch () {  # ckpt -> completed-epoch count, or -1 if missing/unreadable
  python - "$1" <<'PYEOF' 2>/dev/null || echo -1
import sys
from pathlib import Path
ck = Path(sys.argv[1])
if not ck.is_file():
    print(-1); raise SystemExit
import torch
try:
    print(int(torch.load(ck, map_location="cpu", weights_only=False,
                         mmap=True).get("epoch", -1)))
except Exception:
    print(-1)                        # truncated / unreadable -> refit it
PYEOF
}

fit_state () {  # run_dir want_epochs -> prints the state
  local run_dir="$1" want="$2" epoch has_last=0
  # Resuming needs last.ckpt specifically: it is the only checkpoint carrying
  # optimizer/scheduler state, and it is what RESUME_FIT hands --ckpt_path.
  [ -f "$run_dir/checkpoints/last.ckpt" ] && has_last=1
  epoch=$(ckpt_epoch "$run_dir/checkpoints/unet_s2rosa_jointsr_final.ckpt")
  if [ "$epoch" -ge "$want" ] 2>/dev/null; then
    if [ -f "$run_dir/sweep.json" ]; then echo done
    elif [ "$has_last" = "1" ]; then echo sweep_only
    else echo sweep_only_nolast; fi
    return
  fi
  if [ "$has_last" = "1" ]; then echo resume; else echo fresh; fi
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
  # benchmarks_newdata: the TEST split was relabelled in place (181 -> 174
  # tiles, 92 survivors changed) and nothing in the runs table separates the
  # two label sets, so old and new rows must never share a store.
  local store_dir="${STORE_DIR:-/scratch/${user_name}/InstaRoad/benchmarks_newdata}"
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

    state=fresh
    [ "${FORCE_FIT:-0}" = "1" ] || state=$(fit_state "$run_dir" "$epochs")
    resume=0
    case "$state" in
      done)
        echo "########## ${RUN_TAG}  SEED=${seed}  FIT complete (>=${epochs} epochs + sweep.json) — skipping ##########" ;;
      sweep_only)
        echo "########## ${RUN_TAG}  SEED=${seed}  weights at >=${epochs} epochs but NO sweep.json ##########"
        echo "##########   resuming to reach the theta* sweep — no epoch is retrained ##########"
        resume=1 ;;
      resume)
        echo "########## ${RUN_TAG}  SEED=${seed}  partial fit — RESUMING from last.ckpt ##########"
        echo "##########   (FORCE_FIT=1 to discard it and train from scratch) ##########"
        resume=1 ;;
      sweep_only_nolast)
        # Trained weights, no sweep.json, and no last.ckpt to resume from
        # (KEEP_LAST=0 removes it, but only AFTER a successful bench — so this
        # normally means it was deleted by hand). Retraining would be a waste:
        # the sweep is store-free and one inference pass, so run it directly.
        echo "########## ${RUN_TAG}  SEED=${seed}  WARNING ##########"
        echo "##########   weights are complete but sweep.json is missing AND"
        echo "##########   last.ckpt is gone, so the fit cannot be resumed."
        echo "##########   Refusing to retrain ${epochs} epochs. Produce the"
        echo "##########   sweep directly, then re-run this script:"
        echo "##########     python -m benchmarking.cli sweep --model sr \\"
        echo "##########       --checkpoint ${run_dir}/checkpoints/unet_s2rosa_jointsr_final.ckpt \\"
        echo "##########       --dataset-dir \$DATASET_DIR --split val \\"
        echo "##########       --model-name ${MODEL_NAME} --seed ${seed} \\"
        echo "##########       --criterion \${SWEEP_CRITERION:-f1_macro} \\"
        echo "##########       --mask-source raster --mask-dirname mask_new_2pt5 \\"
        echo "##########       --out ${run_dir}/sweep.json"
        echo "##########   (or FORCE_FIT=1 to retrain anyway)"
        continue ;;
      fresh)
        echo "########## ${RUN_TAG}  SEED=${seed}  FIT (${epochs} epochs, train+val) ##########" ;;
    esac
    if [ "$state" != "done" ]; then
      env EXP_TAG="$EXP_TAG" LOSS_ARM="$LOSS_ARM" SEED="$seed" STAGE=fit \
          UPSAMPLER="$UPSAMPLER" FREEZE_SR="$FREEZE_SR" SR_PAD="$SR_PAD" \
          SR_HC="$SR_HC" SR_SNAPSHOT_EVERY="$SR_SNAPSHOT_EVERY" \
          REFIT_EPOCHS="$epochs" RESUME_FIT="$resume" \
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
