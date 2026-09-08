#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --job-name=rseries-rebench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%x-%j.txt
#
# RE-BENCH THE WHOLE R SERIES ON THE NEW DATASET.
#
#   sbatch scripts/hpc/sr/rebench/rebench_rseries.sh
#
# WHY THIS EXISTS
# ---------------
# The ROSA_New TEST split was manually relabelled and replaced in place. The
# new labels are not a subset of the old ones: 7 tiles were dropped (181 ->
# 174) and 92 of the 174 survivors changed. So every existing test bench row is
# a DIFFERENT QUANTITY from anything scored now, and no amount of filtering
# reconciles them -- the whole series has to be re-scored to be internally
# comparable. Train and val were NOT touched, so nothing about the fits, the
# tunes or the theta* selection is invalidated: this is a scoring-only redo.
#
# NOTHING IN THE STORE GUARDS AGAINST MIXING THE TWO. dataset_dir,
# mask_dirname, mask_source, gt_res_m and cell_m are all IDENTICAL across the
# change (the path was reused), so `report` would happily average old-label and
# new-label rows into one mean. Hence STORE_DIR below defaults to a NEW store
# and the script refuses to write into one that already holds old rows.
#
# INPUT: the staged folder, one flat dir of <RUN_TAG>_seed<N>.{ckpt,sweep.json,
# best_params.yaml}. The NAMES ARE THE INTERFACE -- model_name, seed and
# exp_tag are all parsed back out of them, so do not rename the files.
#
#   RUNS_DIR=...        where the staged folder landed
#   STORE_DIR=...       output store (MUST be new/empty of old-label rows)
#   ARMS="r4b r3a"      only these arms (substring match on the tag)
#   RESWEEP=1           re-select theta* rather than reusing the staged
#                       sweep.json. YOU ALMOST CERTAINLY DO NOT WANT THIS --
#                       see below.
#   DRY_RUN=1           print the plan and exit
#
# THETA* IS STILL VALID -- DO NOT RE-SWEEP
# ---------------------------------------
# Only the TEST split was relabelled; train and val were left untouched. theta*
# is selected on VAL, so every staged sweep.json was selected on data that has
# not changed, and the stored theta is exactly the theta the protocol would
# pick today. Reusing it is correct, not a compromise.
#
# RESWEEP=1 therefore re-runs a val sweep that can only reproduce the number
# already sitting in the file -- pure cost, and a fresh chance to diverge. It
# is kept only for the day val itself changes. If you ever DO relabel val, the
# theta in these files goes stale silently: nothing downstream would notice.
set -euo pipefail

USER_NAME="${USER_NAME:-${USER:-yhxjin001}}"
ROOT="${ROOT:-/scratch/${USER_NAME}/InstaRoad}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
RUNS_DIR="${RUNS_DIR:-${ROOT}/runs/rseries_rebench}"
STORE_DIR="${STORE_DIR:-${ROOT}/benchmarks_newdata}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/ROSA_New}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BUFFER_PX="${BUFFER_PX:-1,2,3,4,5}"
AP_BINS="${AP_BINS:-101}"
TILE_METRICS="${TILE_METRICS:-apls,cldice}"

[ -d "$RUNS_DIR" ] || { echo "ERROR: no staged dir at $RUNS_DIR" >&2; exit 1; }
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

# A store that already holds rows scored on 181 tiles is an OLD-label store.
# Appending here would silently blend the two label sets in every mean.
if [ -d "$STORE_DIR/runs" ] && [ -n "$(ls -A "$STORE_DIR/runs" 2>/dev/null)" ]; then
  if python - "$STORE_DIR" <<'PY'
import sys, glob, pandas as pd
old = [f for f in glob.glob(sys.argv[1] + "/runs/*.parquet")
       if int(pd.read_parquet(f, columns=["n_tiles"]).n_tiles.iloc[0]) == 181]
sys.exit(0 if old else 1)
PY
  then
    echo "ERROR: $STORE_DIR already holds 181-tile (old-label) rows." >&2
    echo "  The store does not dedupe and nothing distinguishes the two label" >&2
    echo "  sets, so mixing them corrupts every cross-model mean." >&2
    echo "  Point STORE_DIR at a fresh directory." >&2
    exit 2
  fi
fi

shopt -s nullglob
CKPTS=("$RUNS_DIR"/*.ckpt)
[ ${#CKPTS[@]} -gt 0 ] || { echo "ERROR: no *.ckpt under $RUNS_DIR" >&2; exit 1; }

echo "=== R-SERIES RE-BENCH ==="
echo "  runs   : $RUNS_DIR  (${#CKPTS[@]} checkpoints)"
echo "  store  : $STORE_DIR"
echo "  dataset: $DATASET_DIR   split=$SPLIT"
echo "  metrics: tile=${TILE_METRICS} ap_bins=${AP_BINS} buffer=${BUFFER_PX}"
echo "  theta  : $([ "${RESWEEP:-0}" = 1 ] && echo 'RE-SWEPT (unnecessary: val is unchanged)' || echo 'reused from sweep.json (val unchanged -> still valid)')"
echo

# Seeded empty; every read below uses the ${arr[@]+...} guard because an
# empty array expansion under `set -u` is an error on bash < 4.4.
FAILED=(); DONE=(); SKIPPED=()
for CKPT in "${CKPTS[@]}"; do
  TAG="$(basename "$CKPT" .ckpt)"                       # sr_r1b_new_nohc_..._seed42
  SEED="${TAG##*_seed}"
  RUN_TAG="${TAG%_seed*}"                               # sr_r1b_new_nohc_...
  MODEL_NAME="${RUN_TAG}_ap"                            # the store's grouping key
  EXP_TAG="$(echo "$RUN_TAG" | sed -E 's/^sr_(r[0-9]+[ab]?_new).*/\1/')"

  if [ -n "${ARMS:-}" ]; then
    KEEP=0
    for A in $ARMS; do case "$TAG" in *"$A"*) KEEP=1;; esac; done
    [ $KEEP -eq 1 ] || continue
  fi

  SWEEP="$RUNS_DIR/${TAG}.sweep.json"
  CFG="$RUNS_DIR/${TAG}.best_params.yaml"
  [ -f "$SWEEP" ] || { echo "SKIP $TAG: no sweep.json"; SKIPPED+=("$TAG"); continue; }

  # The upsampler decides WHICH weights dir to hand the loader: sr4rs wants
  # gen_weights.safetensors (SR4RS_RGBN), sen2sr wants model.safetensor
  # (SEN2SRLite_RGBN). Pointing at the wrong one dies at the pre-flight check.
  # bicubic (r0) loads no SR network at all.
  UPS="$(python -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['model'].get('upsampler','sen2sr'))" "$CFG" 2>/dev/null || echo sen2sr)"
  case "$UPS" in
    sr4rs)   SR_ARGS=(--sen2sr-dir "${ROOT}/models/SR4RS_RGBN") ;;
    bicubic) SR_ARGS=() ;;
    *)       SR_ARGS=(--sen2sr-dir "${ROOT}/models/SEN2SRLite_RGBN") ;;
  esac

  if [ "${RESWEEP:-0}" = 1 ]; then
    THETA=""
  else
    # Two schemas exist in the wild: theta_sweep_bench writes iou_mean,
    # _stages_tv.sh writes a bare iou. They are the same quantity.
    THETA="$(python - "$SWEEP" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); g = d.get("sweep", {})
for k in ("iou_mean", "iou"):
    h = {t: v[k] for t, v in g.items() if k in v and v[k] == v[k]}
    if h:
        print(max(h, key=lambda t: h[t])); raise SystemExit
print(d.get("best_threshold", ""))
PY
)"
    [ -n "$THETA" ] || { echo "SKIP $TAG: no theta in sweep.json"; SKIPPED+=("$TAG"); continue; }
  fi

  echo "--- $TAG  (model=$MODEL_NAME seed=$SEED exp=$EXP_TAG ups=$UPS theta=${THETA:-resweep})"
  if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi

  CFG_ARGS=()
  [ -f "$CFG" ] && CFG_ARGS=(--config-yaml "$CFG")   # config_hash provenance only
  METRIC_ARGS=()
  IFS=',' read -r -a _TMS <<< "$TILE_METRICS"
  for _tm in "${_TMS[@]}"; do [ -n "$_tm" ] && METRIC_ARGS+=(--tile-metric "$_tm"); done

  if [ "${RESWEEP:-0}" = 1 ]; then
    python -m benchmarking.cli sweep \
      --dataset-dir "$DATASET_DIR" --checkpoint "$CKPT" --model sr \
      --split val --mask-source raster --mask-dirname mask_new_2pt5 \
      "${SR_ARGS[@]}" --out "$RUNS_DIR/${TAG}.sweep_new.json" \
      || { echo "FAILED sweep $TAG"; FAILED+=("$TAG"); continue; }
    THETA="$(python - "$RUNS_DIR/${TAG}.sweep_new.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); g = d.get("sweep", {})
for k in ("iou_mean", "iou"):
    h = {t: v[k] for t, v in g.items() if k in v and v[k] == v[k]}
    if h:
        print(max(h, key=lambda t: h[t])); raise SystemExit
print(d.get("best_threshold", ""))
PY
)"
  fi

  if python -m benchmarking.cli eval \
      --dataset-dir "$DATASET_DIR" \
      --checkpoint "$CKPT" \
      --model sr --model-name "$MODEL_NAME" --seed "$SEED" \
      --store-dir "$STORE_DIR" --split "$SPLIT" \
      --mask-source raster --mask-dirname mask_new_2pt5 \
      --exp-tag "$EXP_TAG" --label-source new \
      "${METRIC_ARGS[@]}" \
      ${AP_BINS:+--ap-bins "$AP_BINS"} \
      ${BUFFER_PX:+--buffer-px "$BUFFER_PX"} \
      "${SR_ARGS[@]}" \
      "${CFG_ARGS[@]}" \
      --threshold "$THETA" --batch-size "$BATCH_SIZE"
  then DONE+=("$TAG"); else echo "FAILED $TAG"; FAILED+=("$TAG"); fi
done

echo
echo "=== ${#DONE[@]} benched, ${#SKIPPED[@]} skipped, ${#FAILED[@]} failed ==="
for f in ${FAILED[@]+"${FAILED[@]}"}; do echo "  FAILED $f"; done
echo
echo "Report with:"
echo "  python -m benchmarking.cli report --store-dir $STORE_DIR \\"
echo "    --metric f1 --metric iou --metric ap --metric apls --metric cldice \\"
echo "    --aggregation both --macro-unit chip --tile-agg macro --pair-on tile"
[ ${#FAILED[@]} -eq 0 ]
