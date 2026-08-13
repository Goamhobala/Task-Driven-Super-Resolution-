#!/bin/bash
# θ* sweep + FINAL-protocol bench for a set of already-trained run dirs.
#
# WHY THIS EXISTS (and why `--action sweep` is not the right tool for it)
# ---------------------------------------------------------------------
# `scripts/local/theta_sweep_bench.py` was written for the LOSS PILOT, whose
# arms are holdout runs (TRAIN_SPLITS=train): it discovers dirs matching
# `sr_<exp>_<tag>_holdout_seed<N>` and benches on `val`. The runs in
# runs_sr_wbce are neither:
#
#   * they are FINAL-protocol runs (TRAIN_SPLITS='train val', merge_val=1), so
#     their dir names carry no `_holdout` tag and discovery finds nothing; and
#   * `val` is TRAINING data for them. Benching on val would be a train score.
#     `_stages_tv.sh` refuses that combination outright (BENCH_SPLIT=val guard).
#
# So this script does exactly what `_stages_tv.sh STAGE=bench` does for a
# final-protocol arm, for each run dir in turn:
#
#   1. sweep   theta_sweep_bench.py --run-dir ... --skip-bench
#              -> sweep.json in the run dir. θ* is selected on VAL, always —
#              the same decision `_stages_tv.sh` documents at its bench stage.
#              For these runs val is in-sample, so θ* is chosen on data the
#              model saw. That is the honest option: the alternative is to
#              select θ on `test`, which is the split being reported.
#   2. bench   benchmarking.cli eval --split test --threshold θ*
#              -> ONE run row per arm, over ALL chips of the test split.
#   3. report  benchmarking.cli report over the whole store, then again with
#              --by-stratum. The stratified numbers are a SLICE of the same
#              chips (per-chip metrics, per-tile strata) — no re-inference, and
#              arithmetically identical to a per-stratum eval.
#
# The store is append-only with uuid run_ids (`store._write_shard` refuses to
# overwrite), so an arm already present is SKIPPED rather than re-benched: a
# second shard would make every per-model mean average duplicated chips.
# FORCE=1 overrides, ALLOW_DUPLICATE=1 lets the duplicate through knowingly.
#
# Not meant to be run by hand on Modal — `modal_app.py --action final-bench`
# sets the paths and runs it in the container. Locally:
#   REPO_DIR=$PWD DATASET_DIR=... RUNS_ROOT=... MODELS_DIR=... \
#     bash scripts/modal/final_bench.sh
set -euo pipefail
trap 'echo "!! final_bench.sh ABORTED at line $LINENO (exit $?)" >&2' ERR

# --- Paths (defaults match modal_app.py's mount layout) ----------------------
export REPO_DIR="${REPO_DIR:-/root/InstaRoadPrototype}"
export INSTAROAD_ROOT="${INSTAROAD_ROOT:-/out}"
export DATASET_DIR="${DATASET_DIR:-/data/ROSA_New}"
export RUNS_ROOT="${RUNS_ROOT:-${INSTAROAD_ROOT}}"
export MODELS_DIR="${MODELS_DIR:-/data/models}"
export STORE_DIR="${STORE_DIR:-${INSTAROAD_ROOT}/benchmarks_final_wbce}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# --- Protocol constants (match _stages_tv.sh's bench stage) ------------------
BENCH_SPLIT="${BENCH_SPLIT:-test}"      # final protocol reports on test
SWEEP_SPLIT="val"                       # hardcoded in theta_sweep_bench.py
MASK_SOURCE="${MASK_SOURCE:-raster}"    # LABELS=new
MASK_DIRNAME="${MASK_DIRNAME:-mask_new_2pt5}"
LABEL_SOURCE="${LABEL_SOURCE:-new}"
TILE_METRICS="${TILE_METRICS:-apls,cldice}"
SELECT_ON="${SELECT_ON:-iou_mean}"      # θ* on macro per-chip IoU
# Buffered precision/recall/F1 tolerance, in pixels of the 2.5 m GT grid
# (3 px = 7.5 m). Empty = off. Setting it makes the sweep record buffered_*
# at every θ, which is what allows SELECT_ON=buffered_f1_mean; it also adds
# the columns to the benched rows so `report --metric buffered_f1` works.
BUFFER_PX="${BUFFER_PX:-}"
# θ grid. Every θ is scored off the SAME forward pass, so widening or refining
# it is essentially free — worth doing when an arm's θ* lands on an endpoint,
# because that means the true optimum is outside the range and the reported θ*
# is censored, not found. SWEEP_LO=0.01 is the usual response to a 0.05 pin.
SWEEP_LO="${SWEEP_LO:-0.05}"
SWEEP_HI="${SWEEP_HI:-0.95}"
SWEEP_STEP="${SWEEP_STEP:-0.025}"
DEVICE="${DEVICE:-cuda}"
# Buffered metrics only exist in the rows when BUFFER_PX was set at bench time.
# `benchmarking.cli report` REJECTS a metric that is in neither the chips nor
# the tiles table (_load_metric_table raises BadParameter) — deliberately, so a
# typo in --metric fails loudly rather than silently reporting less. That means
# this list cannot be passed blind: it is filtered against the store's actual
# columns below. They are not count-derivable, so _run_report aggregates them
# MACRO regardless of --aggregation (cli.py:578), like apls/cldice.
METRICS="${METRICS:-f1,iou,apls,cldice,buffered_f1,buffered_precision,buffered_recall}"
MAX_TILES="${MAX_TILES:-}"              # smoke test: cap tiles everywhere

# --- The arms ----------------------------------------------------------------
# run_dir|model_name|exp_tag|seed|sr_weights_subdir   ('-' = no SR weights)
#
# run_dir is RELATIVE TO RUNS_ROOT (which is /out on Modal), so arms from
# different download folders — runs_sr_wbce/, runs_gap_tl_ce/ — coexist in one
# table without a second orchestrator. The dir names are globally unique, but
# keeping the folder in the path documents where each arm came from.
#
# model_name reproduces `_stages_tv.sh`'s MODEL_NAME exactly:
#   sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}${PROTO_TAG}
# with PROTO_TAG empty under the merge-val protocol. It is what the store keys
# on, so it must match what a `STAGE=bench` on the training node would emit.
#
# sr_weights: the checkpoint's hparams record the TRAINING node's path
# (/scratch/...), which does not exist here, so every SR arm gets an explicit
# override. r2a is SEN2SR (UPSAMPLER=sen2sr), r4b is the SR4RS port
# (UPSAMPLER=sr4rs); r0 is bicubic and builds no SR net at all.
DEFAULT_ARMS=(
  "runs_sr_wbce/sr_r0_new_wbce_seed0|sr_r0_new_wbce|r0_new|0|-"
  "runs_sr_wbce/sr_r2a_new_wbce_seed0|sr_r2a_new_wbce|r2a_new|0|SEN2SRLite_RGBN"
  "runs_sr_wbce/sr_r2a_new_wbce_noreg_seed333|sr_r2a_new_wbce_noreg|r2a_new|333|SEN2SRLite_RGBN"
  "runs_sr_wbce/sr_r4b_new_wbce_seed0|sr_r4b_new_wbce|r4b_new|0|SR4RS_RGBN"
  "runs_gap_tl_ce/sr_r0_new_gap_tl_ce_seed0|sr_r0_new_gap_tl_ce|r0_new|0|-"
  "runs_gap_tl_ce/sr_r2a_new_gap_tl_ce_seed0|sr_r2a_new_gap_tl_ce|r2a_new|0|SEN2SRLite_RGBN"
  "runs_gap_tl_ce/sr_r2a_new_gap_tl_ce_noreg_seed333|sr_r2a_new_gap_tl_ce_noreg|r2a_new|333|SEN2SRLite_RGBN"
)
FINAL_CKPT="checkpoints/unet_s2rosa_jointsr_final.ckpt"

# ARMS="sr_r0_new_wbce_seed0 sr_r4b_new_wbce_seed0" restricts to those run dirs.
if [ -n "${ARMS:-}" ]; then
  SELECTED=()
  for want in $ARMS; do
    hit=""
    for rec in "${DEFAULT_ARMS[@]}"; do
      path="${rec%%|*}"
      # match the full relative path, its basename, or the model_name — all
      # three are unambiguous and one of them is what you will actually type
      mid="${rec#*|}"
      if [ "$path" = "$want" ] || [ "${path##*/}" = "$want" ] || [ "${mid%%|*}" = "$want" ]; then
        hit="$rec"
      fi
    done
    [ -z "$hit" ] && { echo "ERROR: unknown arm '$want' — add it to DEFAULT_ARMS" >&2; exit 2; }
    SELECTED+=("$hit")
  done
else
  SELECTED=("${DEFAULT_ARMS[@]}")
fi

PY="$(command -v python || command -v python3)"

echo "=============================================================="
echo "final_bench.sh   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  repo      : ${REPO_DIR}  @ ${INSTAROAD_GIT_SHA:-unknown}"
echo "  dataset   : ${DATASET_DIR}"
echo "  runs      : ${RUNS_ROOT}"
echo "  store     : ${STORE_DIR}"
echo "  sr weights: ${MODELS_DIR}"
echo "  protocol  : sweep θ* on '${SWEEP_SPLIT}' (select_on=${SELECT_ON}) -> bench on '${BENCH_SPLIT}'"
echo "  θ grid    : ${SWEEP_LO}..${SWEEP_HI} step ${SWEEP_STEP}"
echo "  gt        : mask_source=${MASK_SOURCE} mask_dirname=${MASK_DIRNAME} label_source=${LABEL_SOURCE}"
echo "  tile mtr  : ${TILE_METRICS}${MAX_TILES:+   MAX_TILES=${MAX_TILES} (SMOKE TEST — not a result)}"
echo "  arms      : ${#SELECTED[@]}"
echo "=============================================================="

# `val` must not be the bench split under the merge-val protocol — the same
# guard _stages_tv.sh applies, restated here because this script bypasses it.
if [ "$BENCH_SPLIT" = "val" ]; then
  echo "ERROR: BENCH_SPLIT=val, but these runs folded val into training." >&2
  echo "  That score would be a training score. Use BENCH_SPLIT=test." >&2
  exit 2
fi

# --- Store duplicate guard ---------------------------------------------------
in_store() {  # model_name seed split -> 0 if already present
  "$PY" - "$STORE_DIR" "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path
store, name, seed, split = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
try:
    from benchmarking.store import load_runs
    runs = load_runs(Path(store))
except Exception:
    sys.exit(1)                       # absent/empty store: nothing to collide
if runs is None or getattr(runs, "empty", True) or "model_name" not in runs.columns:
    sys.exit(1)
hit = runs[(runs["model_name"] == name) & (runs["seed"] == seed)]
if "dataset_split" in runs.columns:
    hit = hit[hit["dataset_split"] == split]
sys.exit(0 if len(hit) else 1)
PY
}

benched=(); skipped=(); failed=()

for rec in "${SELECTED[@]}"; do
  IFS='|' read -r dirname model_name exp_tag seed sr_sub <<< "$rec"
  RUN_DIR="${RUNS_ROOT}/${dirname}"
  CKPT="${RUN_DIR}/${FINAL_CKPT}"

  echo
  echo "--------------------------------------------------------------"
  echo "[$model_name] seed=${seed}  ${RUN_DIR}"

  if [ ! -f "$CKPT" ]; then
    echo "  SKIP — no final checkpoint at ${CKPT}" >&2
    failed+=("${model_name}: no final ckpt")
    continue
  fi

  # theta_sweep_bench.py defaults --sen2sr-dir to $SEN2SR_DIR, which _base_env
  # always exports. For a bicubic arm that would silently hand the sweep a
  # different SR config from the bench (which gets no override at all) — both
  # unused, but only because bicubic builds no SR net. Unset it so the two
  # stages provably see the same thing rather than the same thing by luck.
  unset SEN2SR_DIR
  SR_ARGS=()
  if [ "$sr_sub" != "-" ]; then
    SR_DIR="${MODELS_DIR}/${sr_sub}"
    if [ ! -d "$SR_DIR" ]; then
      echo "  SKIP — SR weights missing: ${SR_DIR}" >&2
      failed+=("${model_name}: missing ${sr_sub}")
      continue
    fi
    SR_ARGS=(--sen2sr-dir "$SR_DIR")
  fi

  CONFIG_ARGS=()
  [ -f "${RUN_DIR}/best_params.yaml" ] && CONFIG_ARGS=(--config-yaml "${RUN_DIR}/best_params.yaml")

  # --- 1. θ* sweep on val (one inference pass scores the whole grid) --------
  # A sweep.json left by a MAX_TILES smoke run holds a θ* fitted to a handful
  # of tiles, and theta_sweep_bench.py records that in `sweep_max_tiles`.
  # Reusing it in a full run would quietly bench every arm at a smoke-test
  # threshold — the numbers would look plausible and be wrong. Re-sweep
  # instead, unless this run is itself capped.
  STALE=0
  if [ -f "${RUN_DIR}/sweep.json" ] && [ -z "$MAX_TILES" ]; then
    # `|| echo 1` rather than `|| echo 0`: an unreadable or malformed
    # sweep.json must re-sweep, not silently fall through to reusing a θ* this
    # check could not verify. (It also keeps `set -e` from killing the run on a
    # corrupt file.)
    STALE=$("$PY" - "${RUN_DIR}/sweep.json" "$SELECT_ON" 2>/dev/null <<'PYEOF' || echo 1
import json, sys
s = json.load(open(sys.argv[1]))
key = sys.argv[2]
# 1. a capped (smoke) sweep is never a valid basis for a full run
if s.get("sweep_max_tiles"):
    print(1); raise SystemExit
# 2. the requested criterion must actually be present in the recorded curve.
#    Selecting on buffered_f1_mean against a sweep taken without BUFFER_PX
#    would otherwise hard-fail later; re-sweeping is the fix, and it is the
#    ONLY case that needs new inference when the criterion changes.
if not any(key in v for v in s["sweep"].values()):
    print(1); raise SystemExit
print(0)
PYEOF
)
    if [ "$STALE" = "1" ]; then
      echo "        sweep.json is stale for SELECT_ON=${SELECT_ON} (capped run, or the"
      echo "        criterion was not recorded) — re-sweeping on the full split"
    fi
  fi

  if [ ! -f "${RUN_DIR}/sweep.json" ] || [ "${REFRESH_SWEEP:-0}" = "1" ] || [ "$STALE" = "1" ]; then
    echo "  [1/2] sweep θ on '${SWEEP_SPLIT}' -> sweep.json"
    # --refresh-sweep is NOT optional when a sweep.json is present. Deciding to
    # re-sweep out here only chooses to INVOKE the script; theta_sweep_bench's
    # own _process() then finds the same file and reuses it ("reusing
    # sweep.json: θ* = ..."), so without this flag the stale-θ guard above
    # announces a re-sweep and silently delivers the stale value anyway.
    # Reaching here means we have DECIDED to re-sweep, so if a sweep.json is
    # still on disk the flag is mandatory: theta_sweep_bench's own _process()
    # short-circuits on an existing file ("reusing sweep.json: θ* = ...") and
    # would hand back the very value we just rejected.
    SWEEP_EXTRA=()
    [ -f "${RUN_DIR}/sweep.json" ] && SWEEP_EXTRA+=(--refresh-sweep)
    [ -n "$MAX_TILES" ] && SWEEP_EXTRA+=(--max-tiles "$MAX_TILES")
    "$PY" "${REPO_DIR}/scripts/local/theta_sweep_bench.py" \
      --run-dir "$RUN_DIR" \
      --model-name "$model_name" \
      --exp-tag "$exp_tag" \
      --seed "$seed" \
      --dataset-dir "$DATASET_DIR" \
      --runs-dir "$RUNS_ROOT" \
      --store-dir "${STORE_DIR}_sweepscratch" \
      --select-on "$SELECT_ON" \
      --lo "$SWEEP_LO" --hi "$SWEEP_HI" --step "$SWEEP_STEP" \
      ${BUFFER_PX:+--buffer-px "$BUFFER_PX"} \
      --device "$DEVICE" \
      "${SR_ARGS[@]}" \
      --skip-bench ${SWEEP_EXTRA[@]+"${SWEEP_EXTRA[@]}"} \
      || { echo "  FAILED at sweep" >&2; failed+=("${model_name}: sweep"); continue; }
  else
    echo "  [1/2] reusing existing sweep.json (REFRESH_SWEEP=1 to redo)"
  fi

  # Re-derive θ* for THIS run's SELECT_ON from the recorded curve, rather than
  # trusting sweep.json's `best_threshold` — that field records whichever
  # criterion wrote the file. The sweep stores every metric at every θ, so
  # switching criterion costs no inference at all; it is a re-argmax.
  THETA=$("$PY" - "${RUN_DIR}/sweep.json" "$SELECT_ON" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
key = sys.argv[2]
curve = s["sweep"]
have = {t: v[key] for t, v in curve.items() if key in v and v[key] == v[key]}
if not have:
    sys.exit(f"sweep.json has no '{key}' — re-sweep "
             f"(buffered_* needs BUFFER_PX set at sweep time). "
             f"Recorded: {sorted(next(iter(curve.values())))}")
best = max(have, key=lambda t: have[t])
print(float(best))
PYEOF
)
  EDGE=$("$PY" - "${RUN_DIR}/sweep.json" "$THETA" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
ts = sorted(float(t) for t in s["sweep"])
print("True" if float(sys.argv[2]) in (ts[0], ts[-1]) else "False")
PYEOF
)
  echo "        θ* = ${THETA}  [${SELECT_ON}]$([ "$EDGE" = "True" ] && echo '   ** ON GRID EDGE — optimum may lie outside the swept range **')"

  # --- 2. bench on test at θ*, over ALL chips ------------------------------
  # SWEEP_ONLY=1 stops here. One sweep records EVERY metric at EVERY θ, so a
  # sweep-only pass yields θ* for iou_mean, f1_mean AND buffered_f1_mean at
  # once — for ~6 min/arm instead of the ~30 min/arm a bench costs. Run it
  # first, read the θ* table, and only then pay for the benches that are
  # actually distinct: where two criteria pick the same θ the benched rows
  # would be identical work.
  if [ "${SWEEP_ONLY:-0}" = "1" ]; then
    echo "  [2/2] SKIP bench — SWEEP_ONLY=1"
    skipped+=("$model_name")
    continue
  fi

  if in_store "$model_name" "$seed" "$BENCH_SPLIT" && [ "${ALLOW_DUPLICATE:-0}" != "1" ]; then
    echo "  [2/2] SKIP bench — ${model_name} seed=${seed} split=${BENCH_SPLIT} is already"
    echo "        in this store. It is append-only with uuid run_ids, so a second"
    echo "        shard would double-count every chip. ALLOW_DUPLICATE=1 to force."
    skipped+=("$model_name")
    continue
  fi

  METRIC_ARGS=()
  if [ -n "$TILE_METRICS" ]; then
    IFS=',' read -r -a _TMS <<< "$TILE_METRICS"
    for _tm in "${_TMS[@]}"; do METRIC_ARGS+=(--tile-metric "${_tm}"); done
  fi
  MASK_ARGS=(--mask-source "$MASK_SOURCE")
  [ "$MASK_SOURCE" = "raster" ] && MASK_ARGS+=(--mask-dirname "$MASK_DIRNAME")
  [ -n "$MAX_TILES" ] && METRIC_ARGS+=(--max-tiles "$MAX_TILES")

  echo "  [2/2] bench on '${BENCH_SPLIT}' at θ*=${THETA} (${TILE_METRICS})"
  T0=$SECONDS
  "$PY" -m benchmarking.cli eval \
    --dataset-dir "$DATASET_DIR" \
    --checkpoint "$CKPT" \
    --model sr \
    --model-name "$model_name" \
    --seed "$seed" \
    --store-dir "$STORE_DIR" \
    --split "$BENCH_SPLIT" \
    --exp-tag "$exp_tag" \
    --label-source "$LABEL_SOURCE" \
    --threshold "$THETA" \
    --device "$DEVICE" \
    ${BUFFER_PX:+--buffer-px "$BUFFER_PX"} \
    "${SR_ARGS[@]}" \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    ${METRIC_ARGS[@]+"${METRIC_ARGS[@]}"} \
    "${MASK_ARGS[@]}" \
    || { echo "  FAILED at bench" >&2; failed+=("${model_name}: bench"); continue; }

  # Wall time per arm, so a --max-tiles smoke run extrapolates to a bill
  # instead of a guess. The sweep pass has no tile metrics and the bench does,
  # so the gap between them is roughly what APLS+clDice cost — which is what
  # decides whether a cheaper/slower GPU would save money or waste it.
  echo "        bench wall: $((SECONDS - T0))s${MAX_TILES:+ for ${MAX_TILES} tiles}"
  touch "${RUN_DIR}/.final_bench_done"
  benched+=("$model_name")
done

echo
echo "=============================================================="
echo "benched ${#benched[@]}  skipped ${#skipped[@]}  failed ${#failed[@]}"
[ ${#benched[@]} -gt 0 ] && printf '  benched  %s\n' "${benched[@]}"
[ ${#skipped[@]} -gt 0 ] && printf '  skipped  %s\n' "${skipped[@]}"
[ ${#failed[@]}  -gt 0 ] && printf '  FAILED   %s\n' "${failed[@]}"
echo "=============================================================="

# --- 3. reports --------------------------------------------------------------
# Whole split first, then each stratum. Both read the same store; the stratified
# pass is a slice of the same chips, not a re-run.
#
# This runs INSIDE the container (REPORT=1) rather than being chained from the
# local client. It costs a few minutes of otherwise-idle GPU — the report is
# pandas plus a 2000-sample paired bootstrap, pure CPU — and that is the price
# of the .md being GUARANTEED on the Volume. Chaining it locally was cheaper
# but died with the client, which is exactly how a disconnect cost a run on
# 2026-08-11. Re-report any time without a GPU:
#   modal run scripts/modal/modal_app.py --action report --store <dir> --by-stratum
#
# Copies land in every runs/ folder that contributed an arm, so a
# `modal volume get /runs_gap_tl_ce` brings the tables down with the runs they
# describe. Filenames carry SELECT_ON: tuning the same arms on iou_mean, then
# f1_mean, then buffered_f1_mean writes three sets side by side instead of
# silently overwriting one.
if [ "${REPORT:-1}" = "1" ] && [ "${SWEEP_ONLY:-0}" != "1" ]; then
  # Keep only the metrics this store actually carries, so one store benched
  # without BUFFER_PX and another benched with it can share a METRICS default.
  AVAIL=$("$PY" - "$STORE_DIR" "$METRICS" <<'PYEOF'
import sys
from pathlib import Path
store, wanted = sys.argv[1], [m.strip() for m in sys.argv[2].split(",") if m.strip()]
cols = set()
try:
    from benchmarking.store import load_chips, load_tiles
    cols |= set(load_chips(Path(store)).columns)
    try:
        cols |= set(load_tiles(Path(store)).columns)
    except Exception:
        pass
except Exception:
    print(",".join(wanted)); raise SystemExit   # let the CLI speak if unreadable
keep = [m for m in wanted if m in cols]
drop = [m for m in wanted if m not in cols]
if drop:
    print("SKIP " + ",".join(drop), file=sys.stderr)
print(",".join(keep))
PYEOF
)
  echo "  metrics   : ${AVAIL}"
  REPORT_ARGS=()
  IFS=',' read -r -a _MS <<< "$AVAIL"
  for _m in "${_MS[@]}"; do REPORT_ARGS+=(--metric "${_m}"); done

  ALL_MD="${STORE_DIR}/report_${SELECT_ON}_all.md"
  STRAT_MD="${STORE_DIR}/report_${SELECT_ON}_by_stratum.md"

  echo
  echo "######################################################################"
  echo "# ALL CHIPS — ${BENCH_SPLIT} split, θ* on ${SELECT_ON}"
  echo "######################################################################"
  "$PY" -m benchmarking.cli report --store-dir "$STORE_DIR" \
    "${REPORT_ARGS[@]}" --out "$ALL_MD" || true

  echo
  echo "######################################################################"
  echo "# BY STRATUM (urbanisation_classification)"
  echo "######################################################################"
  # --by-stratum fans --out into report_..._<Stratum>.md, one file per stratum.
  "$PY" -m benchmarking.cli report --store-dir "$STORE_DIR" \
    "${REPORT_ARGS[@]}" --by-stratum --out "$STRAT_MD" || true

  # Distinct runs/ folders among the arms actually selected for this run.
  RUNS_FOLDERS=$(for rec in "${SELECTED[@]}"; do
                   rel="${rec%%|*}"; dir="${rel%/*}"
                   [ "$dir" != "$rel" ] && echo "$dir"
                 done | sort -u)
  for folder in $RUNS_FOLDERS; do
    dest="${RUNS_ROOT}/${folder}"
    [ -d "$dest" ] || continue
    for md in "${STORE_DIR}"/report_"${SELECT_ON}"_*.md; do
      [ -f "$md" ] || continue
      cp "$md" "${dest}/$(basename "$md")"
      echo "  report -> ${dest}/$(basename "$md")"
    done
  done
fi

[ ${#failed[@]} -eq 0 ] || exit 1
exit 0
