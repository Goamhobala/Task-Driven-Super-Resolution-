#!/usr/bin/env bash
# Pull the Modal `instaroad-runs` volume down to the local runs folder, WITHOUT
# touching anything already there (your manual Lightning / Kaggle downloads).
#
#   bash scripts/local/fetch_modal_runs.sh
#   DEST=/some/other/dir bash scripts/local/fetch_modal_runs.sh
#
# It downloads into a staging dir first, then copies across ONLY the arm dirs
# that do not already exist locally. Collisions are listed and left in staging
# for you to resolve by hand — this script never overwrites and never deletes.
set -euo pipefail

VOLUME="${VOLUME:-instaroad-runs}"
DEST="${DEST:-/Volumes/MAC_KIOXIA/Data/runslightning}"
STAGE="${STAGE:-${DEST}/_modal_staging}"
STORE_DEST="${STORE_DEST:-${DEST}/../benchmarks_loss_pilot_modal}"

command -v modal >/dev/null || { echo "modal CLI not on PATH" >&2; exit 1; }
mkdir -p "$DEST" "$STAGE"

echo "== downloading ${VOLUME}:/runs -> ${STAGE} =="
modal volume get "$VOLUME" runs "$STAGE" --force

echo
echo "== downloading ${VOLUME}:/benchmarks_loss_pilot -> ${STORE_DEST} =="
# The Modal-side bench shards. Kept SEPARATE from any other store on purpose —
# see the note about append-only uuid shards at the bottom.
modal volume get "$VOLUME" benchmarks_loss_pilot "$STORE_DEST" --force || \
  echo "(no benchmark store on the volume yet — fine if nothing was benched there)"

echo
echo "== merging arm dirs into ${DEST} (no overwrites) =="
collisions=0
copied=0
shopt -s nullglob
for d in "$STAGE"/*/; do
  arm="$(basename "$d")"
  if [ -e "${DEST}/${arm}" ]; then
    echo "  COLLISION  ${arm}  (already local — left in staging, nothing changed)"
    collisions=$((collisions + 1))
  else
    cp -R "$d" "${DEST}/${arm}"
    echo "  copied     ${arm}"
    copied=$((copied + 1))
  fi
done

echo
echo "copied ${copied} arm dir(s), ${collisions} collision(s)."
echo "staging kept at: ${STAGE}  ($(du -sh "$STAGE" 2>/dev/null | cut -f1)) — remove it yourself once happy."
echo
echo "Local arm dirs now present:"
for d in "$DEST"/sr_*_holdout_seed*/; do
  [ -d "$d" ] || continue
  ck="${d}checkpoints/unet_s2rosa_jointsr_final.ckpt"
  printf '  %-46s %s\n' "$(basename "$d")" \
    "$([ -f "$ck" ] && echo 'final.ckpt ok' || echo 'NO FINAL CKPT')"
done

cat <<'EOF'

NOTE on merging benchmark stores — read before combining them.
  runner.evaluate() assigns run_id = uuid4(), and store._write_shard REFUSES to
  overwrite (the store is append-only). So shards from two platforms NEVER
  collide by filename: if you copy an old Lightning shard and a fresh Modal
  shard for the SAME arm into one store, load_joined() returns BOTH and every
  per-model mean is computed over duplicated chips.
  Keep re-benched arms in a fresh store dir, or delete the superseded shard
  yourself. `benchmarking.cli eval-dir --skip-existing` dedups on
  (model_name, seed) but only within a single store.
EOF
