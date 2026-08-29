#!/usr/bin/env python3
"""Regenerate the Lightning Studio SR engine from its HPC twin.

`scripts/LightningStudio/sr/_stages_tv.sh` is not maintained by hand. It is the
HPC engine plus four mechanical substitutions — environment block, /scratch
paths, loader-worker derivation, and two diagnostics that name cluster-only
things. Hand-porting features into it is how it fell ~29 kB behind the twin
(missing SR_HC, HEAD/HEAD_TAG and the std-band rails as of 2026-08-29), and a
Lightning arm that quietly means something different from its cluster twin is
worse than one that will not run at all.

    python scripts/LightningStudio/sync_engine.py          # regenerate
    python scripts/LightningStudio/sync_engine.py --check   # CI-style: differ?

Add features to the HPC engine, then run this. If a substitution's anchor stops
matching, this script FAILS rather than emitting a half-ported engine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts/hpc/sr/_stages_tv.sh"
DST = ROOT / "scripts/LightningStudio/sr/_stages_tv.sh"

HEADER_OLD = """#!/bin/bash
# Shared tune/fit/bench engine for the FINAL (train+val refit) SR series.
# NOT submitted directly — each r*_new.sh sets its config and sources this.
"""

HEADER_NEW = """#!/bin/bash
# Shared tune/fit/bench engine for the FINAL (train+val refit) SR series —
# LIGHTNING STUDIO port of scripts/hpc/sr/_stages_tv.sh.
# NOT run directly — each r*_new.sh / rl/*.sh sets its config and sources this.
#
# GENERATED FILE — do not edit. Add the feature to the HPC twin, then run
#   python scripts/LightningStudio/sync_engine.py
# The port is four mechanical substitutions and nothing else, so `diff` against
# the twin should show only:
#   1. this header;
#   2. the environment block (env.sh instead of USER_NAME/VENV_DIR — Lightning
#      has no /scratch and no module system);
#   3. /scratch/$USER/InstaRoad -> $INSTAROAD_ROOT throughout;
#   4. the loader-worker derivation (nproc instead of SLURM_CPUS_PER_TASK) and
#      the two diagnostics that name cluster-only things.
# Anything else in that diff is drift, and drift here silently changes what a
# Lightning arm means relative to its cluster twin.
"""

ENV_OLD = """USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/InstaRoad/.venv}"
"""

ENV_NEW = """# Lightning Studio config (paths, venv, GPU defaults) — single source of truth.
# It defines REPO_DIR, INSTAROAD_ROOT, VENV_DIR, SEARCH_GPUS/REFIT_GPUS,
# PRECISION and NUM_WORKERS, each behind a ${VAR:-default} guard, so anything an
# arm script or a submit-time KEY=VALUE already set survives.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
"""

WORKERS_OLD = """if [ -z "${NUM_WORKERS}" ]; then
  JOB_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-4}}"
  if [ "${STAGE}" = "tune" ]; then
    NUM_WORKERS=$((JOB_CPUS / SEARCH_GPUS))
  else
    NUM_WORKERS=$((JOB_CPUS - 1))
  fi
  [ "${NUM_WORKERS}" -lt 1 ] && NUM_WORKERS=1
fi
echo "loader: num_workers=${NUM_WORKERS} (job_cpus=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-4}}, search_gpus=${SEARCH_GPUS})"
"""

WORKERS_NEW = """# There is no SLURM allocation to ask, so the budget is the machine: `nproc` on
# a Lightning job box IS what that job gets (an L4 studio job is 8 vCPU).
# NB env.sh exports NUM_WORKERS=0 by default — a DDP-era GDAL fork guard. With
# the pre-rasterised mask COGs the loaders open rasters lazily inside
# __getitem__ and are fork-safe, so set NUM_WORKERS= (empty) to reach this
# derivation, or pin a number at submit time. The rl campaign pins 4.
JOB_CPUS="$( (command -v nproc >/dev/null 2>&1 && nproc) \\
  || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
if [ -z "${NUM_WORKERS}" ]; then
  if [ "${STAGE}" = "tune" ]; then
    NUM_WORKERS=$((JOB_CPUS / SEARCH_GPUS))
  else
    NUM_WORKERS=$((JOB_CPUS - 1))
  fi
  [ "${NUM_WORKERS}" -lt 1 ] && NUM_WORKERS=1
fi
echo "loader: num_workers=${NUM_WORKERS} (job_cpus=${JOB_CPUS}, search_gpus=${SEARCH_GPUS})"
"""

MOUNT_OLD = '''  echo "ERROR: ${DATASET_DIR} not visible on $(hostname). Is /scratch mounted?" >&2
  echo "  (LABELS=${LABELS}. Upload the final dataset, or override DATASET_DIR.)" >&2'''

MOUNT_NEW = '''  echo "ERROR: ${DATASET_DIR} not visible on $(hostname)." >&2
  echo "  (LABELS=${LABELS}. Is INSTAROAD_ROOT=${INSTAROAD_ROOT} right and the" >&2
  echo "  dataset present? A Lightning batch job inherits the STUDIO's files, so" >&2
  echo "  upload ROSA_New to the studio once — do not stage it per job.)" >&2'''

SUBMIT_OLD = ('echo "Bench: bash scripts/hpc/submit.sh sr/${EXP_TAG}.sh '
              'STAGE=bench SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}"')
SUBMIT_NEW = ('echo "Bench: bash scripts/LightningStudio/run.sh sr/${EXP_TAG}.sh '
              'STAGE=bench SEED=${SEED}${LOSS_ARM:+ LOSS_ARM=${LOSS_ARM}}"')

# (anchor, replacement, expected occurrences)
RULES = [
    (HEADER_OLD, HEADER_NEW, 1),
    (ENV_OLD, ENV_NEW, 1),
    ("/scratch/${USER_NAME}/InstaRoad", "${INSTAROAD_ROOT}", 8),
    (WORKERS_OLD, WORKERS_NEW, 1),
    (MOUNT_OLD, MOUNT_NEW, 1),
    (SUBMIT_OLD, SUBMIT_NEW, 1),
]


def port(text: str) -> str:
    for old, new, want in RULES:
        got = text.count(old)
        if got != want:
            head = old.splitlines()[0][:70]
            raise SystemExit(
                f"sync_engine: anchor {head!r} matched {got}x, expected {want}x.\n"
                "  The HPC engine moved under this port. Fix the rule above "
                "rather than editing the generated file."
            )
        text = text.replace(old, new)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the generated file is stale; write nothing")
    args = ap.parse_args()

    want = port(SRC.read_text())
    have = DST.read_text() if DST.exists() else None
    if want == have:
        print(f"sync_engine: {DST.relative_to(ROOT)} is in sync with the HPC twin.")
        return 0
    if args.check:
        print(f"sync_engine: {DST.relative_to(ROOT)} is STALE — "
              "run `python scripts/LightningStudio/sync_engine.py`.", file=sys.stderr)
        return 1
    DST.write_text(want)
    print(f"sync_engine: regenerated {DST.relative_to(ROOT)} "
          f"({len(want)} bytes) from {SRC.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
