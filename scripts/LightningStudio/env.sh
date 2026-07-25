#!/bin/bash
# =============================================================================
# Central configuration for running the InstaRoad experiments on Lightning
# Studio (lightning.ai). This is the Lightning analogue of the cluster's
# SLURM headers + /scratch layout, collapsed into one file.
#
# It is sourced by:
#   * the dispatchers  (run.sh / run_both.sh / run_pair.sh / job.sh)
#   * every experiment + engine script (unet/*.sh, sr/*.sh, loss/*.sh)
# so it is the SINGLE place to change paths, the venv, and GPU defaults.
#
# Everything below is overridable: export the variable in your shell, or pass
# it as KEY=VALUE to run.sh, and the ${VAR:-default} guards here respect it.
# =============================================================================

# Idempotent — safe to source many times within one process.
if [ -n "${_INSTAROAD_ENV_LOADED:-}" ]; then return 0 2>/dev/null || true; fi
_INSTAROAD_ENV_LOADED=1

# --- Locations ---------------------------------------------------------------
# This file lives at <repo>/scripts/LightningStudio/env.sh.
LS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LS_DIR
export REPO_DIR="${REPO_DIR:-$(cd "$LS_DIR/../.." && pwd)}"

# INSTAROAD_ROOT is the base dir holding the datasets, SR model weights, per-run
# outputs (runs/) and benchmark stores — the Lightning stand-in for the
# cluster's /scratch/$USER/InstaRoad.
#
# There is no /scratch on Lightning, so by default these sit RIGHT NEXT TO the
# repo: INSTAROAD_ROOT = the repo's parent directory. Clone the repo and drop
# the data folders beside it, and every path resolves:
#
#   <parent>/                    (= INSTAROAD_ROOT, the repo's parent)
#   ├── InstaRoadPrototype/      (the repo, = REPO_DIR)
#   ├── ROSA_all/  ROSA_Dense_CDNGI/  ...            (datasets)
#   ├── runs/                                        (per-run outputs)
#   └── benchmarks/  benchmarks_loss/  models/       (stores + SR weights)
#
# Override if your data lives elsewhere (e.g. an attached Lightning Drive):
#   export INSTAROAD_ROOT=/teamspace/studios/this_studio
#   export INSTAROAD_ROOT=/teamspace/datasets/InstaRoad
export INSTAROAD_ROOT="${INSTAROAD_ROOT:-$(dirname "$REPO_DIR")}"

# uv-managed virtualenv (created by setup.sh). Repo-local by default so a plain
# `uv sync` / `uv pip install` populates it and the engines' `source
# $VENV_DIR/bin/activate` just works. Don't move this unless you know why.
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

# --- Compute defaults --------------------------------------------------------
# The Optuna search runs ONE independent worker per GPU. Free tier = one GPU,
# so keep these at 1. On a multi-GPU Studio, bump SEARCH_GPUS to the GPU count
# to fan the search out (the sr/unet engines cap it to the visible GPUs anyway).
export SEARCH_GPUS="${SEARCH_GPUS:-1}"
export REFIT_GPUS="${REFIT_GPUS:-1}"

# GDAL/rasterio segfault when forked, so the data loaders stay single-process.
export NUM_WORKERS="${NUM_WORKERS:-0}"

# Mixed precision. bf16 matches the cluster (L40S) and is fine on L4 / A10G /
# A100. NOTE: the free-tier T4 (Turing) has NO native bf16 — if your Studio is
# on a T4 and a run errors or crawls, switch to fp16:
#   export PRECISION=16-mixed
export PRECISION="${PRECISION:-bf16-mixed}"

# Let the CUDA allocator reclaim freed blocks during the long Optuna loops
# (avoids "reserved but unallocated" fragmentation across trials).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Weights & Biases. Set WANDB_MODE=offline (or `wandb login`) before a run;
# offline keeps everything local under each run dir.
export WANDB_MODE="${WANDB_MODE:-online}"
