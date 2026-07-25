#!/bin/bash
# =============================================================================
# One-time environment setup for a fresh Lightning Studio. The Studio's
# filesystem persists between sessions, so you normally run this ONCE.
#
#   bash scripts/LightningStudio/setup.sh              # UNet + loss experiments
#   bash scripts/LightningStudio/setup.sh --sr         # + SEN2SR / SR4RS (joint-SR arms)
#   bash scripts/LightningStudio/setup.sh --sr --mamba # + full (Mamba) SEN2SR / r3 arms
#
# What it does: installs uv (if missing), builds the repo-local .venv, installs
# the optional-dependency groups these scripts need, and scaffolds the data
# root. Datasets and SR weights you upload yourself (see the echo at the end).
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

WITH_SR=1; WITH_MAMBA=0
for a in "$@"; do
  case "$a" in
    --sr)    WITH_SR=1 ;;
    --mamba) WITH_SR=1; WITH_MAMBA=1 ;;
    *) echo "unknown flag: $a (use --sr and/or --mamba)" >&2; exit 2 ;;
  esac
done

echo "repo       = $REPO_DIR"
echo "data root  = $INSTAROAD_ROOT"
echo "venv       = $VENV_DIR"
echo "extras     = unet, sentinel2, benchmarking$([ $WITH_SR = 1 ] && echo ', sr(sen2sr/mlstac)')$([ $WITH_MAMBA = 1 ] && echo ', mamba-ssm')"
echo

# 1. uv (the repo's package manager).
if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "uv = $(command -v uv)"

# 2. Build the venv and install the base experiment stack.
#    We use `uv pip install` (a fresh per-platform resolve) rather than
#    `uv sync --locked` because the repo's uv.lock pins macOS-only environments;
#    a fresh resolve is what works on Lightning's Linux GPU boxes.
cd "$REPO_DIR"
[ -d "$VENV_DIR" ] || uv venv "$VENV_DIR" --python 3.12

# unet        -> torch/lightning/smp/albumentations/optuna/jsonargparse/wandb
#                (the UNet baseline AND the loss-ablation arms)
# sentinel2   -> rasterio/geopandas/rioxarray/jenkspy (dataset + norm-stats)
# benchmarking-> networkx (APLS), typer, scipy (the benchmark store CLI)
echo "=== installing base extras (unet, sentinel2, benchmarking) ==="
uv pip install --python "$VENV_DIR" -e ".[unet,sentinel2,benchmarking]"

# 3. Joint-SR arms (r*): SEN2SR-Lite + SR4RS need these; installed additively so
#    they don't drag in the CUDA-compiled mamba-ssm unless you ask for it.
if [ "$WITH_SR" = 1 ]; then
  echo "=== installing SR deps (sen2sr, mlstac, safetensors) ==="
  uv pip install --python "$VENV_DIR" sen2sr mlstac safetensors
fi

# 4. Full (Mamba-backbone) SEN2SR — only the sen2sr_full / r3 arms need it.
#    Compiles CUDA kernels from source, so it needs the Studio's CUDA toolkit
#    (nvcc) and a few minutes. Skip unless you're running r3.
if [ "$WITH_MAMBA" = 1 ]; then
  echo "=== installing mamba-ssm (compiles CUDA kernels; needs nvcc) ==="
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARN: nvcc not found — mamba-ssm build will likely fail. Start a GPU" >&2
    echo "      machine with the CUDA toolkit, or skip r3/sen2sr_full arms." >&2
  fi
  uv pip install --python "$VENV_DIR" --no-build-isolation mamba-ssm causal-conv1d
fi

# 5. Scaffold the data-root layout (datasets/weights you add yourself).
mkdir -p "$INSTAROAD_ROOT"/{runs,benchmarks,benchmarks_loss,models}

cat <<EOF

=== setup done ===
venv: $VENV_DIR   (the engines activate this automatically)

Before your first run:
  1. Upload datasets under $INSTAROAD_ROOT
        ROSA_all/  ROSA_Dense_CDNGI/  (and ROSA_Dense_Overture/ if used)
  2. Upload SR weights under $INSTAROAD_ROOT/models
        SEN2SRLite_RGBN/model.safetensor        (sr r1/r2/r7 arms)
        SR4RS_RGBN/gen_weights.safetensors ...   (sr r4/r5/r6 arms)
  3. Generate norm stats if missing:
        source $VENV_DIR/bin/activate
        python -m sentinel2data.cli norm-stats --dataset-dir $INSTAROAD_ROOT/ROSA_all \\
               --out $REPO_DIR/src/unet/configs/norm_stats.yaml
  4. wandb login          # or:  export WANDB_MODE=offline

Smoke test (UNet baseline, quick Optuna search):
  bash scripts/LightningStudio/run.sh unet/cdngi.sh STAGE=tune N_TRIALS=2 TUNE_EPOCHS=1
EOF
