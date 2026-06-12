#!/bin/bash
# Prepares a Kaggle environment for Dlinknet fine-tuning on the Sentinel-2 Roads Dataset.
#
# Steps:
#   1. Install Python dependencies (terra + unet extras)
#   2. Download the Sentinel-2 Roads Dataset via KaggleHub
#   3. Download the TerraMind-1.0-base backbone checkpoint from HuggingFace
#   4. Fetch the D-LinkNet source code topology

set -e

cd /kaggle/working

echo "Installing python dependencies..."
pip install uv
# terra  — terratorch, huggingface-hub, wandb, smp, albumentations, etc.
# unet   — needed because terramind.dataset re-exports from unet.dataset
uv pip install --system -e "InstaRoadPrototype[terra,unet]"

echo "Preparing Kaggle dataset"
uv run /kaggle/working/InstaRoadPrototype/src/dlinknet/prep/kaggle_dependencies.py

echo "Fetching D-LinkNet architecture source files..."
# Clone only the history depth needed to save speed/bandwidth
git clone --depth 1 https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge.git

# Move the network definitions module to your active workspace directory
mv DeepGlobe-Road-Extraction-Challenge/networks /kaggle/working/InstaRoadPrototype/src/dlinknet/

# Clean up the residual repository metadata files cleanly
rm -rf DeepGlobe-Road-Extraction-Challenge

echo "Environment ready! D-LinkNet source code added to working directory."
