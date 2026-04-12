#!/bin/bash
# Prepares a Kaggle environment for TerraMind fine-tuning on the Sentinel-2 Roads Dataset.
#
# Steps:
#   1. Install Python dependencies (terra + unet extras)
#   2. Download the Sentinel-2 Roads Dataset via KaggleHub
#   3. Download the TerraMind-1.0-base backbone checkpoint from HuggingFace

set -e

cd /kaggle/working

echo "Installing python dependencies..."
pip install uv
# terra  — terratorch, huggingface-hub, wandb, smp, albumentations, etc.
# unet   — needed because terramind.dataset re-exports from unet.dataset
uv pip install --system -e "InstaRoadPrototype[terra,unet]"

echo "Preparing Kaggle dataset and TerraMind checkpoint..."
uv run /kaggle/working/InstaRoadPrototype/src/terramind/prep/kaggle_dependencies.py
