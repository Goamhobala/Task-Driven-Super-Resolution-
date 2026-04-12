#!/bin/bash
# Prepares a Kaggle environment for TerraMind fine-tuning on the Sentinel-2 Roads Dataset.

set -e

cd /kaggle/working

echo "Installing python dependencies..."
pip install uv
uv pip install --system -e "InstaRoadPrototype[terra,unet]"

# Download dataset and create symlink
echo "Preparing Kaggle dataset..."
uv run /kaggle/working/InstaRoadPrototype/src/terramind/prep/kaggle_dependencies.py
