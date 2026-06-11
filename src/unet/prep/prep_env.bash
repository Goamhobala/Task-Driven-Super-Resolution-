#!/bin/bash
# Prepares a Kaggle environment for the UNet + S2-ROSA dataset.
set -e

cd /kaggle/working

echo "Installing python dependencies..."
pip install uv
uv pip install --system -e "InstaRoadPrototype[unet]"

# Download the S2-ROSA dataset and symlink it into the repo.
echo "Preparing kaggle dataset..."
uv run /kaggle/working/InstaRoadPrototype/src/unet/prep/kaggle_dependencies.py
