#!/bin/bash
# Prepares a Kaggle Environment for Unet++ with S2 Dataset

cd /kaggle/working

echo "Installing python dependencies..."
pip install uv
uv pip install --system -e "InstaRoadPrototype[unet]"

# Prepare dependencies and dataset
echo "Preparing kaggle dependencies..."
uv run /kaggle/working/InstaRoadPrototype/src/unet/prep/kaggle_dependencies.py

