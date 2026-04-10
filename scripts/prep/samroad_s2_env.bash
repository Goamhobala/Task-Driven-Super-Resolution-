#!/bin/bash
# Prepares a Kaggle Environment for SAM Road Prototype with S2 Dataset

cd /kaggle/working

# Install dependencies
# git config --global url."https://github.com/".insteadOf git@github.com:
# git clone --recursive "https://github.com/htcr/sam_road.git"

echo "Installing python dependencies..."
pip install uv
uv pip install --system -e "InstaRoad[samroad,sentinel2]"

# Prepare dependencies and dataset
echo "Preparing kaggle dependencies..."
uv run /kaggle/working/InstaRoad/src/prototype/sentinel2_prototype/kaggle_dependencies.py

# Generate subset of Sentinel2 dataset
# echo "Generating subset of Sentinel2 dataset..."
# uv run /kaggle/working/InstaRoad/src/prototype/sentinel2_prototype/generate_kaggle_s2_subset.py

# Preprocess subset of dataset into graph format
# echo "Preprocessing Sentinel2 dataset into graph format..."
# mkdir -p /kaggle/working/InstaRoad/sam_road
# uv run /kaggle/working/InstaRoad/src/prototype/sentinel2_prototype/preprocess_s2_dataset.py