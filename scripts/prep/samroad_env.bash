#!/bin/bash
# Prepares a Kaggle Environment for SAM Road Prototype

cd /kaggle/working

# Install dependencies
git config --global url."https://github.com/".insteadOf git@github.com:
git clone --recursive "https://github.com/htcr/sam_road.git"
pip install uv
uv pip install --system -e "InstaRoad[samroad]"

# Prepare dependencies and dataset
uv run /kaggle/working/InstaRoad/src/prototype/sam_road_prototype/kaggle_dependencies.py

# Generate labels for CityScale and SpaceNet datasets
cd /kaggle/working/sam_road/cityscale/
uv run ./generate_labels.py
cd /kaggle/working/sam_road/spacenet/
uv run ./generate_labels.py