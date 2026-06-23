# Steps:
#   1. Install Python dependencies 
#   2. Download the Sentinel-2 Roads Dataset via KaggleHub
#   3. Fetch the DCSNet source code 

!set -e

!cd /kaggle/working

!echo "Installing python dependencies..."
!pip install uv
# terra  — terratorch, huggingface-hub, wandb, smp, albumentations, etc.
# unet   — needed because terramind.dataset re-exports from unet.dataset
!uv pip install --system -e "InstaRoadPrototype[terra,unet, dlinknet]"

!echo "Preparing Kaggle dataset"
!uv run /kaggle/working/InstaRoadPrototype/src/dlinknet/prep/kaggle_dependencies.py

!echo "Fetching DSCNet architecture source files..."
# Clone only the history depth needed to save speed/bandwidth
!git clone --depth 1 https://github.com/YaoleiQi/DSCNet.git

# Move the network definitions module to your active workspace directory
!mv DCSNet/DSCNet_2D_opensource/Code/DRIVE /kaggle/working/InstaRoadPrototype/src/finetuning_tm/dscnet
!touch /kaggle/working/InstaRoadPrototype/src/finetuning_tm/dscnet/__init__.py

# Clean up the residual repository metadata files cleanly
!rm -rf DCSNet

!echo "Environment ready! DCSNet source code added to working directory."