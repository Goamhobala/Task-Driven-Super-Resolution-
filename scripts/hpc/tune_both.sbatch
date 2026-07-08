#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="unet-tune-both"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out
#SBATCH --mem-per-cpu=8G

# Overnight batch job: run the Optuna search TWICE on the same 2-GPU allocation --
# first with ImageNet-pretrained encoders, then with random init -- so you can
# compare the two in the morning. Each search fans out one tuner per GPU.
#
#   sbatch scripts/tune_both.sbatch
#
# Because this runs under `sbatch` it is fully detached: a broken SSH pipe or
# logout can't kill it. Each search writes its own runs/unet_optuna_<tag>_seed0/
# (best_params.yaml, study.db, tune_*.log). Refit each winner afterwards with
# fit_only.sh (set STUDY_TAG to match).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
N_TRIALS="${N_TRIALS:-100}"        # total per search, split across the 2 GPUs

echo "### SEARCH 1/2: ImageNet-pretrained ###"
ENCODER_WEIGHTS=imagenet STUDY_TAG=imagenet SEARCH_GPUS=2 N_TRIALS="$N_TRIALS" \
  bash "$REPO_DIR/scripts/hpc/tune_only.sh"

echo "### SEARCH 2/2: random init ###"
ENCODER_WEIGHTS=none STUDY_TAG=random SEARCH_GPUS=2 N_TRIALS="$N_TRIALS" \
  bash "$REPO_DIR/scripts/hpc/tune_only.sh"

echo "### BOTH SEARCHES DONE ###"
