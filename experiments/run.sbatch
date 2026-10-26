#!/bin/bash
#SBATCH --job-name=curvature
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-21%2
#SBATCH --output=slurm-%A_%a.out
# from the repo root after prepare --models, add --partition/--account for the cluster
# python -m graphcurv.run --list gives the index of each pair
set -euo pipefail
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python -m graphcurv.run --cell-index "${SLURM_ARRAY_TASK_ID}"
