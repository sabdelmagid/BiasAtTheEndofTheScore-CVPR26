#!/bin/bash
#SBATCH --job-name=renoinv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/renowinv/%x_%A_%a.out
#SBATCH --error=logs/renowinv/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Add an account directive here if required by your cluster, for example:
# #SBATCH --account=YOUR_ACCOUNT

# Cluster environment setup. Override these variables at submission time if needed.
CONDA_MODULE="${CONDA_MODULE:-anaconda3/2024.02}"
CONDA_ENV="${CONDA_ENV:-reno}"

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Must equal the number of jobs in the Slurm array above.
export ARRAY_COUNT=3

# Enable these only when all required model files are already cached locally.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Example:
# sbatch slurm/reno_with_init_inversion.sh \
#   --seed 3 --enable_imagereward --model pixart --save_some_images
python -u main_reno_with_init_inversion.py "$@"
