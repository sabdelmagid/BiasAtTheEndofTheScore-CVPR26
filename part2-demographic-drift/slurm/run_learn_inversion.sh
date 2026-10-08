#!/bin/bash
#SBATCH --job-name=learninv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=24:00:00
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
export ARRAY_COUNT=5

# Example:
# sbatch slurm/run_learn_inversion.sh --model sdxl-turbo --seed 0
python -u main_inversion.py "$@"
