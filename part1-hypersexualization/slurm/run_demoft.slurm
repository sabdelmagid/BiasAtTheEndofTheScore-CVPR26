#!/bin/bash
# Public release: update YOUR_ACCOUNT and /path/to/conda/env for your cluster.
#SBATCH --job-name=demoft
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/logs_demoft/%x_%A_%a.out
#SBATCH --error=logs/logs_demoft/%x_%A_%a.err
#SBATCH --array=0-49              
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A YOUR_ACCOUNT

# --- setup ---
CONDA_ENV="/path/to/conda/env"
module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda config --add pkgs_dirs /path/to/conda/package_cache
conda activate $CONDA_ENV

export ARRAY_COUNT=50              
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

## call it like this sbatch run_demoft.sh --enable_pickscore sbatch ./slurm/run_demoft.sh --enable_imagereward
# --- run ---
python -u main_n_iter.py "$@"
