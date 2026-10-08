#!/bin/bash
# Public release: update YOUR_ACCOUNT and /path/to/conda/env for your cluster.
#SBATCH --job-name=skinperson
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --output=logs2_skinperson/%x_%A_%a.out
#SBATCH --error=logs2_skinperson/%x_%A_%a.err
#SBATCH --array=0-9              
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=25G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A YOUR_ACCOUNT

# --- setup ---
CONDA_ENV="/path/to/conda/env"
module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda config --add pkgs_dirs /path/to/conda/package_cache
conda activate $CONDA_ENV

export ARRAY_COUNT=10              


## call it like this sbatch run_skinperson.sh --data_root 'M=pixart$_$REG=True$_$REGW=0.0' --reward_model pickscore 
## note the single quote around the data_root! 
# --- run ---
python -u skin_person_detection.py "$@"
