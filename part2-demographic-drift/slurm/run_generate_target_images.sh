#!/bin/bash

# Activate the environment used for the experiment. Override CONDA_ENV when needed:
#   CONDA_ENV=/path/to/environment bash slurm/run_generate_target_images.sh
CONDA_ENV="${CONDA_ENV:-reno}"
conda activate "$CONDA_ENV"

python -u generate_target_images_flux.py
