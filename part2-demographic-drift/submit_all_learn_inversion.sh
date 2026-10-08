#!/bin/bash

for seed in 1 2 3 4; do
  for model in sdxl-turbo pixart sd-turbo; do
    jobid=$(sbatch ./slurm/run_learn_inversion.sh --model $model --seed $seed | awk '{print $4}')
    echo "Submitted $model seed $seed -> JobID $jobid"
    sleep 1
  done
done
