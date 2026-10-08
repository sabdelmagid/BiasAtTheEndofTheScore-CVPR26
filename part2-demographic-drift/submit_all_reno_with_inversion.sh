#!/bin/bash

models=("sdxl-turbo" "pixart" "sd-turbo")
reward_flags=(
  "--enable_pickscore"
  "--enable_imagereward"
  "--enable_hps"
  "--enable_clip"
  "--enable_aesthetic"
  "--enable_incompression"
)

for seed in 3 4; do
  for model in "${models[@]}"; do
    for reward_flag in "${reward_flags[@]}"; do
      for negative_flag in "" "--set_weights_negative"; do
        cmd="sbatch slurm/reno_with_init_inversion.sh $reward_flag --model $model --seed $seed --save_some_images $negative_flag"
        echo "Submitting: $cmd"

        jobid=$(eval $cmd | awk '{print $4}')
        echo "Submitted model=$model seed=$seed reward=$reward_flag negative=${negative_flag:-OFF} -> JobID $jobid"

        sleep 3
      done
    done
  done
done