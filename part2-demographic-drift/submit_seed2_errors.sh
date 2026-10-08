#!/bin/bash

sbatch slurm/reno_with_init_inversion.sh --enable_aesthetic --model pixart --seed 2 --save_some_images --set_weights_negative
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_incompression --model pixart --seed 2 --save_some_images
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_incompression --model pixart --seed 2 --save_some_images --set_weights_negative
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_pickscore --model sd-turbo --seed 2 --save_some_images
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_pickscore --model sd-turbo --seed 2 --save_some_images --set_weights_negative
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_imagereward --model sd-turbo --seed 2 --save_some_images
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_imagereward --model sd-turbo --seed 2 --save_some_images --set_weights_negative
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_hps --model sd-turbo --seed 2 --save_some_images
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_hps --model sd-turbo --seed 2 --save_some_images --set_weights_negative
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_clip --model sd-turbo --seed 2 --save_some_images
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_clip --model sd-turbo --seed 2 --save_some_images --set_weights_negative
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_aesthetic --model sd-turbo --seed 2 --save_some_images
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_aesthetic --model sd-turbo --seed 2 --save_some_images --set_weights_negative
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_incompression --model sd-turbo --seed 2 --save_some_images
sleep 3
sbatch slurm/reno_with_init_inversion.sh --enable_incompression --model sd-turbo --seed 2 --save_some_images --set_weights_negative