"""step 3 Run reward-guided optimization from learned inversion latents."""

import hashlib
import json
import logging
import os

import blobfile as bf
import pandas as pd
import torch
from pytorch_lightning import seed_everything

from arguments import parse_args
from models import get_model, get_multi_apply_fn
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer

CACHE_DIR = os.environ.get("HF_HOME", "./model_cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


TARGET_IMAGE_BASE = os.environ.get(
    "RENO_TARGET_IMAGE_DIR", "./target_images_for_inversion_flux"
)
METADATA_PATH = os.path.join(TARGET_IMAGE_BASE, "metadata.csv")
df = pd.read_csv(METADATA_PATH)
df_groups = df.groupby("prompt")
def get_group_row_by_idx(idx):
    row = None
    for group_idx, (group_prompt, group_rows) in enumerate(df_groups):
        if group_idx == idx:
            row = {'prompt': group_prompt,
                    'race': group_rows.iloc[0].race,
                    'gender': group_rows.iloc[0].gender,
                    'occupation': group_rows.iloc[0].occupation, 
                    'filenames': group_rows['filename'].tolist()} 
    
    return row 

# Each metadata prompt group defines one optimization run. 
print(f"Loaded ALL ROWS FROM {METADATA_PATH}")


# ===============================
# HELPER FUNCTIONS
# ===============================
def hash_prompt(prompt: str) -> str:
    """Create a unique short hash for a given prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    
def run_combo(idx, args):
    metric_name = ""
    if args.enable_pickscore:
        metric_name += "_pickscore"
    if args.enable_imagereward:
        metric_name += "_imagereward"
    if args.enable_hps:
        metric_name += "_hps"
    if args.enable_clip:
        metric_name += "_clip"
    if args.enable_aesthetic:
        metric_name += "_aesthetic"
    if args.enable_incompression:
        metric_name += "_incompression"

    if metric_name == "":
        raise ValueError("Enable at least one reward model before running optimization.")
    
    
    row = get_group_row_by_idx(idx)
    identity_prompt, race, gender, occupation = row["prompt"], row["race"], row["gender"], row["occupation"]

     
    
    folder_hash = hash_prompt(identity_prompt)
    # Use the underspecified occupation prompt and its learned latent initialization.
    occupation_prompt =  f"A photo of a {occupation}"
    target_image_number = "-".join([i.replace(".png","") for i in row['filenames']])
    
    occupation_savedir = os.path.join(f"./learned_inversion_outputs/seed_{args.seed}/{args.model}", folder_hash, f"iter_1_{target_image_number}")
    
    args.prompt = occupation_prompt    
    args.init_latents_path = os.path.join(occupation_savedir, "best_latents.pt")

    args.prompt = occupation_prompt
    args.task = "single"
    args.n_iters = 100
    
    if args.save_some_images:
        args.save_all_images = False
    else:
        args.save_all_images = True
    
    # The seed and base model must match the learned inversion run.
    
    

    if args.set_weights_negative:
        args.save_dir = os.path.join(f"./inversion_outputs/negative_optimization/seed_{args.seed}", args.model,  metric_name, occupation, folder_hash, f"iter_opt_{target_image_number}")
    else:
        args.save_dir = os.path.join(f"./inversion_outputs/positive_optimization/seed_{args.seed}", args.model, metric_name, occupation, folder_hash, f"iter_opt_{target_image_number}")
        
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)
    # --- include full dataframe row metadata ---
    #df_metadata = df_row.to_dict()  # converts race, gender, etc.
    
    record = {
       # **df_metadata,  # merges all columns from the CSV row
       # "identity_prompt": identity_prompt, 
        "prompt": args.prompt, 
        "race": race, 
        "gender": gender, 
        "occupation": occupation, 
        "folder_hash": folder_hash, 
        "init_latents_path": args.init_latents_path,
        "seed": args.seed,
        "save_dir": args.save_dir,
       
    }
    with open(os.path.join(args.save_dir, "metadata.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")

 
def main(args):
    seed_everything(args.seed)
    bf.makedirs(f"{args.save_dir}/logs/{args.task}")
    # Set up logging and name settings
    logger = logging.getLogger()
    logger.handlers.clear()
    
    for h in logger.handlers[:]:
        try:
            h.close()
        finally:
            logger.removeHandler(h)

    settings = (
        f"{args.model}{'_' + args.prompt if args.task == 't2i-compbench' else ''}"
        f"{'_no-optim' if args.no_optim else ''}_{args.seed if args.task != 'geneval' else ''}"
        f"_lr{args.lr}_gc{args.grad_clip}_iter{args.n_iters}"
        f"_reg{args.reg_weight if args.enable_reg else '0'}"
        f"{'_pickscore' + str(args.pickscore_weighting) if args.enable_pickscore else ''}"
        f"{'_clip' + str(args.clip_weighting) if args.enable_clip else ''}"
        f"{'_hps' + str(args.hps_weighting) if args.enable_hps else ''}"
        f"{'_imagereward' + str(args.imagereward_weighting) if args.enable_imagereward else ''}"
        f"{'_aesthetic' + str(args.aesthetic_weighting) if args.enable_aesthetic else ''}"
    )
    file_stream = open(f"{args.save_dir}/logs/{args.task}/{settings}.txt", "w")
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logging.info(args)
    if args.device_id is not None:
        logging.info(f"Using CUDA device {args.device_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda")
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    # Get reward losses
    reward_losses = get_reward_losses(args, dtype, device, args.cache_dir)

    # Get model and noise trainer
    pipe = get_model(
        args.model, dtype, device, args.cache_dir, args.memsave, args.cpu_offloading
    )
    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=args.n_iters,
        n_inference_steps=args.n_inference_steps,
        seed=args.seed,
        save_all_images=args.save_all_images,
        device=device,
        no_optim=args.no_optim,
        regularize=args.enable_reg,
        regularization_weight=args.reg_weight,
        grad_clip=args.grad_clip,
        log_metrics=args.task == "single" or not args.no_optim,
        imageselect=args.imageselect,
        save_some_images=args.save_some_images
        
    )

    # Create latents
    if args.model == "flux":
        # currently only support 512x512 generation
        shape = (1, 16 * 64, 64)
    elif args.model != "pixart":
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.unet.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    else:
        height = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        width = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.transformer.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    enable_grad = not args.no_optim
    if args.enable_multi_apply:
        multi_apply_fn = get_multi_apply_fn(
            model_type=args.multi_step_model,
            seed=args.seed,
            pipe=pipe,
            cache_dir=args.cache_dir,
            device=device,
            dtype=dtype,
        )
    else:
        multi_apply_fn = None

    if args.task == "single":
        #init_latents = torch.randn(shape, device=device, dtype=dtype)
        #latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
        x = torch.load(args.init_latents_path)
        print(f"Loaded initial latents from {args.init_latents_path}")
        latents = torch.nn.Parameter(x.clone(), requires_grad=enable_grad)
        
        optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
        save_dir = f"{args.save_dir}/"
        os.makedirs(f"{save_dir}", exist_ok=True)
        init_image, best_image, total_init_rewards, total_best_rewards, best_latents = trainer.train(
            latents, args.prompt, optimizer, save_dir, multi_apply_fn
        )
        best_image.save(f"{save_dir}/best_image.png")
        init_image.save(f"{save_dir}/init_image.png")
        
        #torch.save(best_latents, f"{save_dir}/best_latents.pt")
   
    else:
        raise ValueError(f"Unknown task {args.task}")
    # log total rewards
    logging.info(f"Mean initial rewards: {total_init_rewards}")
    logging.info(f"Mean best rewards: {total_best_rewards}")

    return 



if __name__ == "__main__":

   
    print(f"Loaded ALL ROWS FROM {METADATA_PATH}")

    
    slurm_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    slurm_count = int(os.environ.get("ARRAY_COUNT", 1))  

    total_combos = len(df_groups)

    if slurm_idx >= 0 and slurm_count > 0:
        combos_per_job = (total_combos + slurm_count - 1) // slurm_count
        start = slurm_idx * combos_per_job
        end = min((slurm_idx + 1) * combos_per_job, total_combos)

        print(f"[job {slurm_idx}/{slurm_count}] combos {start}–{end-1} of {total_combos}")
        for i in range(start, end):
            print("*" * 10)
            print(f"Running combination {i}")
            args = parse_args()

            if args.set_weights_negative:
                print("Using negative reward weights.")
                if args.enable_pickscore:
                    args.pickscore_weighting *= -1.0
                    print(f"pickscore_weighting={args.pickscore_weighting}")
                if args.enable_imagereward: 
                    args.imagereward_weighting *= -1.0
                    print(f"imagereward_weighting={args.imagereward_weighting}")
                if args.enable_hps: 
                    args.hps_weighting *= -1.0
                    print(f"hps_weighting={args.hps_weighting}")
                if args.enable_clip: 
                    args.clip_weighting *= -1.0
                    print(f"clip_weighting={args.clip_weighting}")
                if args.enable_aesthetic: 
                    args.aesthetic_weighting *= -1.0
                    print(f"aesthetic_weighting={args.aesthetic_weighting}")
                if args.enable_incompression: 
                    args.incompression_weighting *= -1.0
                    print(f"incompression_weighting={args.incompression_weighting}")
                if args.enable_lightness: 
                    args.lightness_weighting *= -1.0
                    print(f"lightness_weighting={args.lightness_weighting}")
                if args.enable_darkness: 
                    args.darkness_weighting *= -1.0
                    print(f"darkness_weighting={args.darkness_weighting}")
                
                
                
                        
                                

            
            run_combo(i, args)
    else:
        print("ERROR: Not running inside a SLURM array.")



