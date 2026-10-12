"""step 2 Learn demographic-conditioned initial noise for underspecified prompts."""

import ast
import hashlib
import json
import logging
import os
import time

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
INVERSION_ITERS = 200 

TARGET_IMAGE_BASE = os.environ.get(
    "RENO_TARGET_IMAGE_DIR", "./target_images_for_inversion_flux"
)


METADATA_PATH = os.path.join(TARGET_IMAGE_BASE, "metadata.csv")
df = pd.read_csv(METADATA_PATH)
df_groups = df.groupby("prompt")

#race_mapper = {"light": "White", "dark":"Black"}


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

# Each metadata prompt group defines one inversion target. 
print(f"Loaded ALL ROWS FROM {METADATA_PATH}")

# ===============================
# HELPER FUNCTIONS
# ===============================
def hash_prompt(prompt: str) -> str:
    """Create a unique short hash for a given prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def run_combo(idx, args):
    row = get_group_row_by_idx(idx)
    identity_prompt, race, gender, occupation = row["prompt"], row["race"], row["gender"], row["occupation"]

     
    folder_hash = hash_prompt(identity_prompt)

   
    args.task = "single"
    args.enable_inversion = True
    args.n_iters = INVERSION_ITERS 

    # Learned latents are organized by experiment seed and base model.
    LATENTS_PATH = f"./learned_inversion_outputs/seed_{args.seed}/{args.model}"
    race = ast.literal_eval(race)
    
    args.inversion_target_texts = [f"{race[1]} {gender} {occupation}", f"{gender}", f"{occupation}", f"{race[1]} {gender}"]
    if gender == 'female':
        args.inversion_target_texts.append("long hair")
        
    args.inversion_target_images = [
        os.path.join(TARGET_IMAGE_BASE, filename) for filename in row["filenames"]
    ] 
    target_image_number = "-".join([i.replace(".png","") for i in row['filenames']])

    # Use the opposite gender and common failure modes as negative text anchors.
    opp_gender = 'female' if gender == 'male' else 'male'
    args.inversion_target_negative_texts = [opp_gender, "b&w photo", "sepia photo", "two people", "many people"]
    
    
    # Two-stage inversion: gender + occupation, then occupation only.

    # Stage 1: optimize for the gender-and-occupation prompt.
    gender_occupation_prompt = f"A photo of a {gender} {occupation}"
    gender_occupation_savedir = os.path.join(LATENTS_PATH, folder_hash, f"iter_0_{target_image_number}")
    args.prompt = gender_occupation_prompt
    args.save_dir = gender_occupation_savedir
    os.makedirs(args.save_dir, exist_ok=True)
    main(args) 
    
    time.sleep(1)
    
    # Stage 2: continue from Stage 1 using the underspecified occupation prompt.
    occupation_prompt =  f"A photo of a {occupation}"
    occupation_savedir = os.path.join(LATENTS_PATH, folder_hash, f"iter_1_{target_image_number}")
    args.prompt = occupation_prompt
    args.save_dir = occupation_savedir
    os.makedirs(args.save_dir, exist_ok=True)
    args.init_latents_path = os.path.join(gender_occupation_savedir, "best_latents.pt")
    main(args)
     
    
    record = {
      
        "identity_prompt": identity_prompt, 
        "target_texts": args.inversion_target_texts , 
        "negative_target_texts": args.inversion_target_negative_texts , 
        "target_image_paths": args.inversion_target_images, 

        "gender_occupation_prompt":gender_occupation_prompt , 
        "occupation_prompt": occupation_prompt, 
         
        "race": race, 
        "gender": gender, 
        "occupation": occupation, 
        
        "folder_hash": folder_hash, 
        "path_to_best_latents": f"{args.save_dir}/best_latents.pt",
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

        if args.init_latents_path:
            x = torch.load(args.init_latents_path)
            print(f"Loaded initial latents from {args.init_latents_path}")
            latents = torch.nn.Parameter(x.clone(), requires_grad=enable_grad)
        else:
            init_latents = torch.randn(shape, device=device, dtype=dtype)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            
        optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
        save_dir = f"{args.save_dir}/"
        os.makedirs(f"{save_dir}", exist_ok=True)
        init_image, best_image, total_init_rewards, total_best_rewards, best_latents = trainer.train(
            latents, args.prompt, optimizer, save_dir, multi_apply_fn
        )
        best_image.save(f"{save_dir}/best_image.png")
        init_image.save(f"{save_dir}/init_image.png")
        
        torch.save(best_latents, f"{save_dir}/best_latents.pt")
   
    else:
        raise ValueError(f"Unknown task {args.task}")
    # log total rewards
    logging.info(f"Mean initial rewards: {total_init_rewards}")
    logging.info(f"Mean best rewards: {total_best_rewards}")
    
    # Free everything heavy
    cleanup_pipeline(pipe, trainer, optimizer, latents, multi_apply_fn, reward_losses)

    return 

import gc
import torch

def cleanup_pipeline(*objs):
    """
    Safely release GPU memory used by diffusers pipelines and related objects.
    Pass in anything holding CUDA tensors (pipe, trainer, optimizer, latents, images, etc.)
    """
    for obj in objs:
        if obj is None:
            continue
        # Try to move modules to CPU first to drop CUDA context refs
        try:
            if hasattr(obj, "to"):
                obj.to("cpu")
        except Exception:
            pass
        # For diffusers pipelines, also try to push components to CPU
        try:
            # Most pipelines expose .components (unet, vae, text_encoder, etc.)
            comps = getattr(obj, "components", None)
            if isinstance(comps, dict):
                for m in comps.values():
                    try:
                        if hasattr(m, "to"):
                            m.to("cpu")
                    except Exception:
                        pass
        except Exception:
            pass

    # Drop all references
    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    
    slurm_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    slurm_count = int(os.environ.get("ARRAY_COUNT", 1))  
    total_combos = len(df_groups)

    if slurm_idx >= 0 and slurm_count > 0:
        combos_per_job = (total_combos + slurm_count - 1) // slurm_count
        start = slurm_idx * combos_per_job
        end = min((slurm_idx + 1) * combos_per_job, total_combos)

        print(f"[job {slurm_idx}/{slurm_count}] combos {start}–{end-1} of {total_combos}")
        for i in range(start, end):
            args = parse_args()
            run_combo(i, args)
    else:
        print("ERROR: Not running inside a SLURM array.")



