import os
CACHE_DIR = "./model_cache"
os.environ['HF_HOME'] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hashlib

import logging
import blobfile as bf
import torch
from datasets import load_dataset
from pytorch_lightning import seed_everything
from arguments import parse_args
from models import get_model, get_multi_apply_fn
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer
import random
import json
import pandas as pd
 
CSV_PROMPTSET_PATH = "./prompt_sets/demographic_prompts_reno.csv"
df = pd.read_csv(CSV_PROMPTSET_PATH)
all_rows = [i[1] for i in df.iterrows()]
print(f"Loaded ALL ROWS FROM {CSV_PROMPTSET_PATH}")

# ===============================
# HELPER FUNCTIONS
# ===============================
def hash_prompt(prompt: str) -> str:
    """Create a unique short hash for a given prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def run_combo(idx, args):

    ## note that many things are overwritten here! 
    ## we use default 100 iters
    
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
        assert -1==1, "At least one reward metric must be enabled."
    
    
    print("The metric name is ---->", metric_name) 
    
    df_row = all_rows[idx]
    prompt = df_row['prompt'] ## !
    print(prompt)
    args.prompt = prompt
    folder_hash = hash_prompt(prompt)
    args.task = "single"
    args.n_iters = 100

    ## manually setting here. hehe not the cleanest way
    args.save_some_images = True 
    
    if args.save_some_images:
        args.save_all_images = False
    else:
        args.save_all_images = True

    ## run each prompt 5 times with a different seed.
    for iter in range(5):
       
        args.seed = random.randint(0, 2**32 - 1)
        path_key = f"M={args.model}$_$REG={args.enable_reg}$_$REGW={args.reg_weight}"
        
        args.save_dir = os.path.join("./ReNODEM/demographic_outputs", path_key, metric_name, folder_hash, f"iter_{iter}")
        
        os.makedirs(args.save_dir, exist_ok=True)
       
    
        init_img, best_img, init_rewards, best_rewards = train_main(args)
        # --- include full dataframe row metadata ---
        df_metadata = df_row.to_dict()  # converts race, gender, etc.
        
        record = {
            **df_metadata,  # merges all columns from the CSV row
            
            "folder_hash": folder_hash, 
            "seed": args.seed,
            "save_dir": args.save_dir,
            "rewards": {
                "initial": init_rewards,
                "best": best_rewards,
            },
        }
        with open(os.path.join(args.save_dir, "metadata.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")


def train_main(args):
    seed_everything(args.seed)
    bf.makedirs(f"{args.save_dir}/logs/{args.task}")
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

    print(" ARGS. ENABLE REG IS -----> ", args.enable_reg)
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

    init_latents = torch.randn(shape, device=device, dtype=dtype)
    latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)

    optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
    
    
    init_image, best_image, total_init_rewards, total_best_rewards = trainer.train(
        latents, args.prompt, optimizer, args.save_dir, multi_apply_fn
    )
    best_image.save(f"{args.save_dir}/best_image.png")
    init_image.save(f"{args.save_dir}/init_image.png")
        
    # log total rewards
    logging.info(f"Mean initial rewards: {total_init_rewards}")
    logging.info(f"Mean best rewards: {total_best_rewards}")

    return init_image, best_image, total_init_rewards, total_best_rewards

 

if __name__ == "__main__":
    args = parse_args()

    if not all_rows:
        raise RuntimeError("No examples were found in all_rows.")

    for combo_idx in range(len(all_rows)):
        print(f"Running example {combo_idx + 1}/{len(all_rows)}")
        run_combo(combo_idx, args)



