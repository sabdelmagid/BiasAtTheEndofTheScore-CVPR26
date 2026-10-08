import argparse
import os


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Process Reward Optimization.")

    # Paths
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="HF cache directory",
        default=os.environ.get("HF_HOME", "./model_cache"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save images",
        default="./outputs",
    )
    parser.add_argument("--show_images", default=False, help="flag to show the images as they're generated")
    parser.add_argument(
        "--set_weights_negative",
        default=False,
        action="store_true",
        help="set the weight of the reward model to be negative the default value",
    )

    # model and optim
    parser.add_argument("--model", type=str, help="Model to use", default="sdxl-turbo")
    parser.add_argument("--init_latents_path", type=str, help="pt to use", default=None)
    parser.add_argument("--save_some_images", default=False, action="store_true")
    parser.add_argument("--lr", type=float, help="Learning rate", default=5.0)
    parser.add_argument("--n_iters", type=int, help="Number of iterations", default=50)
    parser.add_argument(
        "--n_inference_steps", type=int, help="Number of iterations", default=1
    )
    parser.add_argument(
        "--optim",
        choices=["sgd", "adam", "lbfgs"],
        default="sgd",
        help="Optimizer to be used",
    )
    parser.add_argument("--nesterov", default=True, action="store_false")
    parser.add_argument(
        "--grad_clip", type=float, help="Gradient clipping", default=0.1
    )
    parser.add_argument("--seed", type=int, help="Seed to use", default=0)

    # Reward losses (default: all disabled unless explicitly enabled)

    parser.add_argument(
        "--enable_hps",
        default=False,
        action="store_true",
        help="Enable HPS reward",
    )
    parser.add_argument(
        "--hps_weighting",
        type=float,
        default=5.0,
        help="Weighting for HPS reward",
    )
    
    parser.add_argument(
        "--enable_imagereward",
        default=False,
        action="store_true",
        help="Enable ImageReward reward",
    )
    parser.add_argument(
        "--imagereward_weighting",
        type=float,
        default=1.0,
        help="Weighting for ImageReward",
    )
    
    parser.add_argument(
        "--enable_inversion",
        default=False,
        action="store_true",
        help="Enable CLIP-based Inversion reward",
    )
    parser.add_argument(
        "--inversion_weighting",
        type=float,
        default=1.0,
        help="Weighting for InversionLoss",
    )
    
    parser.add_argument(
        "--enable_clip",
        default=False,
        action="store_true",
        help="Enable CLIP reward",
    )
    parser.add_argument(
        "--clip_weighting",
        type=float,
        default=0.01,
        help="Weighting for CLIP reward",
    )
    
    parser.add_argument(
        "--enable_pickscore",
        default=False,
        action="store_true",
        help="Enable PickScore reward",
    )
    parser.add_argument(
        "--pickscore_weighting",
        type=float,
        default=0.05,
        help="Weighting for PickScore reward",
    )
    
    parser.add_argument(
        "--enable_aesthetic",
        default=False,
        action="store_true",
        help="Enable Aesthetic reward",
    )
    parser.add_argument(
        "--aesthetic_weighting",
        type=float,
        help="Weighting for Aesthetic",
        default=0.1,  # Scales the loss to approximately the same range as other rewards.
    )

    parser.add_argument(
        "--enable_incompression",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--incompression_weighting",
        type=float,
        help="Weighting for Incompression",
        default=1.10, 
    )

    parser.add_argument(
        "--disable_reg", default=True, action="store_false", dest="enable_reg"
    )
    parser.add_argument(
        "--reg_weight", type=float, help="Regularization weight", default=0.01
    )

    # task specific
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run",
        default="single",
        choices=[
            "t2i-compbench",
            "single",
            "parti-prompts",
            "geneval",
            "example-prompts",
        ],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to run",
        default="A red dog and a green cat",
    )
    parser.add_argument(
        "--benchmark_reward",
        help="Reward to benchmark on",
        default="total",
        choices=["ImageReward", "PickScore", "HPS", "CLIP", "total"],
    )

    # general
    parser.add_argument("--save_all_images", default=False, action="store_true")
    parser.add_argument("--no_optim", default=False, action="store_true")
    parser.add_argument("--imageselect", default=False, action="store_true")
    parser.add_argument("--memsave", default=False, action="store_true")
    parser.add_argument("--dtype", type=str, help="Data type to use", default="float16")
    parser.add_argument("--device_id", type=str, help="Device ID to use", default=None)
    parser.add_argument(
        "--cpu_offloading",
        help="Enable CPU offloading",
        default=False,
        action="store_true",
    )

    # optional multi-step model
    parser.add_argument("--enable_multi_apply", default=False, action="store_true")
    parser.add_argument(
        "--multi_step_model", type=str, help="Model to use", default="flux"
    )

    #args = parser.parse_args()
    
    
    #args = parser.parse_args()
    # Add this line to ignore Jupyter’s extra args:
    args, unknown = parser.parse_known_args()
    return args
