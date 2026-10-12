import os
CACHE_DIR = "./model_cache"
os.environ['HF_HOME'] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import argparse
from skin_person_utils import (
    list_hash_dirs,
    list_iter_dirs,
    ensure_dir,
)
from PIL import Image
import torch
import json
from transformers import pipeline

def classification_to_flat(classification):
    """
    classification: list[dict] from HF image-classification pipeline
    Returns (top_label, top_score, raw_json_str)
    """
    if not classification:
        return None, None, None
    best = max(classification, key=lambda x: x.get("score", 0.0))
    return best.get("label"), float(best.get("score", 0.0)), json.dumps(classification)

def process_iter_batch(iter_path, parent_folder, hash_id, iter_name, model):
    """
    Processes all images in an iter_* folder and returns rows for a master CSV.
    Each row is a dict with flat fields.
    """
    image_paths = sorted([f for f in os.listdir(iter_path) if f.endswith(".png")])

    batch_size = 128
    classifications_all = []
    num_batches = max(1, math.ceil(len(image_paths) / batch_size))

    for i in range(0, len(image_paths), batch_size):
        print(f"BATCH {i//batch_size + 1}/{num_batches} in {hash_id}/{iter_name}")
        batch_files = image_paths[i:i + batch_size]
        batch_images = [
            Image.open(os.path.join(iter_path, f)).convert("RGB")
            for f in batch_files
        ]
        batch_captions = model(batch_images)
        classifications_all.extend(batch_captions)

    rows = []
    for img_name, classification in zip(image_paths, classifications_all):
        top_label, top_score, raw_json_str = classification_to_flat(classification)

        rel_path = os.path.join(hash_id, iter_name, img_name)
        abs_path = os.path.join(iter_path, img_name)

        rows.append({
            "parent_folder": parent_folder,
            "hash_folder": hash_id,
            "iter_folder": iter_name,
            "image_name": img_name,
            "image_rel_path": rel_path,
            "image_abs_path": abs_path,
            "nsfw_label": top_label,
            "nsfw_score": top_score,
            "nsfw_raw": raw_json_str,
        })

    return rows


def run_combo(i, args, model, parent_folder):
    """Process a single (hash, iter) combo index from all_rows list."""
    hash_id, iter_name = all_rows[i]
    iter_path = os.path.join(args.data_root, hash_id, iter_name)
    rows = process_iter_batch(iter_path, parent_folder, hash_id, iter_name, model)
    print(f"[OK] {hash_id}/{iter_name} processed.")
    return rows


import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSFW classification across hash and iter folders."
    )

    ## M=pixart$_$REG=True$_$REGW=0.0
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--reward_model", default=None)
    
    args = parser.parse_args()
    
  
    base_folder = "./ReNODEM/demographic_outputs/"
    parent_folder = args.data_root
    args.data_root = base_folder + args.data_root + f'/_{args.reward_model}'

    print(f"[Verbose]: ARGS.DATA_ROOT IS {args.data_root}")
    
    # Build list of (hash_id, iter_name)
    all_rows = []
    for h in list_hash_dirs(args.data_root):
        for it in list_iter_dirs(os.path.join(args.data_root, h)):
            all_rows.append((h, it))

    print(f"[Verbose]: There are {len(all_rows)} hash + iter folders which is ? * ?")

    model = pipeline(
        "image-classification",
        model="Freepik/nsfw_image_detector",
        device=0,
    )
    print("Model loaded!")

    total_combos = len(all_rows)

    if total_combos == 0:
        raise RuntimeError(f"No hash/iter folders found under: {args.data_root}")
    
    all_results = []
    
    for i in range(total_combos):
        print(f"[PROGRESS] Processing {i + 1}/{total_combos}: {all_rows[i]}")
        all_results.extend(
            run_combo(i, args, model, parent_folder)
        )
    
    measurements_root = os.path.join(
        "./ReNODEM_MEASUREMENTS/nsfw_classification",
        parent_folder,
        args.reward_model,
    )
    os.makedirs(measurements_root, exist_ok=True)
    
    output_csv = os.path.join(
        measurements_root,
        "nsfw_classification.csv",
    )
    
    fieldnames = [
        "parent_folder",
        "hash_folder",
        "iter_folder",
        "image_name",
        "image_rel_path",
        "image_abs_path",
        "nsfw_label",
        "nsfw_score",
        "nsfw_raw",
    ]
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"[DONE] Wrote {len(all_results)} rows to {output_csv}")