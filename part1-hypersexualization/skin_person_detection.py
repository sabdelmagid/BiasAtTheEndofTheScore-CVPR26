#!/usr/bin/env python3
## "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
#### IMPORTANTE ! ! ! 
### due to indexing the files must be re-read back as follows: 
#import numpy as np, json
#p = "./ReNODEM/demographic_outputs/_pickscore/00066aaa63c752b3/iter_0/skin_person_detections/"  
#seg = np.load(p + "segmentations.npz")
#with open(p + "index.json") as f:
#    idx = json.load(f)["init_image.png"]
#category_mask = seg["category_mask"][idx]
#conf = seg["confidence_masks"][idx]

import os
import argparse
from skin_person_utils import (
    list_hash_dirs,
    list_iter_dirs,
    list_iter_dirs_inversion,
    list_images,
    ensure_dir,
    load_segmenter,
    process_iter_batch,
    save_json,
)

segmenter = None  # global

def run_combo(i, args, parent_folder):
    """Process a single (hash, iter) combo index from all_rows list."""
    hash_id, iter_name = all_rows[i]
    iter_path = os.path.join(args.data_root, hash_id, iter_name)

    if not os.path.isdir(iter_path):
        print(f"[WARN] Missing iter path: {iter_path}")
        return []

    out_dir = os.path.join(iter_path, "skin_person_detections")
    ensure_dir(out_dir)

    global segmenter
    if segmenter is None:
        segmenter = load_segmenter(args.model_path)  # assuming you have this arg

    result = process_iter_batch(segmenter, iter_path, out_dir)
    if result is None:
        return []

    seg_abs = result["seg_path"]
    # path relative to the root for this (M, reward_model)
    seg_rel = os.path.relpath(seg_abs, args.data_root)

    rows = []
    for meta in result["meta"]:
        img_name = meta["image"]
        idx = result["index_map"][img_name]

        row = {
            "parent_folder": parent_folder,         # e.g. M=pixart$_$...
            "reward_model": args.reward_model,      # e.g. pickscore
            "hash_folder": hash_id,
            "iter_folder": iter_name,
            "image_name": img_name,
            "image_rel_path": os.path.join(hash_id, iter_name, img_name),
            "seg_rel_path": seg_rel,                # where masks live
            "seg_index": idx,                       # index into the NPZ
        }

        # add meta fields (without duplicating "image")
        for k, v in meta.items():
            if k == "image":
                continue
            row[k] = v

        rows.append(row)

    print(f"[OK] {hash_id}/{iter_name} processed ({len(rows)} images).")
    return rows

import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skin/person segmentation across hash and iter folders."
    )
    #python skin_person_detection.py --data_root M='sdxl-turbo$_$REG=True$_$REGW=0.01' --reward_model pickscore 
    parser.add_argument("--data_root", default=None)   # M=...
    parser.add_argument("--reward_model", default=None)
    parser.add_argument("--model_path", default="./model_cache/selfie_multiclass_256x256.tflite")  # used by load_segmenter
    args = parser.parse_args()

    base_folder = "./ReNODEM/demographic_outputs/"
    parent_folder = args.data_root                     # e.g. "M=pixart$_$REG=True$_$REGW=0.0"

    # actual root = base_folder / M=... / _reward_model
    args.data_root = os.path.join(base_folder, args.data_root, f"_{args.reward_model}")

    print(f"[Verbose]: ARGS.DATA_ROOT IS {args.data_root}")

    # Build list of (hash_id, iter_name)
    all_rows = []
    for h in list_hash_dirs(args.data_root):
        for it in list_iter_dirs(os.path.join(args.data_root, h)):
            all_rows.append((h, it))

    print(f"[Verbose]: There are {len(all_rows)} hash + iter folders which is ? * ?")

    segmenter = None

    total_combos = len(all_rows)

    if total_combos == 0:
        raise RuntimeError(f"No hash/iter folders found under: {args.data_root}")
    
    all_results = []
    
    for i in range(total_combos):
        print(f"[PROGRESS] Processing {i + 1}/{total_combos}: {all_rows[i]}")
        all_results.extend(
            run_combo(i, args, parent_folder)
        )
    
    measurements_root = os.path.join(
        "./ReNODEM_MEASUREMENTS/skinperson_detection",
        parent_folder,
        args.reward_model,
    )
    os.makedirs(measurements_root, exist_ok=True)
    
    csv_path = os.path.join(
        measurements_root,
        "skin_person_detection.csv",
    )
    
    if all_results:
        fieldnames = list(all_results[0].keys())
    
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
        print(f"[DONE] Wrote {len(all_results)} rows to {csv_path}")
    else:
        print("[INFO] No rows were produced.")
    
    
