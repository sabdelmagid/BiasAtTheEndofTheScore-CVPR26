import os
import re
import json
import numpy as np
import mediapipe as mp
from datetime import datetime

# ---------------------------
# Constants
# ---------------------------
SKIN_LABELS = {2, 3}  # body-skin + face-skin

##0 - background
#1 - hair
#2 - body-skin
#3 - face-skin
#4 - clothes
#5 - others (accessories)

# ---------------------------
# Utility functions
# ---------------------------

def list_hash_dirs(root):
    """Return sorted list of hash-id subdirectories under root (_pickscore)."""
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def list_iter_dirs(hash_dir_path):
    """Return sorted list of iter_* subdirectories."""
    iters = [d for d in os.listdir(hash_dir_path) if re.match(r"^iter_\d+$", d)]
    iters.sort(key=lambda s: int(s.split("_")[1]))
    return iters

def list_iter_dirs_inversion(hash_dir_path):
    """Return sorted list of iter_* subdirectories but the sorting here doesnt
    actually matter because it is just number of target images."""
    iters = [d for d in os.listdir(hash_dir_path) if re.match(r"^iter_opt_\d+$", d)]
    iters.sort(key=lambda s: int(s.split("_")[2]))
    return iters


def list_images(iter_dir_path):
    """List *.png images in numerical order."""
    pngs = [f for f in os.listdir(iter_dir_path) if f.endswith(".png")]
    return sorted(
        pngs,
        key=lambda fn: (re.search(r"(\d+)", fn) and int(re.search(r"(\d+)", fn).group(1)))
        if re.search(r"(\d+)", fn)
        else float("inf"),
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_segmenter(model_path):
    """Initialize MediaPipe ImageSegmenter."""
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = ImageSegmenterOptions(
       # base_options=BaseOptions(model_asset_path=model_path), because of known issues with numpy, unfortunately gonna have to run this on cpu
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU, ),
        running_mode=RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=True,
    )
    return mp.tasks.vision.ImageSegmenter.create_from_options(options)


# ---------------------------
# Core per-image computation
# ---------------------------

import numpy as np
import json
import os
from datetime import datetime

def process_iter_batch(segmenter, iter_path, out_dir):
    """
    Processes all images in an iter_* folder, saves one NPZ with masks,
    and RETURNS per-image metadata + index mapping (no JSON/JSONL).
    """
    images = sorted([f for f in os.listdir(iter_path) if f.endswith(".png")])
    if not images:
        print(f"[INFO] No images in {iter_path}")
        return None  # nothing to do

    all_meta = []
    all_category = []
    all_conf = []
    index_map = {}

    for idx, img_name in enumerate(images):
        img_path = os.path.join(iter_path, img_name)
        mp_image = mp.Image.create_from_file(img_path)
        result = segmenter.segment(mp_image)

        category_mask = result.category_mask.numpy_view()
        conf_masks = [cm.numpy_view() for cm in result.confidence_masks]
        h, w = category_mask.shape
        image_area = h * w

        # pixel stats
        person_mask = category_mask != 0
        skin_mask = np.isin(category_mask, [2, 3])
        person_pixels = int(np.count_nonzero(person_mask))
        skin_pixels = int(np.count_nonzero(skin_mask))
        skin_to_person = skin_pixels / person_pixels if person_pixels else 0.0
        skin_to_image = skin_pixels / image_area
        person_to_image = person_pixels / image_area

        meta = {
            "image": img_path,
            "height": h,
            "width": w,
            "person_area": person_pixels,
            "skin_area": skin_pixels,
            "skin_to_person": skin_to_person,
            "skin_to_image": skin_to_image,
            "person_to_image": person_to_image,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        all_meta.append(meta)

        all_category.append(category_mask.astype(np.uint8))
        all_conf.append(np.stack(conf_masks).astype(np.float16))
        index_map[img_path] = idx

    # Stack and save masks
    category_stack = np.stack(all_category)       # (N, H, W)
    conf_stack = np.stack(all_conf)              # (N, C, H, W)

    os.makedirs(out_dir, exist_ok=True)
    seg_path = os.path.join(out_dir, "segmentations.npz")

    np.savez_compressed(
        seg_path,
        category_mask=category_stack,
        confidence_masks=conf_stack,
    )

    print(f"[DONE] Saved {len(images)} segmentations for {iter_path} -> {seg_path}")

    # Instead of writing index.json + results.jsonl, just return both:
    return {
        "seg_path": seg_path,
        "index_map": index_map,
        "meta": all_meta,
    }


import numpy as np
import json
import os
from datetime import datetime

def process_image(segmenter, image_path, out_dir=None):
    """
    Run MediaPipe segmentation, compute per-class metrics,
    and optionally dump raw segmentation results (.npz).
    """
    mp_image = mp.Image.create_from_file(image_path)
    result = segmenter.segment(mp_image)

    category_mask = result.category_mask.numpy_view()        # uint8 (H,W)
    conf_masks = [cm.numpy_view() for cm in result.confidence_masks]  # list of (H,W,float32)
    h, w = category_mask.shape
    image_area = int(h * w)

    # --- Per-class statistics ---
    per_class = {}
    num_classes = len(conf_masks)
    for cls_id in range(num_classes):
        mask = category_mask == cls_id
        pixels = int(np.count_nonzero(mask))
        conf_mean = float(conf_masks[cls_id][mask].mean()) if pixels > 0 else 0.0
        conf_sum = float(conf_masks[cls_id][mask].sum()) if pixels > 0 else 0.0
        per_class[cls_id] = {
            "pixels": pixels,
            "confidence_mean": conf_mean,
            "confidence_sum": conf_sum,
        }

    # --- Aggregate areas ---
    person_mask = category_mask != 0
    skin_mask = np.isin(category_mask, list(SKIN_LABELS))
    person_pixels = int(np.count_nonzero(person_mask))
    skin_pixels = int(np.count_nonzero(skin_mask))

    skin_to_person = skin_pixels / person_pixels if person_pixels > 0 else 0.0
    skin_to_image = skin_pixels / image_area if image_area > 0 else 0.0
    person_to_image = person_pixels / image_area if image_area > 0 else 0.0

    # --- Save raw arrays (optional) ---
    raw_save_path = None
    if out_dir is not None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        raw_save_path = os.path.join(out_dir, f"{base}_raw_masks.npz")
        np.savez_compressed(
            raw_save_path,
            category_mask=category_mask.astype(np.uint8),
            confidence_masks=np.stack(conf_masks).astype(np.float32),
        )

    return {
        "image_path": image_path,
        "height": h,
        "width": w,
        "image_area": image_area,
        "person_area": person_pixels,
        "skin_area": skin_pixels,
        "skin_to_person": skin_to_person,
        "skin_to_image": skin_to_image,
        "person_to_image": person_to_image,
        "per_class": per_class,
        "raw_mask_file": raw_save_path,
        "status": "ok" if person_pixels > 0 else "no_person_detected",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
