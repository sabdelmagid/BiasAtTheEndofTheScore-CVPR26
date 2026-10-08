"""
Precompute CLIP ViT-L/14 embeddings for all Chicago Face Database (CFD) images.
Uses HuggingFace transformers CLIPModel.

Saves:
  - Individual embeddings per image in <base_dir>/vitL14_embeddings/<folder_name>/<image_stem>.pt
  - Anchor embeddings in <base_dir>/vitL14_embeddings/anchors/
      - 8 race+gender anchors: AF.pt, AM.pt, BF.pt, BM.pt, LF.pt, LM.pt, WF.pt, WM.pt
      - 2 gender anchors: F.pt, M.pt
      - 4 race anchors: A.pt, B.pt, L.pt, W.pt

Usage:
    python precompute_embeddings.py --cfd_dir "/path/to/CFD Version 3.0/Images/CFD"
"""

import argparse
import os
import glob
from collections import defaultdict
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def parse_race_gender(folder_name: str):
    """Extract race and gender from folder name like 'AF-200'."""
    prefix = folder_name.split("-")[0]
    if len(prefix) == 2 and prefix[0] in "ABLW" and prefix[1] in "FM":
        return prefix[0], prefix[1]
    return None, None


def get_image_files(folder_path: str):
    """Get all image files in a folder."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
        files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP ViT-L/14 embeddings for Chicago Face Database images."
    )
    parser.add_argument(
        "--cfd_dir",
        type=str,
        required=True,
        help="Path to the CFD directory containing subject folders (e.g., AF-200, BM-001, ...).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("MODEL_CACHE_DIR", "./model_cache"),
        help="Directory used to cache Hugging Face model files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for CLIP inference.",
    )
    args = parser.parse_args()

    cfd_dir = args.cfd_dir
    assert os.path.isdir(cfd_dir), f"Directory not found: {cfd_dir}"

    # Setup output directories
    embed_dir = os.path.join(cfd_dir, "vitL14_embeddings")
    anchor_dir = os.path.join(embed_dir, "anchors")
    os.makedirs(anchor_dir, exist_ok=True)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CLIP_MODEL_NAME} on {device} (cache: {args.cache_dir})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=args.cache_dir).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, cache_dir=args.cache_dir)
    model.eval()

    # Collect all subject folders
    subject_folders = sorted([
        d for d in os.listdir(cfd_dir)
        if os.path.isdir(os.path.join(cfd_dir, d)) and d != "vitL14_embeddings"
    ])

    print(f"Found {len(subject_folders)} subject folders")

    # Storage for computing anchors
    rg_embeddings = defaultdict(list)

    # Process each subject folder
    total_images = 0
    for folder_name in tqdm(subject_folders, desc="Processing subjects"):
        race, gender = parse_race_gender(folder_name)
        if race is None:
            print(f"Skipping unrecognized folder: {folder_name}")
            continue

        rg_key = f"{race}{gender}"

        folder_path = os.path.join(cfd_dir, folder_name)
        image_files = get_image_files(folder_path)

        if not image_files:
            print(f"No images found in {folder_name}")
            continue

        # Create output folder for this subject
        subject_embed_dir = os.path.join(embed_dir, folder_name)
        os.makedirs(subject_embed_dir, exist_ok=True)

        # Process in batches
        for i in range(0, len(image_files), args.batch_size):
            batch_files = image_files[i:i + args.batch_size]
            batch_images = []
            valid_files = []

            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    valid_files.append(img_path)
                except Exception as exc:
                    print(f"Error loading {img_path}: {exc}")

            if not batch_images:
                continue

            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu()

            for img_path, emb in zip(valid_files, embeddings):
                stem = Path(img_path).stem
                save_path = os.path.join(subject_embed_dir, f"{stem}.pt")
                torch.save(emb, save_path)
                rg_embeddings[rg_key].append(emb)
                total_images += 1

    print(f"\nProcessed {total_images} images total")

    # Compute and save anchor embeddings
    print("\nComputing anchor embeddings...")

    # 1. Race+Gender anchors (8)
    race_gender_keys = sorted(rg_embeddings.keys())
    print(f"Race+Gender groups: {race_gender_keys}")
    for rg_key in race_gender_keys:
        embs = torch.stack(rg_embeddings[rg_key])
        anchor = embs.mean(dim=0)
        anchor = anchor / anchor.norm()
        save_path = os.path.join(anchor_dir, f"{rg_key}.pt")
        torch.save(anchor, save_path)
        print(f"  {rg_key}: {len(rg_embeddings[rg_key])} images -> {save_path}")

    # 2. Gender anchors (2)
    for g in ["F", "M"]:
        all_embs = []
        for rg_key, embs in rg_embeddings.items():
            if rg_key[1] == g:
                all_embs.extend(embs)
        if all_embs:
            anchor = torch.stack(all_embs).mean(dim=0)
            anchor = anchor / anchor.norm()
            save_path = os.path.join(anchor_dir, f"{g}.pt")
            torch.save(anchor, save_path)
            print(f"  {g}: {len(all_embs)} images -> {save_path}")

    # 3. Race anchors (4)
    for r in ["A", "B", "L", "W"]:
        all_embs = []
        for rg_key, embs in rg_embeddings.items():
            if rg_key[0] == r:
                all_embs.extend(embs)
        if all_embs:
            anchor = torch.stack(all_embs).mean(dim=0)
            anchor = anchor / anchor.norm()
            save_path = os.path.join(anchor_dir, f"{r}.pt")
            torch.save(anchor, save_path)
            print(f"  {r}: {len(all_embs)} images -> {save_path}")

    print("\nDone! All embeddings saved to:", embed_dir)


if __name__ == "__main__":
    main()
