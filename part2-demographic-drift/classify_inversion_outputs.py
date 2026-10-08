import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# =========================================================================
# Positional arguments: task_id, num_tasks
# =========================================================================
task_id  = int(sys.argv[1])   # SLURM_ARRAY_TASK_ID
num_tasks = int(sys.argv[2])  # total array size

# =========================================================================
# Configuration
# =========================================================================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = os.environ.get("HF_HOME", "./model_cache")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
ANCHOR_DIR = os.environ.get("RENO_ANCHOR_DIR", "./demographic_anchors")
BASE_DIR = os.environ.get("RENO_INVERSION_OUTPUT_DIR", "./inversion_outputs")
CSV_PATH    = os.path.join(BASE_DIR, "data_to_classify.csv")
OUT_DIR     = os.path.join(BASE_DIR, "classification_chunks")
os.makedirs(OUT_DIR, exist_ok=True)

RACE_LABELS   = {"A": "asian", "B": "black", "L": "latino", "W": "white"}
GENDER_LABELS = {"F": "female", "M": "male"}

# =========================================================================
# Load data chunk
# =========================================================================
df_full = pd.read_csv(CSV_PATH, index_col=0)
chunk_indices = list(range(task_id, len(df_full), num_tasks))
df = df_full.iloc[chunk_indices].copy().reset_index(drop=False)  # keep original index
print(f"Task {task_id}/{num_tasks}: processing {len(df)} rows")

# =========================================================================
# Load CLIP + anchors
# =========================================================================
model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, cache_dir=CACHE_DIR)
model.eval()

def load_anchors(anchor_dir):
    anchors = {}
    for fname in sorted(os.listdir(anchor_dir)):
        if fname.endswith(".pt"):
            key = fname.replace(".pt", "")
            anchors[key] = torch.load(os.path.join(anchor_dir, fname), map_location="cpu")
    return anchors

anchors = load_anchors(ANCHOR_DIR)
race_gender_anchors = {k: v for k, v in anchors.items() if len(k) == 2 and k[0] in "ABLW" and k[1] in "FM"}

# =========================================================================
# Embed
# =========================================================================
@torch.no_grad()
def embed_images(paths, batch_size=64):
    all_embeds = []
    failed_mask = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images, batch_failed = [], []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                batch_failed.append(False)
            except Exception as e:
                images.append(Image.new("RGB", (224, 224)))
                batch_failed.append(True)
                print(f"  Warning: could not load {p}: {e}")
        failed_mask.extend(batch_failed)
        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        embeds = model.get_image_features(**inputs)
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0), failed_mask

embeddings, failed_mask = embed_images(df["image_path"].tolist())

# =========================================================================
# Classify
# =========================================================================
def classify_joint(df, embeddings, failed_mask):
    keys = sorted(race_gender_anchors.keys())
    anchor_embeds = torch.stack([race_gender_anchors[k] for k in keys], dim=0)
    sims  = embeddings @ anchor_embeds.T
    pred_indices = sims.argmax(dim=-1).tolist()
    pred_keys    = [keys[idx] for idx in pred_indices]

    gender_preds = [GENDER_LABELS[k[1]] for k in pred_keys]
    race_preds   = [RACE_LABELS[k[0]]   for k in pred_keys]

    for i, failed in enumerate(failed_mask):
        if failed:
            gender_preds[i] = "error"
            race_preds[i]   = "error"

    df = df.copy()
    df["pred_gender"] = gender_preds
    df["pred_race"]   = race_preds

    for j, key in enumerate(keys):
        col = f"rg_sim_{RACE_LABELS[key[0]]}_{GENDER_LABELS[key[1]]}"
        df[col] = sims[:, j].tolist()
        df.loc[[i for i, f in enumerate(failed_mask) if f], col] = float("nan")

    return df

df = classify_joint(df, embeddings, failed_mask)

# =========================================================================
# Save chunk
# =========================================================================
out_path = os.path.join(OUT_DIR, f"chunk_{task_id:04d}.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")