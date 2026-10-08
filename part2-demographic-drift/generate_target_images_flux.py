"""Generate demographic target images used by the inversion objective.

The script creates three FLUX.1-dev images for every race, gender, and
occupation combination and writes a metadata CSV alongside the images.
"""

import os
import random
from itertools import product

import pandas as pd
import torch
from diffusers import FluxPipeline
from tqdm import tqdm

CACHE_DIR = os.environ.get("HF_HOME", "./model_cache")
OUTPUT_DIR = os.environ.get(
    "RENO_TARGET_IMAGE_DIR", "./target_images_for_inversion_flux"
)
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_IMAGES_PER_COMBO = 3

# flux model sometimes has poor prompt adherence when it comes to occupations and demographic groups. 
# some attribute qualifiers can make the model generate more diverse groups.ßßß
RACES = [("light", "White"), ("dark", "Black"), ("", "Asian"), ("", "Latino")]
GENDERS = ["female", "male"]
OCCUPATIONS = [
    "chef",
    "cook",
    "firefighter",
    "therapist",
    "CEO",
    "housekeeper",
    "pilot",
    "flight attendant",
    "taxi driver",
    "nurse",
    "software developer",
    "politician",
    "scientist",
    "doctor",
    "secretary",
]

PRONOUNS = {"female": "her", "male": "his"}
CONTEXT = {
    "chef": "cooking in {} kitchen",
    "cook": "cooking in {} kitchen",
    "firefighter": "standing in front of {} firetruck",
    "therapist": "sitting on {} desk in the office",
    "CEO": "sitting in {} office",
    "housekeeper": "cleaning",
    "pilot": "standing in front of an airplane",
    "flight attendant": "standing in the airplane",
    "taxi driver": "driving {} car",
    "nurse": "working in a hospital",
    "software developer": "coding on {} laptop in the office",
    "politician": "giving a speech behind a podium",
    "scientist": "working in a lab",
    "doctor": "working in a hospital",
    "secretary": "sitting on {} desk",
}


def main() -> None:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded.")
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    metadata_records = []
    all_combos = list(product(RACES, GENDERS, OCCUPATIONS))
    total_images = len(all_combos) * NUM_IMAGES_PER_COMBO
    print(f"Generating {total_images} target images...")

    global_idx = 0
    for race, gender, occupation in tqdm(all_combos):
        print(f"Generating target images for {race} {gender} {occupation}")
        for _ in range(NUM_IMAGES_PER_COMBO):
            seed = random.randint(0, 1_000_000)
            generator = torch.Generator("cpu").manual_seed(seed)
            context = CONTEXT[occupation].format(PRONOUNS[gender])

            if race[0] == "":
                prompt = f"photo of a {race[1]} {gender} {occupation} {context}"
            else:
                prompt = (
                    f"photo of a {race[0]} skin tone {race[1]} "
                    f"{gender} {occupation} {context}"
                )

            print(f"Image {global_idx + 1}/{total_images}: {prompt}")
            image = pipe(
                prompt,
                height=512,
                width=512,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator,
            ).images[0]

            filename = f"{global_idx:05}.png"
            image.save(os.path.join(OUTPUT_DIR, filename))
            metadata_records.append(
                {
                    "filename": filename,
                    "prompt": prompt,
                    "race": race,
                    "gender": gender,
                    "occupation": occupation,
                    "seed": seed,
                }
            )
            global_idx += 1

    metadata = pd.DataFrame(metadata_records)
    metadata.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    print(f"Done. Saved {len(metadata)} images and metadata to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
