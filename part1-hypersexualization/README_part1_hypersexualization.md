# Hypersexualization Under Reward-Guided Optimization

This directory contains the code for the **hypersexualization experiments** in our paper.

The experiment tests whether reward-guided optimization changes the amount of sexualized content in generated images, and whether these changes differ across demographic groups. Starting from prompts that explicitly specify demographic attributes, we optimize the initial noise of a one-step text-to-image model using a selected reward model and then compare the initial and optimized images using:

1. an NSFW image classifier; and
2. a skin-exposure metric based on semantic person segmentation.

The main workflow is:

```text
demographic prompt set
        │
        ▼
main_n_iter.py
        │
        ▼
initial + reward-optimized images
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
nsfw_classification.py        skin_person_detection.py
        │                               │
        ▼                               ▼
NSFW classifications          skin/person segmentation metrics
```

---

## Repository structure

```text
main_n_iter.py
    Main entrypoint for reward-guided noise optimization.

arguments.py
    Shared command-line arguments and default ReNO hyperparameters.

models/
    Differentiable one-step text-to-image generation pipelines.

rewards/
    Reward/loss implementations, including PickScore, ImageReward,
    HPS, CLIP, Aesthetic, and Incompression.

training/
    Latent-noise optimization loop and optimizer utilities.

nsfw_classification.py
    Run the pretrained NSFW classifier over generated images.

skin_person_detection.py
    Run person/skin segmentation over generated images and save
    skin-exposure measurements.

skin_person_utils.py
    Shared segmentation and filesystem utilities.
```

---

# Setup

The code expects a CUDA-capable environment with the dependencies used by the generation, reward, and measurement models.

Core dependencies include:

```text
torch
torchvision
diffusers
transformers
huggingface_hub
safetensors
pytorch-lightning
datasets
pandas
numpy
Pillow
tqdm
blobfile
ImageReward
hpsv2
openai-clip
mediapipe
```

Model weights are cached locally by default:

```text
./model_cache
```

If access to a gated Hugging Face model is required, export your token in the shell:

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
```

Do not place authentication tokens directly in source files.

---

## Required prompt file

`main_n_iter.py` expects the hypersexualization prompt set at:

```text
prompt_sets/demographic_prompts_reno.csv
```

At minimum, this CSV must contain a column named:

```text
prompt
```

Any additional columns in the CSV are copied into each run's `metadata.jsonl`, allowing demographic attributes and other prompt metadata to be associated with the generated images.

---

## Required auxiliary model files

### Aesthetic predictor

When using:

```text
--enable_aesthetic
```

place the aesthetic-predictor checkpoint at:

```text
model_cache/ava+logos-l14-linearMSE.pth
```

### MediaPipe segmentation model

The skin-exposure pipeline expects the MediaPipe multiclass selfie-segmentation model at:

```text
model_cache/selfie_multiclass_256x256.tflite
```

The model used in the experiments is the MediaPipe `selfie_multiclass_256x256` image segmenter, available at:

```text
https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

---

# Part A: Reward-guided image optimization

The main experiment is run through:

```bash
python main_n_iter.py <reward flag>
```

For example:

```bash
python main_n_iter.py --enable_pickscore
```

or:

```bash
python main_n_iter.py --enable_imagereward
```

Supported reward flags include:

```text
--enable_pickscore
--enable_imagereward
--enable_hps
--enable_clip
--enable_aesthetic
--enable_incompression
```

One reward objective should normally be enabled per run.

Supported base models include:

```text
sdxl-turbo
sd-turbo
pixart
```

For example:

```bash
python main_n_iter.py \
    --model sdxl-turbo \
    --enable_pickscore
```

The code also contains support for additional model variants inherited from the ReNO implementation.

---

## Optimization settings

The release uses the default ReNO optimization settings.

---

## Reward values in the logs

The quantities logged under names such as:

```text
ImageReward
PickScore
HPS
```

are the **optimization losses**, not necessarily the raw reward-model scores.

The reward implementations are written so that minimizing the loss corresponds to maximizing the desired reward. Therefore, for some objectives, a decreasing logged value indicates that the underlying reward score is increasing.

---

## Generated outputs

For each model/reward configuration, outputs are written under:

```text
ReNODEM/demographic_outputs/
```

with a structure of the form:

```text
ReNODEM/
└── demographic_outputs/
    └── M=<model>$_$REG=<bool>$_$REGW=<weight>/
        └── _<reward>/
            └── <prompt_hash>/
                ├── iter_0/
                │   ├── init_image.png
                │   ├── best_image.png
                │   ├── metadata.jsonl
                │   └── logs/
                ├── iter_1/
                ├── ...
                └── iter_4/
```

Each prompt is identified by a deterministic SHA-256-derived short hash.

Each `metadata.jsonl` record contains the original prompt metadata, sampled seed, output directory, and the initial/best reward values.

When `save_some_images` is enabled, intermediate optimization images are also saved periodically.

---

# Part B: NSFW classification

After reward optimization, run:

```bash
python nsfw_classification.py \
    --data_root 'M=sdxl-turbo$_$REG=True$_$REGW=0.01' \
    --reward_model pickscore
```

**Use single quotes around `--data_root`.** The directory name contains `$` characters, which Bash otherwise interprets as shell-variable expansion.

The script evaluates all PNG images under the selected model/reward output directory using:

```text
Freepik/nsfw_image_detector
```

and writes one CSV:

```text
ReNODEM_MEASUREMENTS/
└── nsfw_classification/
    └── <model-config>/
        └── <reward>/
            └── nsfw_classification.csv
```

The output includes:

```text
parent_folder
hash_folder
iter_folder
image_name
image_rel_path
image_abs_path
nsfw_label
nsfw_score
nsfw_raw
```

`nsfw_raw` stores the complete classifier output for the image, while `nsfw_label` and `nsfw_score` store the highest-scoring class.

---

# Part C: Skin-exposure measurement

Skin exposure is measured using MediaPipe multiclass person segmentation.

Run:

```bash
python skin_person_detection.py \
    --data_root 'M=sdxl-turbo$_$REG=True$_$REGW=0.01' \
    --reward_model pickscore
```

The default segmentation model path is:

```text
./model_cache/selfie_multiclass_256x256.tflite
```

A different file can be supplied with:

```bash
--model_path /path/to/selfie_multiclass_256x256.tflite
```

The MediaPipe segmentation labels used by this code are:

| Label | Region |
|---:|---|
| 0 | background |
| 1 | hair |
| 2 | body skin |
| 3 | face skin |
| 4 | clothes |
| 5 | other/accessories |

Skin pixels are defined as labels **2 and 3**.

The primary quantities computed for each image are:

```text
person_area
skin_area
skin_to_person
skin_to_image
person_to_image
```

The paper's skin-exposure measurement is represented by:

```text
skin_to_person = skin pixels / visible person pixels
```

The script also stores segmentation arrays in compressed `.npz` files under each processed iteration directory.

Summary measurements are written to:

```text
ReNODEM_MEASUREMENTS/
└── skinperson_detection/
    └── <model-config>/
        └── <reward>/
            └── skin_person_detection.csv
```

---

# End-to-end example

For PickScore with SDXL-Turbo:

```bash
# 1. Generate and optimize images.
python main_n_iter.py \
    --model sdxl-turbo \
    --enable_pickscore

# 2. Run NSFW classification.
python nsfw_classification.py \
    --data_root 'M=sdxl-turbo$_$REG=True$_$REGW=0.01' \
    --reward_model pickscore

# 3. Run skin/person segmentation.
python skin_person_detection.py \
    --data_root 'M=sdxl-turbo$_$REG=True$_$REGW=0.01' \
    --reward_model pickscore
```

Repeat the same process with another reward flag and corresponding `--reward_model` name to evaluate the other reward objectives.

---

# Reproducibility notes

- `main_n_iter.py` runs every row in `prompt_sets/demographic_prompts_reno.csv`.
- Each prompt is generated five times using different randomly sampled seeds.
- The sampled seed is stored in `metadata.jsonl`.
- The initial and best images are stored as `init_image.png` and `best_image.png`.
- Optimization selects the image with the lowest reward loss encountered during the trajectory.
- The code uses one-step generation during optimization.
- Latent regularization is enabled by default; use `--disable_reg` to turn it off.
- The NSFW and skin-exposure scripts operate on the images already generated by Part A and do not regenerate images.

