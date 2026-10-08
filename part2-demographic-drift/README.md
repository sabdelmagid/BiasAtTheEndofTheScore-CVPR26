# Demographic Drift Under Reward-Guided Optimization

This directory contains the code for the **demographic drift experiments** in our paper.

The goal of this experiment is to test whether reward-guided optimization changes the perceived demographic attributes of generated subjects when those attributes are **not specified by the prompt**.

For example, we first find an initial noise tensor that generates an Asian female doctor when conditioned only on the underspecified prompt:

> `A photo of a doctor`

We then perform reward-guided optimization from that initialization and measure whether the perceived race or gender of the generated subject changes.

The full pipeline has three stages:

1. **Demographic initialization**
   - generate demographic target images;
   - optimize an initial noise tensor so that an underspecified occupation prompt generates an image visually similar to the desired demographic target set.

2. **Reward-guided optimization**
   - initialize ReNO from the learned noise tensor;
   - optimize the image using a selected reward model.

3. **Demographic measurement**
   - construct demographic anchor embeddings from the Chicago Face Database;
   - classify generated subjects before and after reward optimization;
   - merge the classification results for downstream analysis.

---

## Repository structure

```text
generate_target_images_flux.py
    Generate demographic target images used by the inversion objective.

main_inversion.py
    Learn demographic-conditioned initial noise tensors.

main_reno_with_init_inversion.py
    Run reward-guided ReNO optimization from the learned initialization.

prepare_classification_manifest.py
    Build inversion_outputs/data_to_classify.csv from the generated outputs.

precompute_embeddings.py
    Compute CLIP embeddings and demographic anchors from the
    Chicago Face Database.

classify_inversion_outputs.py
    Classify perceived race and gender using the demographic anchors.

merge_classification_chunks.py
    Merge parallel demographic-classification outputs.

arguments.py
    Shared command-line arguments.

models/
    Differentiable image-generation pipelines.

rewards/
    Reward functions and inversion objectives.

training/
    Latent optimization loop and optimizer utilities.

submit_all_learn_inversion.sh
    Submit inversion-learning jobs.

submit_all_reno_with_inversion.sh
    Submit reward-guided optimization jobs.

slurm/run_generate_target_images.sh
    Launcher for target-image generation.

slurm/run_learn_inversion.sh
    Slurm-array wrapper for inversion learning.

slurm/reno_with_init_inversion.sh
    Slurm-array wrapper for ReNO optimization.

slurm/run_classify_inversion_outputs.sh
    Slurm-array wrapper for demographic classification.
```

`main.py` is the general ReNO entrypoint inherited from the base implementation. The demographic-drift experiments use `main_inversion.py` and `main_reno_with_init_inversion.py`.

The active inversion objective is implemented in:

```text
rewards/inversionscore_centroid.py
```

Other inversion-score implementations are retained as experimental variants.

---

# Installation

Create the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate reno
```

The code downloads model weights from Hugging Face. If a model requires authentication, export your Hugging Face token in the shell:

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
```

Do not place authentication tokens directly in source files.

Model weights can be cached locally. For example:

```bash
export HF_HOME=/path/to/model_cache
```

Some classification utilities also support:

```bash
export MODEL_CACHE_DIR=/path/to/model_cache
```

The aesthetic reward expects the following checkpoint when that reward is used:

```text
$HF_HOME/ava+logos-l14-linearMSE.pth
```

---

# Part A: Learn Demographic-Conditioned Initial Noise

## A1. Generate demographic target images

The first step is to generate reference images representing combinations of race, gender, and occupation.

Run:

```bash
python generate_target_images_flux.py
```

or use the provided launcher:

```bash
bash slurm/run_generate_target_images.sh
```

The target-image generation setup covers:

- four race labels:
  - White
  - Black
  - Asian
  - Latino
- two gender labels:
  - female
  - male
- fifteen occupations
- three generated images per demographic/occupation combination

This produces 360 target images.

The expected output structure is:

```text
target_images_for_inversion_flux/
├── 00000.png
├── 00001.png
├── ...
└── metadata.csv
```

The metadata file contains:

```text
filename, prompt, race, gender, occupation, seed
```

These target images are used only as visual references for learning an appropriate initial noise tensor.

---

## A2. Learn the demographic-conditioned initialization

For each demographic and occupation combination, `main_inversion.py` optimizes the initial noise in two stages.

### Stage 1: demographic + occupation prompt

The first stage uses an explicitly demographic prompt such as:

```text
A photo of a female doctor
```

The latent is optimized to produce an image visually similar to the corresponding demographic target-image set.

### Stage 2: underspecified occupation prompt

Optimization then continues from the Stage-1 latent using the underspecified prompt:

```text
A photo of a doctor
```

The objective encourages the generated image to retain the desired visual demographic characteristics even though the demographic attribute is no longer present in the text prompt.

The final latent from Stage 2 is used as the initialization for reward-guided optimization in Part B.

A single inversion job can be submitted with:

```bash
sbatch slurm/run_learn_inversion.sh   --model sdxl-turbo   --seed 0
```

The full inversion experiment can be submitted with:

```bash
mkdir -p logs
bash submit_all_learn_inversion.sh
```

The submission script controls the evaluated seeds and base models.

Expected outputs follow a structure similar to:

```text
learned_inversion_outputs/
└── seed_<seed>/
    └── <model>/
        └── <prompt_hash>/
            ├── iter_0_<target_ids>/
            │   └── best_latents.pt
            │
            └── iter_1_<target_ids>/
                ├── best_latents.pt
                ├── init_image.png
                ├── best_image.png
                └── metadata.jsonl
```

The Stage-2 `best_latents.pt` file is the initialization used in Part B.

---

# Part B: Reward-Guided Optimization

`main_reno_with_init_inversion.py` loads the learned Stage-2 latent and performs reward-guided optimization using the underspecified occupation prompt.

For example:

```text
A photo of a doctor
```

The demographic attribute is therefore not explicitly specified during reward optimization.

A single run can be submitted with:

```bash
sbatch slurm/reno_with_init_inversion.sh   --seed 3   --enable_imagereward   --model pixart   --save_some_images
```

The full experiment matrix can be submitted with:

```bash
mkdir -p logs/renowinv
bash submit_all_reno_with_inversion.sh
```

The experiments evaluate the supported base models and reward objectives, including:

```text
--enable_pickscore
--enable_imagereward
--enable_hps
--enable_clip
--enable_aesthetic
--enable_incompression
```

Exactly one reward objective should normally be enabled for each run.

Positive and negative reward optimization are stored separately:

```text
inversion_outputs/
├── positive_optimization/
│   └── seed_<seed>/
│       └── <model>/
│           └── <reward>/
│               └── <occupation>/
│                   └── <prompt_hash>/
│                       └── iter_opt_<target_ids>/
│
└── negative_optimization/
    └── seed_<seed>/
        └── <model>/
            └── <reward>/
                └── <occupation>/
                    └── <prompt_hash>/
                        └── iter_opt_<target_ids>/
```

Runs contain files such as:

```text
init_image.png
best_image.png
metadata.jsonl
```

When `--save_some_images` is enabled, intermediate optimization images are also saved.

Negative optimization is enabled using:

```text
--set_weights_negative
```

which reverses the enabled reward weight.

---

# Part C: Measure Demographic Changes

Part C measures perceived race and gender before and after reward-guided optimization.

The pipeline consists of four steps:

```text
prepare_classification_manifest.py
        ↓
data_to_classify.csv
        ↓
classify_inversion_outputs.py
        ↓
classification_chunks/chunk_*.csv
        ↓
merge_classification_chunks.py
        ↓
data_classified.csv
```

Before classification, demographic anchor embeddings must also be constructed using `precompute_embeddings.py`.

---

## C1. Construct demographic anchors

The demographic classifier uses CLIP ViT-L/14 image embeddings derived from the **Chicago Face Database (CFD)**.

The CFD images are not included in this repository and must be obtained separately according to the dataset's access conditions.

Given a local CFD installation, run:

```bash
python precompute_embeddings.py   --cfd_dir "/path/to/CFD Version 3.0/Images/CFD"
```

An optional model-cache location can also be provided:

```bash
python precompute_embeddings.py   --cfd_dir "/path/to/CFD Version 3.0/Images/CFD"   --cache_dir "./model_cache"
```

For each CFD image, the script computes a normalized CLIP ViT-L/14 image embedding.

It then averages the embeddings within each demographic group and normalizes the resulting centroid.

The eight joint race × gender anchors are:

```text
AF.pt
AM.pt
BF.pt
BM.pt
LF.pt
LM.pt
WF.pt
WM.pt
```

where:

```text
A = Asian
B = Black
L = Latino
W = White

F = female
M = male
```

The script additionally creates marginal race and gender anchors.

The resulting directory is:

```text
<CFD directory>/
└── vitL14_embeddings/
    ├── <subject folders>/
    │   └── <image>.pt
    │
    └── anchors/
        ├── AF.pt
        ├── AM.pt
        ├── BF.pt
        ├── BM.pt
        ├── LF.pt
        ├── LM.pt
        ├── WF.pt
        ├── WM.pt
        ├── A.pt
        ├── B.pt
        ├── L.pt
        ├── W.pt
        ├── F.pt
        └── M.pt
```

For classification, point the classifier to the anchor directory:

```bash
export DEMOGRAPHIC_ANCHOR_DIR="/path/to/CFD/vitL14_embeddings/anchors"
```

---

## C2. Prepare the images for classification

Run:

```bash
python prepare_classification_manifest.py
```

By default, the script scans:

```text
./inversion_outputs
```

A different location can be provided with:

```bash
python prepare_classification_manifest.py   --base_dir /path/to/inversion_outputs
```

The script identifies the images used for the demographic-transition analysis and creates:

```text
inversion_outputs/data_to_classify.csv
```

For positive optimization, the selected images are:

```text
init_image.png
best_image.png
```

For negative optimization, the selected images are:

```text
init_image.png
15.png
```

The negative-optimization endpoint can be changed with:

```bash
--negative_best_step
```

For example:

```bash
python prepare_classification_manifest.py   --negative_best_step 15.png
```

Metadata from each run's `metadata.jsonl` is merged into the classification manifest.

---

## C3. Classify perceived race and gender

`classify_inversion_outputs.py` computes normalized CLIP ViT-L/14 embeddings for the generated images and assigns each image to the closest joint race × gender anchor by cosine similarity.

The input is:

```text
inversion_outputs/data_to_classify.csv
```

The CSV must contain an:

```text
image_path
```

column.

For a single-process run:

```bash
python classify_inversion_outputs.py 0 1
```

The two positional arguments are:

```text
<task_id> <num_tasks>
```

The classifier can therefore also be distributed across multiple jobs.

For example, with 20 tasks:

```bash
python classify_inversion_outputs.py "$TASK_ID" 20
```

The provided Slurm wrapper can run the classification in parallel:

```bash
mkdir -p logs_classify
sbatch slurm/run_classify_inversion_outputs.sh
```

Each task writes:

```text
inversion_outputs/
└── classification_chunks/
    ├── chunk_0000.csv
    ├── chunk_0001.csv
    └── ...
```

Each output row contains:

```text
pred_race
pred_gender
```

as well as the cosine similarity to each joint demographic anchor.

---

## C4. Merge classification outputs

Once all classification tasks have completed, run:

```bash
python merge_classification_chunks.py
```

By default, the script reads:

```text
inversion_outputs/classification_chunks/chunk_*.csv
```

and writes:

```text
inversion_outputs/data_classified.csv
```

For a non-default output directory:

```bash
python merge_classification_chunks.py   --base_dir /path/to/inversion_outputs
```

The script restores the original row ordering from `data_to_classify.csv` before saving the merged result.

The resulting `data_classified.csv` is the primary output used to measure demographic transitions before and after reward-guided optimization.

---

# End-to-End Workflow

The complete experiment can be summarized as:

```text
1. Generate demographic target images
        │
        ▼
generate_target_images_flux.py
        │
        ▼
target_images_for_inversion_flux/
        │
        ▼
2. Learn demographic-conditioned initial noise
        │
        ▼
main_inversion.py
        │
        ▼
learned_inversion_outputs/
        │
        ▼
3. Run reward-guided optimization
        │
        ▼
main_reno_with_init_inversion.py
        │
        ▼
inversion_outputs/
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
4a. Build CFD anchors       4b. Prepare classification manifest
precompute_embeddings.py    prepare_classification_manifest.py
        │                             │
        ▼                             ▼
demographic anchors         data_to_classify.csv
        │                             │
        └──────────────┬──────────────┘
                       ▼
             5. Classify demographics
             classify_inversion_outputs.py
                       │
                       ▼
             classification_chunks/
                       │
                       ▼
             6. Merge classification
             merge_classification_chunks.py
                       │
                       ▼
             data_classified.csv
```

---

# Reproducibility notes

- Target-image generation samples random seeds at runtime. The sampled seed for every generated target is stored in `metadata.csv`.
- Inversion outputs are indexed by experiment seed, base model, prompt hash, and target-image IDs.
- Reward-guided optimization in Part B must use the corresponding seed and base model from Part A so that the correct learned initialization can be loaded.
- Reward implementations are formulated as losses that are minimized. As a result, some logged loss values have the opposite numerical direction from the underlying reward score.
- Demographic classification uses normalized CLIP ViT-L/14 embeddings and nearest-anchor cosine similarity.
- The demographic labels assigned by the classifier represent **perceived attributes of generated image subjects**, rather than self-identified demographic information.
- The Chicago Face Database is not redistributed with this repository.

---

# External data

This experiment requires the Chicago Face Database for construction of the demographic anchor embeddings.

Users should obtain the dataset directly from its official distribution source and comply with its license and terms of use.

The repo does not redistribute CFD images.

