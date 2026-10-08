"""
Create the demographic-classification manifest used for Part 2 demographic drift.

This script scans the ReNO inversion outputs, keeps the initial and final images
used in the demographic-transition analysis, merges run metadata, and writes:

    inversion_outputs/data_to_classify.csv

Expected image hierarchy:

    inversion_outputs/
        <optimization_type>/
        <seed>/
        <base_model>/
        <reward_model>/
        <occupation>/
        <folder_hash>/
        <iter_dir>/
        <image>.png

For positive optimization, the analysis uses:
    - init_image.png
    - best_image.png

For negative optimization, the analysis uses:
    - init_image.png
    - 15.png

Usage:
    python prepare_classification_manifest.py

Optional:
    python prepare_classification_manifest.py \
        --base_dir ./inversion_outputs \
        --negative_best_step 15.png
"""

import argparse
import glob
import json
import os

import pandas as pd


def parse_image_path(path):
    """Parse experiment metadata encoded in an inversion-output image path."""
    parts = os.path.normpath(path).split(os.sep)

    return {
        "opt_type": parts[-8],
        "seed": parts[-7],
        "base_model": parts[-6],
        "reward_model": parts[-5].lstrip("_"),
        "occupation": parts[-4],
        "folder_hash": parts[-3],
        "iter_dir": parts[-2],
        "image_name": parts[-1],
        "image_path": path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create inversion_outputs/data_to_classify.csv."
    )
    parser.add_argument(
        "--base_dir",
        default=os.environ.get("INVERSION_OUTPUTS_DIR", "./inversion_outputs"),
        help="Root directory containing the inversion outputs.",
    )
    parser.add_argument(
        "--negative_best_step",
        default="15.png",
        help="Final image filename used for negative optimization.",
    )
    args = parser.parse_args()

    image_paths = glob.glob(
        os.path.join(
            args.base_dir,
            "*",
            "*",
            "*",
            "*",
            "*",
            "*",
            "*",
            "*.png",
        )
    )
    print(f"Found {len(image_paths)} images")

    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {args.base_dir}. "
            "Check the inversion-output directory structure."
        )

    df = pd.DataFrame([parse_image_path(path) for path in image_paths])

    positive_mask = (
        (df["opt_type"] == "positive_optimization")
        & df["image_name"].isin(["best_image.png", "init_image.png"])
    )

    negative_mask = (
        (df["opt_type"] == "negative_optimization")
        & df["image_name"].isin(
            [args.negative_best_step, "init_image.png"]
        )
    )

    df_filtered = df[positive_mask | negative_mask].reset_index(drop=True)

    print(
        f"Keeping {len(df_filtered)} images used in the "
        "demographic-transition analysis"
    )

    if df_filtered.empty:
        raise RuntimeError(
            "No matching positive/negative optimization images were found."
        )

    # Load one metadata.jsonl file for each experiment hash.
    hash_to_path = (
        df_filtered.groupby("folder_hash")["image_path"]
        .first()
        .apply(
            lambda path: os.path.join(
                os.path.dirname(path),
                "metadata.jsonl",
            )
        )
        .to_dict()
    )

    print(f"Loading metadata for {len(hash_to_path)} unique hashes")

    hash_to_metadata = {}

    for folder_hash, metadata_path in hash_to_path.items():
        with open(metadata_path, "r") as f:
            metadata = json.loads(f.readline())

        # These fields are either already represented in the table
        # or contain run-specific filesystem information.
        metadata.pop("seed", None)
        metadata.pop("init_latents_path", None)
        metadata.pop("save_dir", None)

        hash_to_metadata[folder_hash] = metadata

    metadata_df = pd.DataFrame.from_dict(
        hash_to_metadata,
        orient="index",
    )

    if "folder_hash" in metadata_df.columns:
        metadata_df = metadata_df.drop(columns=["folder_hash"])

    metadata_df.index.name = "folder_hash"
    metadata_df = metadata_df.reset_index()

    duplicate_columns = [
        column
        for column in metadata_df.columns
        if column in df_filtered.columns and column != "folder_hash"
    ]

    df_filtered = df_filtered.merge(
        metadata_df.drop(columns=duplicate_columns),
        on="folder_hash",
        how="left",
    )

    output_path = os.path.join(
        args.base_dir,
        "data_to_classify.csv",
    )

    # Keep the dataframe index because classify_inversion_outputs.py uses
    # it to restore the original ordering after parallel classification.
    df_filtered.to_csv(output_path)

    print(f"Saved {len(df_filtered)} rows to {output_path}")


if __name__ == "__main__":
    main()
