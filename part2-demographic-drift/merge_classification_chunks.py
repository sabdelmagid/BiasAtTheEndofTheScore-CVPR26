"""
Merge demographic-classification chunks produced by classify_inversion_outputs.py.

Input:
    inversion_outputs/classification_chunks/chunk_*.csv

Output:
    inversion_outputs/data_classified.csv

Usage:
    python merge_classification_chunks.py

Optional:
    python merge_classification_chunks.py \
        --base_dir ./inversion_outputs
"""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge classification_chunks/chunk_*.csv."
    )
    parser.add_argument(
        "--base_dir",
        default=os.environ.get("INVERSION_OUTPUTS_DIR", "./inversion_outputs"),
        help="Root directory containing classification_chunks.",
    )
    args = parser.parse_args()

    chunk_dir = os.path.join(
        args.base_dir,
        "classification_chunks",
    )

    chunk_paths = sorted(
        glob.glob(
            os.path.join(chunk_dir, "chunk_*.csv")
        )
    )

    print(f"Found {len(chunk_paths)} classification chunks")

    if not chunk_paths:
        raise FileNotFoundError(
            f"No chunk_*.csv files found in {chunk_dir}. "
            "Run classify_inversion_outputs.py first."
        )

    df_results = pd.concat(
        [pd.read_csv(path) for path in chunk_paths],
        ignore_index=True,
    )

    if "index" not in df_results.columns:
        raise KeyError(
            "Expected an 'index' column in the classification chunks. "
            "This column is written by classify_inversion_outputs.py "
            "and is used to restore the original row order."
        )

    df_results = (
        df_results
        .sort_values("index")
        .reset_index(drop=True)
    )

    output_path = os.path.join(
        args.base_dir,
        "data_classified.csv",
    )

    df_results.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {len(df_results)} rows to {output_path}")


if __name__ == "__main__":
    main()
