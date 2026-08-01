"""
Data preprocessing pipeline for the Clinical Decision Support System (CDSS).

This module:
1. Downloads the dataset from Hugging Face.
2. Cleans the radiology reports.
3. Removes invalid samples.
4. Saves processed CSV files.
"""

from __future__ import annotations

import re

import pandas as pd
from datasets import DatasetDict, load_dataset

from config import (
    DATASET_NAME,
    PROCESSED_DATA_DIR,
)
import dataset
from utils import create_directories, log_message


def clean_text(text: str) -> str:
    """
    Clean a text string by removing unnecessary whitespace.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess one dataset split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Clean dataframe.
    """

    # Keep only required columns
    df = df[["findings", "impression"]].copy()

    # Remove missing rows
    df = df.dropna()

    # Clean text
    df["findings"] = df["findings"].apply(clean_text)
    df["impression"] = df["impression"].apply(clean_text)

    # Remove empty strings
    df = df[
        (df["findings"] != "")
        &
        (df["impression"] != "")
    ]

    return df.reset_index(drop=True)


def save_split(df: pd.DataFrame, filename: str) -> None:
    """
    Save a processed dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.

    filename : str
        Output filename.
    """

    output_path = PROCESSED_DATA_DIR / filename

    df.to_csv(output_path, index=False)

    log_message(f"Saved {filename}")


def main() -> None:
    """
    Run the preprocessing pipeline.
    """

    create_directories([PROCESSED_DATA_DIR])

    log_message("Loading dataset...")

    dataset: DatasetDict = load_dataset(DATASET_NAME)

    for split_name in dataset.keys():
        log_message(f"Processing {split_name} split...")

        df = preprocess_split(dataset[split_name].to_pandas())

        save_split(df, f"{split_name}.csv")

    log_message("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()