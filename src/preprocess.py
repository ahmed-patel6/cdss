"""
Data preprocessing pipeline for the Clinical Decision Support System (CDSS).

This module:
1. Downloads the dataset from Hugging Face.
2. Cleans the radiology reports.
3. Removes invalid samples.
4. Splits the training set into train and test (90:10).
5. Saves processed CSV files.
"""

from __future__ import annotations

import re

import pandas as pd
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from src.utils import create_directories, log_message


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

    df = df[["findings", "impression"]].copy()

    df = df.dropna()

    df["findings"] = df["findings"].apply(clean_text)
    df["impression"] = df["impression"].apply(clean_text)

    df = df[
        (df["findings"] != "")
        & (df["impression"] != "")
    ]

    return df.reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the processed training data into
    train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Clean training dataframe.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train dataframe, Test dataframe.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split(
    df: pd.DataFrame,
    filename: str,
) -> None:
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

    # -------------------------------
    # Process Training Split
    # -------------------------------

    log_message("Processing train split...")

    train_df = preprocess_split(
        dataset["train"].to_pandas()
    )

    train_df, test_df = split_train_test(train_df)

    save_split(train_df, "train.csv")
    save_split(test_df, "test.csv")

    # -------------------------------
    # Process Validation Split
    # -------------------------------

    log_message("Processing validation split...")

    validation_df = preprocess_split(
        dataset["validation"].to_pandas()
    )

    save_split(validation_df, "validation.csv")

    log_message("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()