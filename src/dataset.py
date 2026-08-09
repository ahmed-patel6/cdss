"""
PyTorch Dataset for Radiology Report Summarization.

This module converts processed CSV files into a Dataset
that can be used by Hugging Face's Trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)


class RadiologyDataset(Dataset):
    """
    Dataset for radiology report summarization.

    Parameters
    ----------
    csv_file : Path
        Path to the processed CSV.

    tokenizer : AutoTokenizer, optional
        Hugging Face tokenizer.
        If None, the tokenizer is loaded automatically.
    """

    def __init__(
        self,
        csv_file: Path,
        tokenizer: AutoTokenizer | None = None,
    ) -> None:

        self.data = pd.read_csv(csv_file)

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(MODEL_NAME)
        )

    def __len__(self) -> int:
        """
        Return the total number of samples.
        """

        return len(self.data)

    def tokenize_input(self, text: str) -> Dict[str, list[int]]:
        """
        Tokenize the Findings section.

        The tokenizer returns Python lists rather than PyTorch tensors.
        The data collator will convert these lists into batched tensors.
        """

        return self.tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
        )

    def tokenize_target(self, text: str) -> Dict[str, list[int]]:
        """
        Tokenize the Impression section.

        The tokenizer returns Python lists rather than PyTorch tensors.
        The data collator will create the final tensors.
        """

        return self.tokenizer(
            text,
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, list[int]]:
        """
        Return one tokenized training sample.

        Parameters
        ----------
        idx : int
            Index of the requested sample.

        Returns
        -------
        Dict[str, list[int]]
            Tokenized input and target sequences.
        """

        findings = self.data.loc[idx, "findings"]

        impression = self.data.loc[idx, "impression"]

        model_inputs = self.tokenize_input(findings)

        labels = self.tokenize_target(impression)

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"],
        }