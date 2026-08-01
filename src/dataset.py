"""
PyTorch Dataset for Radiology Report Summarization.

This module converts processed CSV files into a Dataset
that can be used by Hugging Face's Trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import (
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

    def tokenize_input(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize the Findings section.
        """

        return self.tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def tokenize_target(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize the Impression section.
        """

        return self.tokenizer(
            text,
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Return one training sample.
        """

        findings = self.data.loc[idx, "findings"]

        impression = self.data.loc[idx, "impression"]

        model_inputs = self.tokenize_input(findings)

        labels = self.tokenize_target(impression)

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }