"""
Evaluation utilities for the Clinical Decision Support System (CDSS).

This module provides reusable functions to evaluate the performance
of sequence-to-sequence models using ROUGE metrics.

Responsibilities
----------------
1. Decode model predictions.
2. Decode reference summaries.
3. Compute ROUGE scores.
4. Return metrics in a format expected by Hugging Face Trainer.
"""

from __future__ import annotations

from typing import Tuple

import evaluate
import numpy as np
from transformers import AutoTokenizer

from src.config import MODEL_NAME

# ------------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ------------------------------------------------------------------
# Load ROUGE metric
# ------------------------------------------------------------------

rouge = evaluate.load("rouge")


def postprocess_text(
    predictions: list[str],
    references: list[str],
) -> Tuple[list[str], list[str]]:
    """
    Remove unnecessary whitespace from predictions and references.

    Parameters
    ----------
    predictions : list[str]
        Generated summaries.

    references : list[str]
        Ground truth summaries.

    Returns
    -------
    tuple[list[str], list[str]]
        Cleaned predictions and references.
    """

    predictions = [prediction.strip() for prediction in predictions]
    references = [reference.strip() for reference in references]

    return predictions, references


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """
    Compute ROUGE evaluation metrics.

    This function is automatically called by the Hugging Face Trainer
    during evaluation.

    Parameters
    ----------
    eval_pred : tuple[np.ndarray, np.ndarray]
        Model predictions and labels.

    Returns
    -------
    dict[str, float]
        Dictionary containing ROUGE scores.
    """

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # --------------------------------------------------------------
    # Replace ignored label values (-100) with pad token id
    # --------------------------------------------------------------

    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )

    # --------------------------------------------------------------
    # Decode predictions
    # --------------------------------------------------------------

    decoded_predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )

    # --------------------------------------------------------------
    # Decode labels
    # --------------------------------------------------------------

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    # --------------------------------------------------------------
    # Clean text
    # --------------------------------------------------------------

    decoded_predictions, decoded_labels = postprocess_text(
        decoded_predictions,
        decoded_labels,
    )

    # --------------------------------------------------------------
    # Compute ROUGE
    # --------------------------------------------------------------

    scores = rouge.compute(
        predictions=decoded_predictions,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "rouge1": round(scores["rouge1"] * 100, 2),
        "rouge2": round(scores["rouge2"] * 100, 2),
        "rougeL": round(scores["rougeL"] * 100, 2),
    }