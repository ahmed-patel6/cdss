"""
Utility functions for the Clinical Decision Support System (CDSS).

This module contains reusable helper functions that can be used
throughout the project.

Functions:
    - set_seed()
    - create_directories()
    - log_message()
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_directories(paths: Iterable[Path]) -> None:
    """
    Create directories if they do not already exist.

    Parameters
    ----------
    paths : Iterable[Path]
        Collection of directory paths.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def log_message(message: str) -> None:
    """
    Print a formatted log message.

    Parameters
    ----------
    message : str
        Message to display.
    """
    print(f"[INFO] {message}")


def initialize_project() -> None:
    """
    Initialize the project by creating required directories.
    """
    create_directories(
        [
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            MODELS_DIR,
            OUTPUTS_DIR,
        ]
    )

    log_message("Project directories initialized.")