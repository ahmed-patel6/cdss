"""
Central configuration file for the Clinical Decision Support System (CDSS)
Week 2: Radiology Report Summarization.

This module stores all configurable values such as:
- File paths
- Model configuration
- Training hyperparameters

Keeping these values in one place makes the project easier to
maintain and modify.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = BASE_DIR / "models"

# Output directory
OUTPUTS_DIR = BASE_DIR / "outputs"

# Notebook directory
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# ==========================================================
# Dataset Configuration
# ==========================================================

DATASET_NAME = "tgrex6/mimic-cxr-reports-summarization"

# ==========================================================
# Model Configuration
# ==========================================================

MODEL_NAME = "google/flan-t5-base"

# ==========================================================
# Tokenization
# ==========================================================

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# ==========================================================
# Training Hyperparameters
# ==========================================================

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

LEARNING_RATE = 5e-5

NUM_EPOCHS = 3

RANDOM_SEED = 42

# ==========================================================
# Generation Configuration
# ==========================================================

NUM_BEAMS = 4
EARLY_STOPPING = True