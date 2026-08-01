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

# ==========================================================
# Processed Dataset Files
# ==========================================================

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"

VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.csv"

TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model directory
MODELS_DIR = BASE_DIR / "models"

# Directory where the trained BART model and checkpoints will be stored
OUTPUT_MODEL_DIR = MODELS_DIR / "bart-base"

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

MODEL_NAME = "facebook/bart-base"

# ==========================================================
# Tokenization
# ==========================================================

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# ==========================================================
# Training Hyperparameters
# ==========================================================

# ==========================================================
# Training Hyperparameters
# ==========================================================

TRAIN_BATCH_SIZE = 4

EVAL_BATCH_SIZE = 4

LEARNING_RATE = 5e-5

NUM_EPOCHS = 3

GRADIENT_ACCUMULATION_STEPS = 2

FP16 = True

# ==========================================================
# Trainer Configuration
# ==========================================================

WEIGHT_DECAY = 0.01

LOGGING_STEPS = 100

SAVE_STRATEGY = "epoch"

EVAL_STRATEGY = "epoch"

SAVE_TOTAL_LIMIT = 2

LOAD_BEST_MODEL_AT_END = True

PREDICT_WITH_GENERATE = True

METRIC_FOR_BEST_MODEL = "rougeL"

GREATER_IS_BETTER = True

RANDOM_SEED = 42


# ==========================================================
# Generation Configuration
# ==========================================================

NUM_BEAMS = 4
EARLY_STOPPING = True


LOGS_DIR = BASE_DIR / "logs"


