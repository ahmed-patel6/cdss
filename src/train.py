"""
Training pipeline for the Clinical Decision Support System (CDSS).

This module is responsible for:

1. Loading the tokenizer.
2. Loading the pretrained BART model.
3. Loading the processed datasets.
4. Preparing all components required for training.
"""

from __future__ import annotations

from src.config import (
    MODEL_NAME,
    TRAIN_FILE,
    VALIDATION_FILE,
    OUTPUT_MODEL_DIR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    SAVE_STRATEGY,
    EVAL_STRATEGY,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END,
    METRIC_FOR_BEST_MODEL,
    GREATER_IS_BETTER,
    RANDOM_SEED,
    GRADIENT_ACCUMULATION_STEPS,
    FP16,
)

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)

from src.dataset import RadiologyDataset

from src.utils import (
    initialize_project,
    log_message,
    set_seed,
)


def load_tokenizer() -> AutoTokenizer:
    """
    Load the pretrained tokenizer.

    Returns
    -------
    AutoTokenizer
        Hugging Face tokenizer.
    """

    log_message("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return tokenizer


def load_model() -> BartForConditionalGeneration:
    """
    Load the pretrained BART model.

    Returns
    -------
    BartForConditionalGeneration
        Pretrained BART model.
    """

    log_message("Loading model...")

    model = BartForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    return model


def load_datasets(
    tokenizer: AutoTokenizer,
) -> tuple[RadiologyDataset, RadiologyDataset]:
    """
    Load the processed train and validation datasets.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Tokenizer used for preprocessing.

    Returns
    -------
    tuple
        Training dataset and validation dataset.
    """

    log_message("Loading datasets...")

    train_dataset = RadiologyDataset(
        csv_file=TRAIN_FILE,
        tokenizer=tokenizer,
    )

    validation_dataset = RadiologyDataset(
        csv_file=VALIDATION_FILE,
        tokenizer=tokenizer,
    )

    return train_dataset, validation_dataset

def create_data_collator(
    tokenizer: AutoTokenizer,
    model: BartForConditionalGeneration,
) -> DataCollatorForSeq2Seq:
    """
    Create a sequence-to-sequence data collator.

    The collator dynamically creates batches and handles
    padding for encoder-decoder models.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Hugging Face tokenizer.

    model : BartForConditionalGeneration
        BART model.

    Returns
    -------
    DataCollatorForSeq2Seq
        Data collator for training.
    """

    log_message("Creating data collator...")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    return data_collator

def create_training_arguments() -> TrainingArguments:
    """
    Create Hugging Face training configuration.

    Returns
    -------
    TrainingArguments
        Configuration used by Trainer.
    """

    log_message("Creating training arguments...")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR),

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        weight_decay=WEIGHT_DECAY,

        logging_steps=LOGGING_STEPS,

        eval_strategy=EVAL_STRATEGY,

        save_strategy=SAVE_STRATEGY,

        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,

        metric_for_best_model=METRIC_FOR_BEST_MODEL,

        greater_is_better=GREATER_IS_BETTER,

        fp16=FP16,

        report_to="none",
    )

    return training_args

def main() -> None:
    """
    Test loading all training components.
    """

    initialize_project()

    set_seed(RANDOM_SEED)

    tokenizer = load_tokenizer()

    model = load_model()

    train_dataset, validation_dataset = load_datasets(
        tokenizer
    )

    data_collator = create_data_collator(
    tokenizer,
    model,
    )

    training_args = create_training_arguments()

    log_message("Training arguments created successfully.")

    log_message(f"Train samples: {len(train_dataset)}")

    log_message(
        f"Validation samples: {len(validation_dataset)}"
    )

    log_message("Training setup completed successfully.")


if __name__ == "__main__":
    main()