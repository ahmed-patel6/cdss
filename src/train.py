"""
Training pipeline for the Clinical Decision Support System (CDSS).

This module is responsible for:

1. Loading the tokenizer.
2. Loading the pretrained BART model.
3. Loading the processed datasets.
4. Preparing all components required for training.
"""

from __future__ import annotations
import torch 
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
    SMOKE_TEST,
    SMOKE_TRAIN_SAMPLES,
    SMOKE_VALIDATION_SAMPLES,
    SMOKE_NUM_EPOCHS,
    GENERATION_MAX_LENGTH,
    PREDICT_WITH_GENERATE,
    NUM_BEAMS,
)

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.dataset import RadiologyDataset
from src.evaluate import compute_metrics

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

def prepare_training_subsets(
    train_dataset: RadiologyDataset,
    validation_dataset: RadiologyDataset,
) -> tuple[RadiologyDataset, RadiologyDataset]:
    """
    Optionally reduce the datasets for a smoke test.

    The smoke test allows us to verify the complete training
    pipeline without committing to a full training run.

    Parameters
    ----------
    train_dataset : RadiologyDataset
        Full training dataset.

    validation_dataset : RadiologyDataset
        Full validation dataset.

    Returns
    -------
    tuple[RadiologyDataset, RadiologyDataset]
        Training and validation datasets, either full-size or
        reduced for the smoke test.
    """

    if not SMOKE_TEST:
        return train_dataset, validation_dataset

    log_message(
        f"Smoke test enabled: using first "
        f"{SMOKE_TRAIN_SAMPLES} training samples."
    )

    log_message(
        f"Smoke test enabled: using first "
        f"{SMOKE_VALIDATION_SAMPLES} validation samples."
    )

    train_dataset.data = train_dataset.data.iloc[
        :SMOKE_TRAIN_SAMPLES
    ].reset_index(drop=True)

    validation_dataset.data = validation_dataset.data.iloc[
        :SMOKE_VALIDATION_SAMPLES
    ].reset_index(drop=True)

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

def create_training_arguments() -> Seq2SeqTrainingArguments:
    """
    Create the training configuration for sequence-to-sequence learning.

    Seq2SeqTrainingArguments extends the standard Hugging Face
    training configuration with generation-specific settings needed
    for tasks such as summarization.

    Returns
    -------
    Seq2SeqTrainingArguments
        Configuration used by Seq2SeqTrainer.
    """

    log_message("Creating training arguments...")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR),

        num_train_epochs=(
            SMOKE_NUM_EPOCHS
            if SMOKE_TEST
            else NUM_EPOCHS
        ),

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

        predict_with_generate=PREDICT_WITH_GENERATE,

        generation_max_length=GENERATION_MAX_LENGTH,

        generation_num_beams=NUM_BEAMS,

        fp16=FP16,

        report_to="none",
    )

    return training_args


def create_trainer(
    model: BartForConditionalGeneration,
    tokenizer: AutoTokenizer,
    train_dataset: RadiologyDataset,
    validation_dataset: RadiologyDataset,
    data_collator: DataCollatorForSeq2Seq,
    training_args: Seq2SeqTrainingArguments,
) -> Seq2SeqTrainer:
    """
    Create and configure the sequence-to-sequence Trainer.

    Parameters
    ----------
    model : BartForConditionalGeneration
        BART model being fine-tuned.

    tokenizer : AutoTokenizer
        Tokenizer associated with BART.

    train_dataset : RadiologyDataset
        Training dataset.

    validation_dataset : RadiologyDataset
        Validation dataset.

    data_collator : DataCollatorForSeq2Seq
        Batch construction and padding utility.

    training_args : Seq2SeqTrainingArguments
        Training and generation configuration.

    Returns
    -------
    Seq2SeqTrainer
        Configured sequence-to-sequence Trainer.
    """

    log_message("Creating Seq2SeqTrainer...")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer

def log_device() -> None:
    """Log the device that will be used for training."""

    if torch.cuda.is_available():
        log_message("CUDA is available.")
        log_message(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        log_message(
            f"CUDA version: {torch.version.cuda}"
        )
    else:
        log_message("CUDA is NOT available. Training will use CPU.")

def main() -> None:
    """
    Test loading all training components.
    """

    initialize_project()
    log_device()
    set_seed(RANDOM_SEED)

    tokenizer = load_tokenizer()

    model = load_model()

    train_dataset, validation_dataset = load_datasets(
        tokenizer
    )

    train_dataset, validation_dataset = prepare_training_subsets(
        train_dataset,
        validation_dataset,
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

    trainer = create_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    data_collator=data_collator,
    training_args=training_args,
    )
    log_message("Trainer created successfully.")

    log_message("Starting training...")
    trainer.train()
    log_message("Training completed successfully.")


if __name__ == "__main__":
    main()