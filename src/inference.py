from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MODEL_PATH = "models/bart-base/checkpoint-68658"

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------------

print("[INFO] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

print("[INFO] Loading model...")

model = BartForConditionalGeneration.from_pretrained(
    MODEL_PATH
)

model = model.to(device)
model.eval()

print(f"[INFO] Device: {device}")

if torch.cuda.is_available():
    print(
        f"[INFO] GPU: {torch.cuda.get_device_name(0)}"
    )


# ------------------------------------------------------------------
# Generate summary
# ------------------------------------------------------------------

def generate_impression(
    findings: str,
    max_length: int = 128,
    num_beams: int = 4,
) -> str:
    """
    Generate a radiology impression from findings.

    Parameters
    ----------
    findings : str
        Radiology findings text.

    max_length : int
        Maximum number of generated tokens.

    num_beams : int
        Number of beams used during beam search.

    Returns
    -------
    str
        Generated radiology impression.
    """

    inputs = tokenizer(
        findings,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    impression = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return impression.strip()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    """
    Run a simple inference test.
    """

    findings = """
    The heart is mildly enlarged. No focal airspace
    consolidation, pleural effusion, or pneumothorax.
    Mild bibasilar linear atelectatic changes.
    """

    print("\n" + "=" * 70)
    print("RADIOLOGY SUMMARIZATION")
    print("=" * 70)

    print("\nFINDINGS:")
    print(findings.strip())

    impression = generate_impression(findings)

    print("\nGENERATED IMPRESSION:")
    print(impression, "\n")


if __name__ == "__main__":
    main()