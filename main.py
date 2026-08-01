from pathlib import Path

from dataset import RadiologyDataset

dataset = RadiologyDataset(
    Path("data/processed/train.csv")
)

print(f"Dataset size: {len(dataset)}")

sample = dataset[0]

print(sample.keys())

print(sample["input_ids"].shape)

print(sample["attention_mask"].shape)

print(sample["labels"].shape)