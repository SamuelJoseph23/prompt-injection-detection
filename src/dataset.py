"""
PyTorch Datasets for Ghost in the Machine
==========================================
``PromptDataset``      – standard single-text dataset for the baseline classifier.
``SiamesePairDataset`` – generates positive/negative text pairs for contrastive
                         learning with the Siamese network.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PromptDataset(Dataset):
    """Standard dataset returning tokenized text + binary label.

    Used for the baseline BERT classifier.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SiamesePairDataset(Dataset):
    """Generates pairs of texts for Siamese contrastive learning.

    Each item returns two tokenized texts and a pair label:
      - pair_label = 1  ->  same class  (both benign or both malicious)
      - pair_label = 0  ->  different class

    The dataset oversamples malicious texts when forming pairs to
    counteract class imbalance.

    Args:
        texts: All text samples.
        labels: Binary labels (0=benign, 1=malicious).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token sequence length.
        num_pairs: Total number of pairs to generate per epoch.
                   Defaults to ``2 * len(texts)``.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        num_pairs: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Group indices by class
        self.class_indices: dict[int, list[int]] = {0: [], 1: []}
        for i, lab in enumerate(labels):
            self.class_indices[int(lab)].append(i)

        self.texts = texts
        self.labels = labels
        self.num_pairs = num_pairs or 2 * len(texts)

        # Pre-generate pairs for this epoch
        self._generate_pairs()

    def _generate_pairs(self) -> None:
        """Create a balanced set of positive and negative pairs."""
        pairs: list[tuple[int, int, int]] = []
        half = self.num_pairs // 2

        # Positive pairs (same class)
        for _ in range(half):
            cls = random.choice([0, 1])
            idxs = self.class_indices[cls]
            if len(idxs) < 2:
                cls = 1 - cls
                idxs = self.class_indices[cls]
            i, j = random.sample(idxs, 2)
            pairs.append((i, j, 1))

        # Negative pairs (different class)
        for _ in range(half):
            i = random.choice(self.class_indices[0])
            j = random.choice(self.class_indices[1])
            pairs.append((i, j, 0))

        random.shuffle(pairs)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i, j, pair_label = self.pairs[idx]

        enc1 = self.tokenizer(
            self.texts[i],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            self.texts[j],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids_1": enc1["input_ids"].squeeze(0),
            "attention_mask_1": enc1["attention_mask"].squeeze(0),
            "input_ids_2": enc2["input_ids"].squeeze(0),
            "attention_mask_2": enc2["attention_mask"].squeeze(0),
            "pair_label": torch.tensor(pair_label, dtype=torch.float),
            "label_1": torch.tensor(self.labels[i], dtype=torch.long),
            "label_2": torch.tensor(self.labels[j], dtype=torch.long),
        }


def load_data_from_csv(path: str) -> tuple[list[str], list[int]]:
    """Load texts and labels from a processed CSV file."""
    df = pd.read_csv(path)
    texts = df["text"].astype(str).tolist()
    labels = [int(v) for v in df["label"]]
    return texts, labels
