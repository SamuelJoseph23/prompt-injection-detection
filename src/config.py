"""
Configuration for Ghost in the Machine
=======================================
Centralized dataclass holding all hyperparameters, file paths,
and reproducibility seeds.  Optionally loads overrides from a
YAML file (``config.yaml`` in the project root).

Usage::

    from src.config import Config
    cfg = Config()                       # defaults
    cfg = Config.from_yaml("config.yaml")  # override from file
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Return the project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """All tuneable parameters in one place."""

    # --- Model ---
    model_name: str = "bert-base-uncased"
    max_seq_length: int = 128
    embedding_dim: int = 256
    dropout: float = 0.1
    freeze_encoder_layers: int = 0  # freeze first N transformer layers

    # --- Training ---
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_ratio: float = 0.1
    margin: float = 1.0  # contrastive loss margin
    patience: int = 3  # early stopping patience

    # --- Data ---
    train_file: str = "data/processed/train.csv"
    val_file: str = "data/processed/val.csv"
    test_file: str = "data/processed/test.csv"
    zeroday_file: str = "data/processed/test_zeroday.csv"

    # --- Outputs ---
    model_dir: str = "models"
    results_dir: str = "results"
    log_dir: str = "results/tensorboard"

    # --- Reproducibility ---
    seed: int = 42

    # --- Quick mode (for smoke tests) ---
    quick: bool = False
    quick_samples: int = 200
    quick_epochs: int = 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return _project_root()

    def abs(self, rel_path: str) -> Path:
        """Resolve a project-relative path to absolute."""
        return self.project_root / rel_path

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Create a Config by overlaying values from a YAML file."""
        full_path = _project_root() / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            print(f"  Config file {full_path} not found -- using defaults.")
            return cls()

        with open(full_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        flat: dict[str, Any] = {}
        for section in data.values():
            if isinstance(section, dict):
                flat.update(section)
            # top-level scalars are also accepted

        # Only keep keys that match dataclass fields and cast types
        fields = cls.__dataclass_fields__
        filtered: dict[str, Any] = {}
        for k, v in flat.items():
            if k not in fields:
                continue
            expected_type = fields[k].type
            # Coerce to the expected type when possible
            if expected_type == "float" and isinstance(v, str):
                v = float(v)
            elif expected_type == "int" and isinstance(v, str):
                v = int(v)
            elif expected_type == "bool" and isinstance(v, str):
                v = v.lower() in ("true", "1", "yes")
            filtered[k] = v
        return cls(**filtered)

