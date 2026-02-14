"""
Utility Functions for Ghost in the Machine
==========================================
Seed management, metric computation, visualization helpers,
device detection, and model checkpoint management.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  Using CPU")
    return device


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | list | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary of metric name -> value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc"] = 0.0

    return metrics


def get_classification_report(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
) -> str:
    """Return a formatted sklearn classification report."""
    return classification_report(
        y_true,
        y_pred,
        target_names=["benign", "malicious"],
        zero_division=0,
    )


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malicious"])
    ax.set_yticklabels(["Benign", "Malicious"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    fig.colorbar(im)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """Plot and save an ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curve -> {save_path}")


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    # Accuracy
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="Val Accuracy", color="green")
    if "val_f1" in history:
        axes[1].plot(history["val_f1"], label="Val F1", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved training history -> {save_path}")


# ------------------------------------------------------------------
# Model checkpointing
# ------------------------------------------------------------------
def save_model(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model state dict and optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict()}
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)
    print(f"  Model saved -> {path}")


def load_model(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load model state dict from checkpoint. Returns metadata if present."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Model loaded <- {path}")
    return checkpoint.get("metadata", {})
