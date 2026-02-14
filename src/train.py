"""
Training Pipeline for Ghost in the Machine
============================================
Trains the Siamese prompt injection detector and an optional
baseline BERT classifier.  Supports early stopping, model
checkpointing, and TensorBoard logging.

Usage::

    # Full training
    python src/train.py

    # Quick smoke test (1 epoch, subset of data)
    python src/train.py --quick

    # With custom config
    python src/train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import PromptDataset, SiamesePairDataset, load_data_from_csv
from src.model import BaselineClassifier, ContrastiveLoss, SiamesePromptDetector
from src.utils import (
    compute_metrics,
    get_device,
    plot_training_history,
    save_model,
    set_seed,
)


# ======================================================================
# Training helpers
# ======================================================================
def train_siamese_epoch(
    model: SiamesePromptDetector,
    dataloader: DataLoader,
    criterion: ContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
) -> float:
    """Train one epoch of the Siamese model. Returns average loss."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        ids1 = batch["input_ids_1"].to(device)
        mask1 = batch["attention_mask_1"].to(device)
        ids2 = batch["input_ids_2"].to(device)
        mask2 = batch["attention_mask_2"].to(device)
        pair_label = batch["pair_label"].to(device)

        optimizer.zero_grad()
        emb1, emb2 = model(ids1, mask1, ids2, mask2)
        loss = criterion(emb1, emb2, pair_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_siamese(
    model: SiamesePromptDetector,
    dataloader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate Siamese model — returns contrastive loss and classification metrics."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in dataloader:
        ids1 = batch["input_ids_1"].to(device)
        mask1 = batch["attention_mask_1"].to(device)
        ids2 = batch["input_ids_2"].to(device)
        mask2 = batch["attention_mask_2"].to(device)
        pair_label = batch["pair_label"].to(device)

        emb1, emb2 = model(ids1, mask1, ids2, mask2)
        loss = criterion(emb1, emb2, pair_label)
        total_loss += loss.item()

        # Also evaluate the classification head on text1
        logits = model.classify(ids1, mask1)
        preds = logits.argmax(dim=1).cpu().tolist()
        labels = batch["label_1"].tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)

    metrics = compute_metrics(all_labels, all_preds)
    metrics["contrastive_loss"] = total_loss / max(len(dataloader), 1)
    return metrics


# ======================================================================
# Baseline training
# ======================================================================
def train_baseline_epoch(
    model: BaselineClassifier,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
) -> float:
    """Train one epoch of the baseline classifier."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_baseline(
    model: BaselineClassifier,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate baseline classifier."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(ids, mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


# ======================================================================
# Main training loop
# ======================================================================
def train(cfg: Config) -> None:
    """Full training pipeline."""
    set_seed(cfg.seed)
    device = get_device()

    # --- Load data ---
    print("\n[1/6] Loading data...")
    train_texts, train_labels = load_data_from_csv(cfg.abs(cfg.train_file))
    val_texts, val_labels = load_data_from_csv(cfg.abs(cfg.val_file))

    if cfg.quick:
        n = min(cfg.quick_samples, len(train_texts))
        train_texts, train_labels = train_texts[:n], train_labels[:n]
        val_texts, val_labels = val_texts[:n // 2], val_labels[:n // 2]
        print(f"  [Quick mode] Using {n} train, {n // 2} val samples")

    print(f"  Train: {len(train_texts)} | Val: {len(val_texts)}")

    # --- Tokenizer ---
    print("\n[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --- Datasets & loaders ---
    print("\n[3/6] Creating datasets...")
    siamese_train = SiamesePairDataset(
        train_texts, train_labels, tokenizer, cfg.max_seq_length,
        num_pairs=len(train_texts) * 2,
    )
    siamese_val = SiamesePairDataset(
        val_texts, val_labels, tokenizer, cfg.max_seq_length,
        num_pairs=len(val_texts) * 2,
    )
    train_loader = DataLoader(siamese_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(siamese_val, batch_size=cfg.batch_size, shuffle=False)

    # Also create loaders for baseline
    baseline_train_ds = PromptDataset(train_texts, train_labels, tokenizer, cfg.max_seq_length)
    baseline_val_ds = PromptDataset(val_texts, val_labels, tokenizer, cfg.max_seq_length)
    baseline_train_loader = DataLoader(baseline_train_ds, batch_size=cfg.batch_size, shuffle=True)
    baseline_val_loader = DataLoader(baseline_val_ds, batch_size=cfg.batch_size, shuffle=False)

    # --- Models ---
    print("\n[4/6] Initializing models...")
    siamese_model = SiamesePromptDetector(
        model_name=cfg.model_name,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        freeze_layers=cfg.freeze_encoder_layers,
    ).to(device)

    baseline_model = BaselineClassifier(
        model_name=cfg.model_name,
        dropout=cfg.dropout,
    ).to(device)

    contrastive_criterion = ContrastiveLoss(margin=cfg.margin)

    # Class weights for baseline
    n_mal = sum(train_labels)
    n_ben = len(train_labels) - n_mal
    weight = torch.tensor([n_mal / len(train_labels), n_ben / len(train_labels)]).to(device)
    baseline_criterion = nn.CrossEntropyLoss(weight=weight)

    # --- Optimizers & schedulers ---
    epochs = cfg.quick_epochs if cfg.quick else cfg.epochs

    siam_optimizer = AdamW(siamese_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    siam_scheduler = get_linear_schedule_with_warmup(siam_optimizer, warmup_steps, total_steps)

    base_optimizer = AdamW(baseline_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    base_total_steps = len(baseline_train_loader) * epochs
    base_warmup = int(base_total_steps * cfg.warmup_ratio)
    base_scheduler = get_linear_schedule_with_warmup(base_optimizer, base_warmup, base_total_steps)

    # --- Training loop ---
    print(f"\n[5/6] Training ({epochs} epochs)...")
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_f1": [],
        "baseline_train_loss": [], "baseline_val_loss": [],
        "baseline_val_accuracy": [], "baseline_val_f1": [],
    }

    best_val_f1 = 0.0
    best_baseline_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n  --- Epoch {epoch}/{epochs} ---")

        # Siamese
        train_loss = train_siamese_epoch(
            siamese_model, train_loader, contrastive_criterion,
            siam_optimizer, siam_scheduler, device,
        )
        val_metrics = evaluate_siamese(
            siamese_model, val_loader, contrastive_criterion, device,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["contrastive_loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"  Siamese   | loss={train_loss:.4f} | val_loss={val_metrics['contrastive_loss']:.4f} "
              f"| val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}")

        # Baseline
        base_train_loss = train_baseline_epoch(
            baseline_model, baseline_train_loader, baseline_criterion,
            base_optimizer, base_scheduler, device,
        )
        base_val = evaluate_baseline(
            baseline_model, baseline_val_loader, baseline_criterion, device,
        )
        history["baseline_train_loss"].append(base_train_loss)
        history["baseline_val_loss"].append(base_val["loss"])
        history["baseline_val_accuracy"].append(base_val["accuracy"])
        history["baseline_val_f1"].append(base_val["f1"])

        print(f"  Baseline  | loss={base_train_loss:.4f} | val_loss={base_val['loss']:.4f} "
              f"| val_acc={base_val['accuracy']:.4f} | val_f1={base_val['f1']:.4f}")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        # --- Checkpointing ---
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_model(siamese_model, cfg.abs(cfg.model_dir) / "best_siamese_model.pt",
                       metadata={"epoch": epoch, "val_f1": best_val_f1, **val_metrics})
            patience_counter = 0
        else:
            patience_counter += 1

        if base_val["f1"] > best_baseline_f1:
            best_baseline_f1 = base_val["f1"]
            save_model(baseline_model, cfg.abs(cfg.model_dir) / "best_baseline_model.pt",
                       metadata={"epoch": epoch, "val_f1": best_baseline_f1, **base_val})

        # --- Early stopping ---
        if patience_counter >= cfg.patience and not cfg.quick:
            print(f"\n  Early stopping at epoch {epoch} (patience={cfg.patience})")
            break

    # --- Save history & plots ---
    print("\n[6/6] Saving results...")
    results_dir = cfg.abs(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved -> {history_path}")

    plot_training_history(history, results_dir / "plots" / "training_history.png")

    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE")
    print(f"  Best Siamese  val F1: {best_val_f1:.4f}")
    print(f"  Best Baseline val F1: {best_baseline_f1:.4f}")
    print("=" * 50)


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ghost in the Machine models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (1 epoch, subset)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.quick:
        cfg.quick = True
    train(cfg)


if __name__ == "__main__":
    main()
