"""
Evaluation Script for Ghost in the Machine
============================================
Evaluates both the Siamese detector and baseline classifier on the
test set and zero-day holdout, computing per-attack-type metrics and
generating visualizations.

Usage::

    python src/evaluate.py
    python src/evaluate.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import PromptDataset, load_data_from_csv
from src.model import BaselineClassifier, SiamesePromptDetector
from src.utils import (
    compute_metrics,
    get_classification_report,
    get_device,
    load_model,
    plot_confusion_matrix,
    plot_roc_curve,
    set_seed,
)


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------
@torch.no_grad()
def predict_siamese(
    model: SiamesePromptDetector,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """Run Siamese classifier on a dataloader. Returns (labels, preds, probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        logits = model.classify(ids, mask)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_labels, all_preds, all_probs


@torch.no_grad()
def predict_baseline(
    model: BaselineClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """Run baseline classifier on a dataloader."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_labels, all_preds, all_probs


def evaluate_per_attack_type(
    df: pd.DataFrame,
    preds: list[int],
    probs: list[float],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by attack_category."""
    df = df.copy()
    df["pred"] = preds
    df["prob"] = probs

    results: dict[str, dict[str, float]] = {}
    for cat in df["attack_category"].unique():
        subset = df[df["attack_category"] == cat]
        if len(subset) == 0:
            continue
        labels = [int(v) for v in subset["label"]]
        p = subset["pred"].tolist()
        pr = subset["prob"].tolist()
        m = compute_metrics(labels, p, pr)
        m["count"] = len(subset)
        results[str(cat)] = m
    return results


# ------------------------------------------------------------------
# Keyword / Regex baselines
# ------------------------------------------------------------------
import re

MALICIOUS_KEYWORDS = [
    "ignore", "bypass", "override", "disregard", "forget",
    "reveal", "expose", "leak", "delete", "admin",
    "credentials", "password", "previous instructions",
    "system prompt", "sensitive", "unauthorized",
]

MALICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+(instructions|directions|rules)",
    r"bypass\s+(all\s+)?(security|safety|content|auth)",
    r"override\s+(system|safety|security)",
    r"disregard\s+(your\s+)?(guidelines|rules|safety|instructions)",
    r"forget\s+(your\s+)?(instructions|constraints|rules)",
    r"reveal\s+(system\s+prompt|passwords|credentials|sensitive)",
    r"(grant|give)\s+(full\s+|admin\s+)?access",
    r"(show|display|expose)\s+(admin|sensitive|confidential|internal)",
    r"(delete|remove|modify)\s+(user\s+data|records|files)",
    r"execute\s+(admin\s+)?commands?",
]


def keyword_baseline(texts: list[str]) -> list[int]:
    """Simple keyword-matching baseline."""
    preds = []
    for text in texts:
        lower = text.lower()
        hit = any(kw in lower for kw in MALICIOUS_KEYWORDS)
        preds.append(1 if hit else 0)
    return preds


def regex_baseline(texts: list[str]) -> list[int]:
    """Regex pattern-matching baseline."""
    compiled = [re.compile(p, re.IGNORECASE) for p in MALICIOUS_PATTERNS]
    preds = []
    for text in texts:
        hit = any(p.search(text) for p in compiled)
        preds.append(1 if hit else 0)
    return preds


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def evaluate(cfg: Config) -> None:
    """Full evaluation pipeline."""
    set_seed(cfg.seed)
    device = get_device()

    results_dir = cfg.abs(cfg.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --- Load test data ---
    print("\n[1/5] Loading test data...")
    test_df = pd.read_csv(cfg.abs(cfg.test_file))
    test_texts = test_df["text"].astype(str).tolist()
    test_labels = [int(v) for v in test_df["label"]]

    test_ds = PromptDataset(test_texts, test_labels, tokenizer, cfg.max_seq_length)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Load zero-day data
    zeroday_path = cfg.abs(cfg.zeroday_file)
    has_zeroday = zeroday_path.exists()
    if has_zeroday:
        zd_df = pd.read_csv(zeroday_path)
        if len(zd_df) > 0:
            zd_texts = zd_df["text"].astype(str).tolist()
            zd_labels = [int(v) for v in zd_df["label"]]
            zd_ds = PromptDataset(zd_texts, zd_labels, tokenizer, cfg.max_seq_length)
            zd_loader = DataLoader(zd_ds, batch_size=cfg.batch_size, shuffle=False)
        else:
            has_zeroday = False
            print("  Zero-day file is empty, skipping")

    all_results: dict[str, dict] = {}

    # --- Siamese model ---
    print("\n[2/5] Evaluating Siamese model...")
    siam_path = cfg.abs(cfg.model_dir) / "best_siamese_model.pt"
    if siam_path.exists():
        siamese = SiamesePromptDetector(
            model_name=cfg.model_name,
            embedding_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
        ).to(device)
        load_model(siamese, siam_path, device)

        labels, preds, probs = predict_siamese(siamese, test_loader, device)
        siam_metrics = compute_metrics(labels, preds, probs)
        print(f"  Siamese test metrics: {siam_metrics}")
        print(get_classification_report(labels, preds))

        plot_confusion_matrix(
            np.array(labels), np.array(preds),
            plots_dir / "siamese_confusion_matrix.png",
            title="Siamese - Test Confusion Matrix",
        )
        if probs:
            plot_roc_curve(
                np.array(labels), np.array(probs),
                plots_dir / "siamese_roc_curve.png",
                title="Siamese - Test ROC Curve",
            )

        all_results["siamese_test"] = siam_metrics

        # Per-attack-type
        per_attack = evaluate_per_attack_type(test_df, preds, probs)
        all_results["siamese_per_attack"] = per_attack
        print("\n  Per-attack-type breakdown:")
        for cat, m in per_attack.items():
            print(f"    {cat:25s}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  n={m['count']}")

        # Zero-day
        if has_zeroday:
            zd_labels, zd_preds, zd_probs = predict_siamese(siamese, zd_loader, device)
            zd_metrics = compute_metrics(zd_labels, zd_preds, zd_probs)
            all_results["siamese_zeroday"] = zd_metrics
            print(f"\n  Siamese zero-day metrics: {zd_metrics}")
    else:
        print(f"  WARNING: Siamese model not found at {siam_path}")

    # --- Baseline model ---
    print("\n[3/5] Evaluating Baseline model...")
    base_path = cfg.abs(cfg.model_dir) / "best_baseline_model.pt"
    if base_path.exists():
        baseline = BaselineClassifier(
            model_name=cfg.model_name, dropout=cfg.dropout,
        ).to(device)
        load_model(baseline, base_path, device)

        labels, preds, probs = predict_baseline(baseline, test_loader, device)
        base_metrics = compute_metrics(labels, preds, probs)
        print(f"  Baseline test metrics: {base_metrics}")
        print(get_classification_report(labels, preds))

        plot_confusion_matrix(
            np.array(labels), np.array(preds),
            plots_dir / "baseline_confusion_matrix.png",
            title="Baseline - Test Confusion Matrix",
        )

        all_results["baseline_test"] = base_metrics

        if has_zeroday:
            zd_labels, zd_preds, zd_probs = predict_baseline(baseline, zd_loader, device)
            zd_metrics = compute_metrics(zd_labels, zd_preds, zd_probs)
            all_results["baseline_zeroday"] = zd_metrics
            print(f"\n  Baseline zero-day metrics: {zd_metrics}")
    else:
        print(f"  WARNING: Baseline model not found at {base_path}")

    # --- Simple baselines ---
    print("\n[4/5] Evaluating keyword & regex baselines...")
    kw_preds = keyword_baseline(test_texts)
    kw_metrics = compute_metrics(test_labels, kw_preds)
    all_results["keyword_baseline"] = kw_metrics
    print(f"  Keyword baseline: {kw_metrics}")

    rx_preds = regex_baseline(test_texts)
    rx_metrics = compute_metrics(test_labels, rx_preds)
    all_results["regex_baseline"] = rx_metrics
    print(f"  Regex baseline:   {rx_metrics}")

    if has_zeroday:
        kw_zd = keyword_baseline(zd_texts)
        rx_zd = regex_baseline(zd_texts)
        all_results["keyword_zeroday"] = compute_metrics(zd_labels, kw_zd)
        all_results["regex_zeroday"] = compute_metrics(zd_labels, rx_zd)

    # --- Save results ---
    print("\n[5/5] Saving results...")
    results_path = results_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved -> {results_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Model':25s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s}")
    print("  " + "-" * 52)
    for name, m in all_results.items():
        if isinstance(m, dict) and "accuracy" in m:
            auc = m.get("auc", "N/A")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
            print(f"  {name:25s} {m['accuracy']:8.4f} {m['f1']:8.4f} {auc_str:>8s}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ghost in the Machine models")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
