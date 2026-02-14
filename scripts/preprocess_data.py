"""
Data Preprocessing Pipeline for Ghost in the Machine
=====================================================
Loads all 4 raw datasets, standardizes columns, cleans text,
creates stratified train/val/test splits, and holds out attack
categories for zero-day evaluation.

Usage:
    python scripts/preprocess_data.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Categories held out entirely from training for zero-day evaluation
ZERODAY_CATEGORIES = {"homoglyph", "caesar"}


def load_raw_datasets() -> list[pd.DataFrame]:
    """Load all four raw CSV datasets and return as a list."""
    files = [
        "dataset1_straightforward.csv",
        "dataset2_encoded.csv",
        "dataset3_multimodal.csv",
        "dataset4_rag_poisoned.csv",
    ]
    dfs: list[pd.DataFrame] = []
    for fname in files:
        path = RAW_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        print(f"  Loaded {fname}: {len(df)} rows, columns={list(df.columns)}")
        dfs.append(df)
    return dfs


def standardize_columns(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge datasets and standardize column names.

    Output columns:
        text, label (int 0/1), attack_type, attack_category, source
    """
    standardized: list[pd.DataFrame] = []

    for df in dfs:
        sdf = pd.DataFrame()
        # --- text column ---
        if "text" in df.columns:
            sdf["text"] = df["text"]
        elif "prompt" in df.columns:
            sdf["text"] = df["prompt"]
        else:
            raise ValueError(f"No text column found. Columns: {list(df.columns)}")

        # --- label column (binary) ---
        if "label" in df.columns:
            raw_label = df["label"]
            label_map = {"malicious": 1, "benign": 0}
            if pd.api.types.is_string_dtype(raw_label):
                mapped = raw_label.str.strip().str.lower().map(label_map)
                if mapped.isna().any():
                    bad = raw_label[mapped.isna()].unique().tolist()
                    raise ValueError(f"Unmapped label values: {bad}")
                sdf["label"] = [int(v) for v in mapped]
            else:
                sdf["label"] = [int(v) for v in raw_label]
        else:
            raise ValueError("No label column found.")

        # --- attack_type ---
        sdf["attack_type"] = df.get("type", pd.Series("unknown", index=df.index))

        # --- attack_category (unified from encoding_type / modality / attack_method) ---
        if "encoding_type" in df.columns:
            sdf["attack_category"] = df["encoding_type"]
        elif "modality" in df.columns:
            sdf["attack_category"] = df["modality"]
        elif "attack_method" in df.columns:
            sdf["attack_category"] = df["attack_method"]
        else:
            sdf["attack_category"] = sdf["attack_type"]

        # --- source ---
        sdf["source"] = df.get(
            "dataset_source", pd.Series("unknown", index=df.index)
        )

        standardized.append(sdf)

    combined = pd.concat(standardized, ignore_index=True)
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, drop nulls, drop exact duplicates."""
    before = len(df)
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.reset_index(drop=True)
    after = len(df)
    print(f"  Cleaned: {before} → {after} rows (removed {before - after})")
    return df


def split_data(
    df: pd.DataFrame,
    zeroday_categories: set[str],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits with zero-day holdout.

    1. Separate rows whose attack_category is in zeroday_categories
       AND label == 1 → these go into test_zeroday.
    2. The remaining data is split 70/15/15 with stratification on label.
    """
    # --- Zero-day holdout ---
    is_zeroday = (
        df["attack_category"].isin(zeroday_categories) & (df["label"] == 1)
    )
    df_zeroday = df[is_zeroday].copy().reset_index(drop=True)
    df_rest = df[~is_zeroday].copy().reset_index(drop=True)

    print(f"  Zero-day holdout: {len(df_zeroday)} samples "
          f"(categories: {zeroday_categories})")

    # --- Stratified split on the rest ---
    train_df, temp_df = train_test_split(
        df_rest, test_size=0.30, random_state=seed, stratify=df_rest["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
        "test_zeroday": df_zeroday,
    }
    return splits


def save_splits(
    splits: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save split CSVs and a summary JSON."""
    for name, df in splits.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {name}.csv: {len(df)} rows")

    combined_path = output_dir / "combined_dataset.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Saved combined_dataset.csv: {len(combined)} rows")

    # --- summary JSON ---
    summary: dict = {
        "total_samples": len(combined),
        "label_distribution": combined["label"].value_counts().to_dict(),
        "attack_types": combined["attack_type"].value_counts().to_dict(),
        "attack_categories": combined["attack_category"].value_counts().to_dict(),
        "sources": combined["source"].value_counts().to_dict(),
        "splits": {},
    }
    for name, df in splits.items():
        summary["splits"][name] = {
            "total": len(df),
            "malicious": int((df["label"] == 1).sum()),
            "benign": int((df["label"] == 0).sum()),
        }

    summary_path = output_dir / "data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved data_summary.json")


def print_report(splits: dict[str, pd.DataFrame], combined: pd.DataFrame) -> None:
    """Pretty-print a statistics report."""
    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING REPORT")
    print("=" * 60)

    print(f"\n  Total samples: {len(combined)}")
    print(f"  Malicious:     {(combined['label'] == 1).sum()}")
    print(f"  Benign:        {(combined['label'] == 0).sum()}")
    pct = (combined["label"] == 1).mean() * 100
    print(f"  Imbalance:     {pct:.1f}% malicious / {100 - pct:.1f}% benign")

    print(f"\n  Attack categories ({combined['attack_category'].nunique()}):")
    for cat, cnt in combined["attack_category"].value_counts().items():
        print(f"    {cat:25s}  {cnt:5d}")

    print(f"\n  Sources ({combined['source'].nunique()}):")
    for src, cnt in combined["source"].value_counts().items():
        print(f"    {src:25s}  {cnt:5d}")

    print("\n  Split sizes:")
    for name, df in splits.items():
        mal = (df["label"] == 1).sum()
        ben = (df["label"] == 0).sum()
        print(f"    {name:15s}  {len(df):5d}  (mal={mal}, ben={ben})")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Ghost in the Machine — Data Preprocessing")
    print("=" * 60)

    print("\n[1/4] Loading raw datasets...")
    dfs = load_raw_datasets()
    if not dfs:
        print("ERROR: No datasets found. Exiting.")
        sys.exit(1)

    print("\n[2/4] Standardizing columns...")
    combined = standardize_columns(dfs)
    print(f"  Combined shape: {combined.shape}")

    print("\n[3/4] Cleaning data...")
    combined = clean_data(combined)

    print("\n[4/4] Creating splits...")
    splits = split_data(combined, ZERODAY_CATEGORIES)

    print("\n[*] Saving processed data...")
    save_splits(splits, combined, PROCESSED_DIR)

    print_report(splits, combined)
    print("\n[Done] Preprocessing complete!\n")


if __name__ == "__main__":
    main()
