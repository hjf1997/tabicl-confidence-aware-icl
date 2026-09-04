"""Shared, torch-free evaluation helpers used by the ablation scripts.

Kept free of torch imports on purpose: the trained-ML baseline script links
GBDT libraries whose OpenMP runtime can clash with torch's in one process
(observed segfault on macOS), and it doesn't need the GPU stack anyway.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)


def compute_metrics(y_true, y_proba, threshold=None):
    """Compute metrics. Positive class = bogus (label 0).

    If threshold is None, finds optimal threshold from the data (use for val).
    If threshold is provided, applies it directly (use for test with val-determined threshold).

    Returns dict with roc_auc, pr_auc, f1, precision, recall, optimal_threshold, n_predicted_bogus.
    """
    pos_proba = y_proba[:, 0]  # P(bogus)
    y_binary = (y_true == 0).astype(int)  # 1 where bogus

    roc_auc = roc_auc_score(y_binary, pos_proba)
    pr_auc = average_precision_score(y_binary, pos_proba)

    if threshold is None:
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_binary, pos_proba)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        threshold = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5

    y_pred = (pos_proba >= threshold).astype(int)
    f1 = float(f1_score(y_binary, y_pred))
    prec = float(precision_score(y_binary, y_pred, zero_division=0))
    rec = float(recall_score(y_binary, y_pred, zero_division=0))

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "optimal_threshold": float(threshold),
        "n_predicted_bogus": int(y_pred.sum()),
    }


def array_sha256(*arrays) -> str:
    """Order-sensitive content hash of numpy arrays (16 hex chars).

    Used to verify that runs across model families consumed byte-identical
    inputs: matching hashes = matching support sets / query matrices.
    """
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def load_support_set_csv(csv_path: Path, feature_columns, label_col: str):
    """Load a materialized support set saved by a previous ablation run.

    Returns (X_support float64, y_support). Raises if any expected feature
    column is missing (e.g. --top-features mismatch with the source run).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} lacks {len(missing)} expected feature column(s) "
            f"(first: {missing[:3]}); was the source run using the same --top-features?")
    return df[feature_columns].values.astype(np.float64), df[label_col].values
