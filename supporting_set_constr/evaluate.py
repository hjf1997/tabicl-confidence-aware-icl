import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from typing import Dict


class Evaluator:
    def compute_all(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Compute metrics. Positive class = bogus (label 0).

        Args:
            y_true: Ground truth labels (0=bogus, 1=fraud).
            y_proba: shape (n, 2) — class probabilities [P(bogus), P(fraud)].

        Returns:
            Dict with roc_auc, pr_auc, f1, optimal_threshold, n_predicted_bogus.
        """
        pos_proba = y_proba[:, 0]  # P(bogus)
        y_binary = (y_true == 0).astype(int)  # 1 where bogus

        roc_auc = roc_auc_score(y_binary, pos_proba)
        pr_auc = average_precision_score(y_binary, pos_proba)

        precision, recall, thresholds = precision_recall_curve(y_binary, pos_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5

        y_pred = (pos_proba >= optimal_threshold).astype(int)
        f1 = float(f1_score(y_binary, y_pred))

        return {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "f1": float(f1),
            "optimal_threshold": optimal_threshold,
            "n_predicted_bogus": int(y_pred.sum()),
        }
