import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from typing import Dict, Sequence, Tuple

from config import ReliabilityConfig


def variance_decomposition(
    predictions: np.ndarray, anchor_ids: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample variance decomposition for the factorial anchored probe design.

    Args:
        predictions: (n_samples, K) P(fraud) matrix, K = M * draws_per_anchor.
        anchor_ids: length-K anchor index of each probe column.

    Returns:
        (within, between), each shape (n_samples,):
        - within: mean over anchors of the variance across fraud draws —
          sensitivity to the fraud evidence (the label-noise signal).
        - between: variance of the per-anchor means — sensitivity to the
          trusted anchor choice (model brittleness; should be small if the
          anchored design is working).
    """
    anchor_ids = np.asarray(anchor_ids)
    groups = [predictions[:, anchor_ids == m] for m in np.unique(anchor_ids)]
    within = np.mean([g.var(axis=1) for g in groups], axis=0)
    anchor_means = np.stack([g.mean(axis=1) for g in groups], axis=1)
    between = anchor_means.var(axis=1)
    return within, between


class ReliabilityScorer:
    def __init__(self, config: ReliabilityConfig):
        self.config = config
        self.prediction_vectors_ = None
        self.scores_ = None
        self.components_ = None

    def compute_scores(self, predictions_matrix: np.ndarray) -> np.ndarray:
        """Compute reliability scores for all fraud-labeled samples.

        Args:
            predictions_matrix: shape (n_fraud_samples, K) — P(fraud) from each support set.

        Returns:
            reliability_scores: shape (n_fraud_samples,) in [0, 1]
        """
        self.prediction_vectors_ = predictions_matrix

        mu = self._mean_probability(predictions_matrix)
        sigma = self._stability(predictions_matrix)
        H = self._normalized_entropy(predictions_matrix)
        A = self._neighborhood_agreement(predictions_matrix)
        D = self._density(predictions_matrix)

        scores = (
            self.config.w_mean_prob * mu
            + self.config.w_stability * (1.0 - sigma)
            + self.config.w_entropy * (1.0 - H)
            + self.config.w_agreement * A
            + self.config.w_density * D
        )

        self.components_ = {"mu": mu, "sigma": sigma, "H": H, "A": A, "D": D}
        self.scores_ = scores
        return scores

    def _mean_probability(self, preds: np.ndarray) -> np.ndarray:
        """Higher mean P(fraud) = more consistently classified as fraud = more reliable."""
        return preds.mean(axis=1)

    def _stability(self, preds: np.ndarray) -> np.ndarray:
        """Prediction variance normalized to [0,1]. Lower = more stable."""
        return np.clip(preds.std(axis=1) / 0.5, 0, 1)

    def _normalized_entropy(self, preds: np.ndarray) -> np.ndarray:
        """Binary entropy from mean probability. High entropy = model is confused."""
        mu = preds.mean(axis=1)
        mu_clipped = np.clip(mu, 1e-7, 1 - 1e-7)
        H = -(mu_clipped * np.log2(mu_clipped) + (1 - mu_clipped) * np.log2(1 - mu_clipped))
        return H

    def _neighborhood_agreement(self, preds: np.ndarray) -> np.ndarray:
        """Fraction of prediction-behavior neighbors that also look like fraud."""
        preds_norm = normalize(preds, norm="l2")

        nn = NearestNeighbors(
            n_neighbors=self.config.n_neighbors + 1,
            metric=self.config.similarity_metric,
            algorithm="brute",
        )
        nn.fit(preds_norm)
        _, indices = nn.kneighbors(preds_norm)

        mu = preds.mean(axis=1)
        is_fraud_like = (mu > 0.5).astype(float)

        agreement = np.array([
            is_fraud_like[indices[i, 1:]].mean() for i in range(len(preds))
        ])
        return agreement

    def _density(self, preds: np.ndarray) -> np.ndarray:
        """Local density in prediction-behavior space (inverse mean distance to neighbors)."""
        preds_norm = normalize(preds, norm="l2")

        nn = NearestNeighbors(
            n_neighbors=self.config.n_neighbors + 1,
            metric=self.config.similarity_metric,
            algorithm="brute",
        )
        nn.fit(preds_norm)
        distances, _ = nn.kneighbors(preds_norm)

        mean_dist = distances[:, 1:].mean(axis=1)
        density = 1.0 / (mean_dist + 1e-8)
        d_min, d_max = density.min(), density.max()
        density = (density - d_min) / (d_max - d_min + 1e-8)
        return density

    def classify_samples(self, scores: np.ndarray) -> Dict[str, np.ndarray]:
        """Partition fraud samples into reliable, uncertain, and suspect (boolean masks).

        If threshold_method is "percentile", thresholds are derived from the score
        distribution (top reliable_percentile% = reliable, bottom suspect_percentile% = suspect).
        Otherwise uses fixed thresholds from config.
        """
        if self.config.threshold_method == "percentile":
            reliable_thresh = np.percentile(scores, 100 - self.config.reliable_percentile)
            suspect_thresh = np.percentile(scores, self.config.suspect_percentile)
            self.resolved_thresholds_ = {
                "reliable_threshold": float(reliable_thresh),
                "uncertain_threshold": float(suspect_thresh),
            }
        else:
            reliable_thresh = self.config.reliable_threshold
            suspect_thresh = self.config.uncertain_threshold
            self.resolved_thresholds_ = {
                "reliable_threshold": float(reliable_thresh),
                "uncertain_threshold": float(suspect_thresh),
            }

        return {
            "reliable": scores >= reliable_thresh,
            "uncertain": (scores >= suspect_thresh) & (scores < reliable_thresh),
            "suspect": scores < suspect_thresh,
        }
