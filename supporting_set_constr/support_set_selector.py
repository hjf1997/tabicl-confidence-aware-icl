import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from typing import List, Tuple

from config import SupportSetConfig, ReliabilityConfig


class SupportSetSelector:
    def __init__(self, config: SupportSetConfig, reliability_config: ReliabilityConfig):
        self.config = config
        self.reliability_config = reliability_config

    def build_initial_support_sets(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        K: int,
        size: int,
        positive_class: int = 0,
        random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build K diverse random stratified support sets."""
        rng = np.random.default_rng(random_state)

        pos_idx = np.where(y_train == positive_class)[0]
        neg_idx = np.where(y_train != positive_class)[0]

        n_pos = size // 2
        n_neg = size - n_pos

        support_sets = []
        for k in range(K):
            p_sample = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
            n_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
            idx = np.concatenate([p_sample, n_sample])
            support_sets.append((X_train[idx], y_train[idx]))

        return support_sets

    def build_optimized_support_set(
        self,
        X_positives: np.ndarray,
        y_positives: np.ndarray,
        X_negatives: np.ndarray,
        y_negatives: np.ndarray,
        neg_reliability_scores: np.ndarray,
        neg_prediction_vectors: np.ndarray,
        neg_classification: dict,
        random_state: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Build a single optimized support set (1000 total, balanced 500/500).

        Returns:
            ((X_support, y_support), (pos_indices, neg_indices))
        """
        target = self.config.target_size
        n_pos = int(target * self.config.positive_ratio)
        n_neg_reliable = int(target * self.config.negative_reliable_ratio)
        n_neg_boundary = target - n_pos - n_neg_reliable

        # Select diverse positives via feature-space k-means
        pos_selected_idx = self._diversity_sample_features(X_positives, n_pos, random_state)
        X_pos_sel = X_positives[pos_selected_idx]
        y_pos_sel = y_positives[pos_selected_idx]

        # Select diverse reliable negatives via prediction-space k-means
        reliable_mask = neg_classification["reliable"]
        if reliable_mask.sum() < n_neg_reliable:
            # Fall back: use all reliable + sample from uncertain
            n_from_reliable = int(reliable_mask.sum())
            n_extra = n_neg_reliable - n_from_reliable
            reliable_idx = np.where(reliable_mask)[0]
            uncertain_mask = neg_classification["uncertain"]
            uncertain_idx = np.where(uncertain_mask)[0]
            rng = np.random.default_rng(random_state)
            extra_idx = rng.choice(uncertain_idx, size=min(n_extra, len(uncertain_idx)), replace=False)
            neg_reliable_idx = np.concatenate([reliable_idx, extra_idx])
        else:
            neg_reliable_selected = self._diversity_sample_prediction_space(
                neg_prediction_vectors[reliable_mask], n_neg_reliable, random_state
            )
            neg_reliable_idx = np.where(reliable_mask)[0][neg_reliable_selected]

        # Select boundary negatives from uncertain zone (closest to reliable threshold)
        uncertain_mask = neg_classification["uncertain"]
        if uncertain_mask.sum() > 0:
            neg_boundary_idx = self._boundary_sample(
                neg_reliability_scores[uncertain_mask], n_neg_boundary
            )
            neg_boundary_idx = np.where(uncertain_mask)[0][neg_boundary_idx]
        else:
            neg_boundary_idx = np.array([], dtype=int)

        all_neg_idx = np.concatenate([neg_reliable_idx, neg_boundary_idx])
        X_neg_sel = X_negatives[all_neg_idx]
        y_neg_sel = y_negatives[all_neg_idx]

        X_support = np.vstack([X_pos_sel, X_neg_sel])
        y_support = np.concatenate([y_pos_sel, y_neg_sel])

        return (X_support, y_support), (pos_selected_idx, all_neg_idx)

    def build_K_optimized_support_sets(
        self,
        X_positives: np.ndarray,
        y_positives: np.ndarray,
        X_negatives: np.ndarray,
        y_negatives: np.ndarray,
        neg_reliability_scores: np.ndarray,
        neg_prediction_vectors: np.ndarray,
        neg_classification: dict,
        K: int,
        base_random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build K diverse optimized support sets (different random seeds for k-means)."""
        support_sets = []
        for k in range(K):
            (X_sup, y_sup), _ = self.build_optimized_support_set(
                X_positives, y_positives,
                X_negatives, y_negatives,
                neg_reliability_scores,
                neg_prediction_vectors,
                neg_classification,
                random_state=base_random_state + k,
            )
            support_sets.append((X_sup, y_sup))
        return support_sets

    def _diversity_sample_features(
        self, X: np.ndarray, n: int, random_state: int = 42
    ) -> np.ndarray:
        """Select n samples maximizing feature-space diversity via k-means."""
        if len(X) <= n:
            return np.arange(len(X))

        X_float = np.asarray(X, dtype=np.float64)
        km = KMeans(n_clusters=n, random_state=random_state, n_init=3)
        km.fit(X_float)

        selected = []
        for c in range(n):
            cluster_mask = km.labels_ == c
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            dists = np.linalg.norm(X_float[cluster_indices] - km.cluster_centers_[c], axis=1)
            selected.append(cluster_indices[np.argmin(dists)])

        return np.array(selected)

    def _diversity_sample_prediction_space(
        self, pred_vectors: np.ndarray, n: int, random_state: int = 42
    ) -> np.ndarray:
        """Select n samples diverse in prediction-behavior space via k-means."""
        if len(pred_vectors) <= n:
            return np.arange(len(pred_vectors))

        preds_norm = normalize(np.asarray(pred_vectors, dtype=np.float64), norm="l2")
        km = KMeans(n_clusters=n, random_state=random_state, n_init=3)
        km.fit(preds_norm)

        selected = []
        for c in range(n):
            cluster_mask = km.labels_ == c
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            dists = np.linalg.norm(preds_norm[cluster_indices] - km.cluster_centers_[c], axis=1)
            selected.append(cluster_indices[np.argmin(dists)])

        return np.array(selected)

    def _boundary_sample(
        self, scores: np.ndarray, n: int
    ) -> np.ndarray:
        """Select n samples from uncertain zone closest to reliable threshold (highest scores)."""
        if len(scores) <= n:
            return np.arange(len(scores))
        top_indices = np.argsort(scores)[-n:]
        return top_indices
