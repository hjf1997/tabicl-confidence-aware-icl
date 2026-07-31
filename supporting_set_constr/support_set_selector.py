import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from typing import List, Tuple

from config import SupportSetConfig, ReliabilityConfig


class SupportSetSelector:
    def __init__(self, config: SupportSetConfig, reliability_config: ReliabilityConfig,
                 neg_sampling_strategy: str = "random"):
        self.config = config
        self.reliability_config = reliability_config
        self.neg_sampling_strategy = neg_sampling_strategy

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
        n_neg_override: int = None,
        random_state: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Build optimized support set using all positives + matched negatives.

        Negative sampling strategies:
            - random: random sample from reliable pool
            - kmeans: KMeans centroids in prediction space (representative)
            - boundary: hard examples near decision boundary
            - hybrid: mixture of representative + random + hard + boundary

        Returns:
            ((X_support, y_support), (pos_indices, neg_indices))
        """
        # Use ALL positive samples
        pos_selected_idx = np.arange(len(X_positives))
        X_pos_sel = X_positives
        y_pos_sel = y_positives

        # Number of negatives = number of positives (balanced)
        n_neg_total = n_neg_override if n_neg_override is not None else len(X_positives)

        # Build candidate pool: reliable + uncertain (exclude suspect)
        reliable_mask = neg_classification["reliable"]
        uncertain_mask = neg_classification["uncertain"]
        candidate_mask = reliable_mask | uncertain_mask
        candidate_idx = np.where(candidate_mask)[0]
        candidate_scores = neg_reliability_scores[candidate_mask]
        candidate_pred_vectors = neg_prediction_vectors[candidate_mask]

        rng = np.random.default_rng(random_state)

        if self.neg_sampling_strategy == "random":
            all_neg_idx = self._neg_sample_random(
                candidate_idx, candidate_scores, n_neg_total, rng
            )
        elif self.neg_sampling_strategy == "kmeans":
            all_neg_idx = self._neg_sample_kmeans(
                candidate_idx, candidate_pred_vectors, n_neg_total, random_state
            )
        elif self.neg_sampling_strategy == "boundary":
            all_neg_idx = self._neg_sample_boundary(
                candidate_idx, candidate_pred_vectors, candidate_scores, n_neg_total, rng
            )
        elif self.neg_sampling_strategy == "hybrid":
            all_neg_idx = self._neg_sample_hybrid(
                candidate_idx, candidate_pred_vectors, candidate_scores, n_neg_total, random_state, rng
            )
        elif self.neg_sampling_strategy == "reliable":
            all_neg_idx = self._neg_sample_reliable(
                candidate_idx, candidate_scores, n_neg_total
            )
        elif self.neg_sampling_strategy == "informed":
            all_neg_idx = self._neg_sample_informed(
                candidate_idx, candidate_pred_vectors, candidate_scores, n_neg_total
            )
        else:
            raise ValueError(f"Unknown neg_sampling_strategy: {self.neg_sampling_strategy}")

        X_neg_sel = X_negatives[all_neg_idx]
        y_neg_sel = y_negatives[all_neg_idx]

        X_support = np.vstack([X_pos_sel, X_neg_sel])
        y_support = np.concatenate([y_pos_sel, y_neg_sel])

        return (X_support, y_support), (pos_selected_idx, all_neg_idx)

    def _neg_sample_informed(self, candidate_idx, candidate_pred_vectors, candidate_scores, n):
        """Informativeness-weighted sampling: reliability as foundation, variance as bonus.

        S(x_i) = R(x_i) * (1 + alpha * var_norm)

        Reliability provides the foundation; prediction disagreement provides a bonus.
        Unreliable samples are penalized regardless of variance.
        Percentile normalization (5th/95th) for robustness against outliers.
        """
        if len(candidate_idx) <= n:
            return candidate_idx.copy()

        alpha = 0.25
        variance = candidate_pred_vectors.var(axis=1)

        q05 = np.percentile(variance, 5)
        q95 = np.percentile(variance, 95)
        var_norm = np.clip((variance - q05) / (q95 - q05 + 1e-8), 0, 1)

        score = candidate_scores * (1 + alpha * var_norm)
        top_indices = np.argsort(score)[-n:]
        return candidate_idx[top_indices]

    def _neg_sample_reliable(self, candidate_idx, candidate_scores, n):
        """Top-n samples with highest reliability scores."""
        if len(candidate_idx) <= n:
            return candidate_idx.copy()
        top_indices = np.argsort(candidate_scores)[-n:]
        return candidate_idx[top_indices]

    def _neg_sample_random(self, candidate_idx, candidate_scores, n, rng):
        """Random sampling from candidate pool (preserves natural distribution)."""
        if len(candidate_idx) <= n:
            return candidate_idx.copy()
        selected = rng.choice(len(candidate_idx), size=n, replace=False)
        return candidate_idx[selected]

    def _neg_sample_kmeans(self, candidate_idx, candidate_pred_vectors, n, random_state):
        """KMeans centroids in prediction space (representative sampling)."""
        if len(candidate_idx) <= n:
            return candidate_idx.copy()
        selected_local = self._diversity_sample_prediction_space(
            candidate_pred_vectors, n, random_state
        )
        return candidate_idx[selected_local]

    def _neg_sample_boundary(self, candidate_idx, candidate_pred_vectors, candidate_scores, n, rng):
        """Hard-example sampling: samples near the decision boundary.

        Uses two signals:
        - High mean prediction (fraud samples that look like bogus)
        - High prediction variance (unstable classification across support sets)
        Combines them into a boundary score and selects top-n.
        """
        if len(candidate_idx) <= n:
            return candidate_idx.copy()

        mean_pred = candidate_pred_vectors.mean(axis=1)
        variance_pred = candidate_pred_vectors.var(axis=1)

        mean_pred_norm = (mean_pred - mean_pred.min()) / (mean_pred.max() - mean_pred.min() + 1e-10)
        variance_norm = (variance_pred - variance_pred.min()) / (variance_pred.max() - variance_pred.min() + 1e-10)

        boundary_score = 0.6 * mean_pred_norm + 0.4 * variance_norm
        top_indices = np.argsort(boundary_score)[-n:]
        return candidate_idx[top_indices]

    def _neg_sample_hybrid(self, candidate_idx, candidate_pred_vectors, candidate_scores, n, random_state, rng):
        """Hybrid mixture: representative (40%) + random (20%) + hard (20%) + boundary (20%).

        - Representative: KMeans centroids — clear examples of fraud structure
        - Random: preserves natural distribution
        - Hard: highest mean prediction (fraud that looks like bogus)
        - Boundary: highest prediction variance (unstable across support sets)
        """
        if len(candidate_idx) <= n:
            return candidate_idx.copy()

        n_representative = int(n * 0.4)
        n_random = int(n * 0.2)
        n_hard = int(n * 0.2)
        n_boundary = n - n_representative - n_random - n_hard

        selected_set = set()

        # Representative: KMeans centroids
        repr_local = self._diversity_sample_prediction_space(
            candidate_pred_vectors, n_representative, random_state
        )
        for i in repr_local:
            selected_set.add(int(i))

        # Hard: highest mean prediction (fraud looking like bogus)
        mean_pred = candidate_pred_vectors.mean(axis=1)
        hard_order = np.argsort(mean_pred)[::-1]
        for i in hard_order:
            if len(selected_set) >= n_representative + n_hard:
                break
            if int(i) not in selected_set:
                selected_set.add(int(i))

        # Boundary: highest prediction variance
        variance_pred = candidate_pred_vectors.var(axis=1)
        boundary_order = np.argsort(variance_pred)[::-1]
        for i in boundary_order:
            if len(selected_set) >= n_representative + n_hard + n_boundary:
                break
            if int(i) not in selected_set:
                selected_set.add(int(i))

        # Random: fill remaining from unselected candidates
        remaining_pool = [i for i in range(len(candidate_idx)) if i not in selected_set]
        n_remaining = n - len(selected_set)
        if n_remaining > 0 and len(remaining_pool) > 0:
            random_selected = rng.choice(remaining_pool, size=min(n_remaining, len(remaining_pool)), replace=False)
            for i in random_selected:
                selected_set.add(int(i))

        selected_local = np.array(sorted(selected_set))
        return candidate_idx[selected_local]

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
