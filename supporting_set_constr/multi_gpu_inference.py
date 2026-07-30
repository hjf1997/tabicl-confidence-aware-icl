import numpy as np
import math
import torch.multiprocessing as mp
from typing import List, Tuple, Optional

from config import MultiGPUConfig, TabICLConfig


# Memory estimation coefficients profiled from TabICL source.
# The row encoder (tf_row) is the bottleneck on T4s:
#   peak_mem_MB = coef[0]*eff_batch + coef[1]*n_features + coef[2]*eff_batch*n_features + intercept
# where eff_batch = n_estimators * (n_support + n_query)
_ROW_COEFS = (-2.07e-05, 2.27e-04, 5.37e-03)
_ROW_INTERCEPT = 138.54  # MB


def estimate_max_query_chunk(
    n_support: int,
    n_features: int,
    n_estimators: int,
    gpu_memory_mb: float = 16384.0,
    safety_factor: float = 0.7,
) -> int:
    """Estimate max query samples per forward pass based on T4 memory constraints.

    Uses the row encoder (tf_row) as the bottleneck since its effective batch
    size scales with n_estimators * total_samples.
    """
    available = gpu_memory_mb * safety_factor
    c0, c1, c2 = _ROW_COEFS

    # Solve: available = c0*eff_bs + c1*n_features + c2*eff_bs*n_features + intercept
    # where eff_bs = n_estimators * (n_support + n_query)
    rhs = available - c1 * n_features - _ROW_INTERCEPT
    per_sample_cost = (c0 + c2 * n_features) * n_estimators

    max_total_samples = int(rhs / per_sample_cost)
    max_query = max_total_samples - n_support

    # Clamp to reasonable bounds
    max_query = max(100, min(max_query, 50000))
    return max_query


def _worker_inference(
    gpu_id: int,
    device: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    query_indices: np.ndarray,
    tabicl_kwargs: dict,
    chunk_size: int,
    result_queue: mp.Queue,
):
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, **tabicl_kwargs)
    clf.fit(X_support, y_support)

    n = len(X_query)
    if n <= chunk_size:
        proba = clf.predict_proba(X_query)
    else:
        chunks = math.ceil(n / chunk_size)
        proba_parts = []
        for i in range(chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            proba_parts.append(clf.predict_proba(X_query[start:end]))
        proba = np.vstack(proba_parts)

    result_queue.put((gpu_id, query_indices, proba))


def _worker_multi_support(
    gpu_id: int,
    device: str,
    support_sets: List[Tuple[np.ndarray, np.ndarray]],
    support_set_indices: List[int],
    X_query: np.ndarray,
    tabicl_kwargs: dict,
    chunk_size: int,
    result_queue: mp.Queue,
):
    from tabicl import TabICLClassifier

    n = len(X_query)
    results = []
    for local_idx, global_idx in enumerate(support_set_indices):
        X_sup, y_sup = support_sets[local_idx]
        clf = TabICLClassifier(device=device, **tabicl_kwargs)
        clf.fit(X_sup, y_sup)

        if n <= chunk_size:
            proba = clf.predict_proba(X_query)
        else:
            n_chunks = math.ceil(n / chunk_size)
            proba_parts = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n)
                proba_parts.append(clf.predict_proba(X_query[start:end]))
            proba = np.vstack(proba_parts)

        results.append((global_idx, proba[:, 1]))
    result_queue.put((gpu_id, results))


class MultiGPUInference:
    def __init__(self, config: MultiGPUConfig, tabicl_config: TabICLConfig):
        self.config = config
        self.tabicl_config = tabicl_config
        self.tabicl_kwargs = {
            "n_estimators": tabicl_config.n_estimators,
            "batch_size": tabicl_config.batch_size,
            "softmax_temperature": tabicl_config.softmax_temperature,
            "kv_cache": tabicl_config.kv_cache,
            "random_state": tabicl_config.random_state,
            "verbose": tabicl_config.verbose,
        }
        if tabicl_config.model_path:
            self.tabicl_kwargs["model_path"] = tabicl_config.model_path

    def _get_chunk_size(self, n_support: int, n_features: int) -> int:
        return estimate_max_query_chunk(
            n_support=n_support,
            n_features=n_features,
            n_estimators=self.tabicl_config.n_estimators,
        )

    def predict_proba_parallel(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """Run inference splitting query data across GPUs. Returns (n_query, n_classes)."""
        n = len(X_query)
        n_gpus = self.config.num_gpus
        n_features = X_support.shape[1]
        chunk_size = self._get_chunk_size(len(X_support), n_features)

        chunks = np.array_split(np.arange(n), n_gpus)

        result_queue = mp.Queue()
        processes = []

        for gpu_id, idx_chunk in enumerate(chunks):
            if len(idx_chunk) == 0:
                continue
            p = mp.Process(
                target=_worker_inference,
                args=(
                    gpu_id,
                    self.config.devices[gpu_id],
                    X_support,
                    y_support,
                    X_query[idx_chunk] if isinstance(X_query, np.ndarray) else X_query.iloc[idx_chunk].values,
                    idx_chunk,
                    self.tabicl_kwargs,
                    chunk_size,
                    result_queue,
                ),
            )
            processes.append(p)
            p.start()

        results = {}
        for _ in processes:
            gpu_id, indices, proba = result_queue.get()
            results[gpu_id] = (indices, proba)

        for p in processes:
            p.join()

        n_classes = list(results.values())[0][1].shape[1]
        all_proba = np.zeros((n, n_classes))
        for indices, proba in results.values():
            all_proba[indices] = proba

        return all_proba

    def predict_proba_multi_support(
        self,
        support_sets: List[Tuple[np.ndarray, np.ndarray]],
        X_query: np.ndarray,
    ) -> np.ndarray:
        """Run K support sets against all query samples across GPUs.

        Returns: (n_query, K) matrix of P(fraud) for each support set.
        """
        K = len(support_sets)
        n_gpus = self.config.num_gpus
        n_features = support_sets[0][0].shape[1]
        n_support = len(support_sets[0][0])
        chunk_size = self._get_chunk_size(n_support, n_features)

        gpu_assignments = [[] for _ in range(n_gpus)]
        gpu_support_sets = [[] for _ in range(n_gpus)]
        for k in range(K):
            gpu_idx = k % n_gpus
            gpu_assignments[gpu_idx].append(k)
            gpu_support_sets[gpu_idx].append(support_sets[k])

        if isinstance(X_query, np.ndarray):
            X_query_np = X_query
        else:
            X_query_np = X_query.values

        result_queue = mp.Queue()
        processes = []

        for gpu_id in range(n_gpus):
            if not gpu_assignments[gpu_id]:
                continue
            p = mp.Process(
                target=_worker_multi_support,
                args=(
                    gpu_id,
                    self.config.devices[gpu_id],
                    gpu_support_sets[gpu_id],
                    gpu_assignments[gpu_id],
                    X_query_np,
                    self.tabicl_kwargs,
                    chunk_size,
                    result_queue,
                ),
            )
            processes.append(p)
            p.start()

        predictions = np.zeros((len(X_query_np), K))
        for _ in processes:
            gpu_id, results_list = result_queue.get()
            for global_idx, proba_col in results_list:
                predictions[:, global_idx] = proba_col

        for p in processes:
            p.join()

        return predictions
