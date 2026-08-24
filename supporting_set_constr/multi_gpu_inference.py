import numpy as np
import math
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import List, Tuple, Optional

from config import MultiGPUConfig, TabICLConfig


# Memory estimation coefficients profiled from TabICL source.
_ROW_COEFS = (-2.07e-05, 2.27e-04, 5.37e-03)
_ROW_INTERCEPT = 138.54  # MB

_PROGRESS_MSG = "__progress__"

# Fixed outer query-chunk size for TabPFN (its memory_saving_mode handles the
# fine-grained batching internally; the outer chunk just drives progress and
# bounds transfer sizes). The TabICL chunk comes from the profiled estimator.
TABPFN_QUERY_CHUNK = 5000


def _make_classifier(model_family: str, device: str, model_kwargs: dict):
    """Instantiate the in-context classifier for a worker process."""
    if model_family == "tabpfn":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device=device, **model_kwargs)
    from tabicl import TabICLClassifier

    return TabICLClassifier(device=device, **model_kwargs)


def estimate_max_query_chunk(
    n_support: int,
    n_features: int,
    n_estimators: int,
    gpu_memory_mb: float = 16384.0,
    safety_factor: float = 0.7,
) -> int:
    """Estimate max query samples per forward pass based on T4 memory constraints."""
    available = gpu_memory_mb * safety_factor
    c0, c1, c2 = _ROW_COEFS

    rhs = available - c1 * n_features - _ROW_INTERCEPT
    per_sample_cost = (c0 + c2 * n_features) * n_estimators

    max_total_samples = int(rhs / per_sample_cost)
    max_query = max_total_samples - n_support

    max_query = max(100, min(max_query, 50000))
    return max_query


def _worker_inference(
    gpu_id: int,
    device: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    query_indices: np.ndarray,
    model_family: str,
    model_kwargs: dict,
    chunk_size: int,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    clf = _make_classifier(model_family, device, model_kwargs)
    clf.fit(X_support, y_support)

    n = len(X_query)
    if n <= chunk_size:
        proba = clf.predict_proba(X_query)
        progress_queue.put((_PROGRESS_MSG, n))
    else:
        n_chunks = math.ceil(n / chunk_size)
        proba_parts = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            proba_parts.append(clf.predict_proba(X_query[start:end]))
            progress_queue.put((_PROGRESS_MSG, end - start))
        proba = np.vstack(proba_parts)

    result_queue.put((gpu_id, query_indices, proba))


def _worker_multi_support(
    gpu_id: int,
    device: str,
    support_sets: List[Tuple[np.ndarray, np.ndarray]],
    support_set_indices: List[int],
    X_query: np.ndarray,
    model_family: str,
    model_kwargs: dict,
    chunk_size: int,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    n = len(X_query)
    results = []
    for local_idx, global_idx in enumerate(support_set_indices):
        X_sup, y_sup = support_sets[local_idx]
        clf = _make_classifier(model_family, device, model_kwargs)
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
        progress_queue.put((_PROGRESS_MSG, 1))

    result_queue.put((gpu_id, results))


class MultiGPUInference:
    def __init__(self, config: MultiGPUConfig, tabicl_config: TabICLConfig):
        self.config = config
        self.tabicl_config = tabicl_config
        self.model_family = getattr(tabicl_config, "model_family", "tabicl")
        if self.model_family == "tabpfn":
            # tabpfn v2 (pip tabpfn==2.2.1): near-mirror of the TabICL API.
            # fit_with_cache is the kv_cache analog (caches train-side state
            # for repeated predict_proba calls on one fitted context).
            self.model_kwargs = {
                "n_estimators": tabicl_config.n_estimators,
                "softmax_temperature": tabicl_config.softmax_temperature,
                "random_state": tabicl_config.random_state,
                "fit_mode": "fit_with_cache",
                "memory_saving_mode": "auto",
                "ignore_pretraining_limits": True,
            }
            if tabicl_config.model_path:
                self.model_kwargs["model_path"] = tabicl_config.model_path
        else:
            self.model_kwargs = {
                "n_estimators": tabicl_config.n_estimators,
                "batch_size": tabicl_config.batch_size,
                "softmax_temperature": tabicl_config.softmax_temperature,
                "kv_cache": tabicl_config.kv_cache,
                "random_state": tabicl_config.random_state,
                "verbose": tabicl_config.verbose,
            }
            if tabicl_config.model_path:
                self.model_kwargs["model_path"] = tabicl_config.model_path
        # Backward-compat alias (older call sites referenced tabicl_kwargs).
        self.tabicl_kwargs = self.model_kwargs

    def _get_chunk_size(self, n_support: int, n_features: int) -> int:
        if self.model_family == "tabpfn":
            return TABPFN_QUERY_CHUNK
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
        desc: str = "Inference",
    ) -> np.ndarray:
        """Run inference splitting query data across GPUs. Returns (n_query, n_classes)."""
        n = len(X_query)
        n_gpus = self.config.num_gpus
        n_features = X_support.shape[1]
        chunk_size = self._get_chunk_size(len(X_support), n_features)

        chunks = np.array_split(np.arange(n), n_gpus)

        result_queue = mp.Queue()
        progress_queue = mp.Queue()
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
                    self.model_family,
                    self.model_kwargs,
                    chunk_size,
                    result_queue,
                    progress_queue,
                ),
            )
            processes.append(p)
            p.start()

        pbar = tqdm(total=n, desc=desc, unit="samples")
        results = {}
        while len(results) < len(processes):
            try:
                msg = progress_queue.get_nowait()
                if msg[0] == _PROGRESS_MSG:
                    pbar.update(msg[1])
            except Exception:
                pass

            try:
                gpu_id, indices, proba = result_queue.get(timeout=0.1)
                results[gpu_id] = (indices, proba)
            except Exception:
                pass

        # Drain remaining progress
        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                if msg[0] == _PROGRESS_MSG:
                    pbar.update(msg[1])
            except Exception:
                break
        pbar.close()

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
        desc: str = "Multi-support inference",
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
        progress_queue = mp.Queue()
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
                    self.model_family,
                    self.model_kwargs,
                    chunk_size,
                    result_queue,
                    progress_queue,
                ),
            )
            processes.append(p)
            p.start()

        pbar = tqdm(total=K, desc=desc, unit="support_sets")
        predictions = np.zeros((len(X_query_np), K))
        results_collected = 0

        while results_collected < len(processes):
            try:
                msg = progress_queue.get_nowait()
                if msg[0] == _PROGRESS_MSG:
                    pbar.update(msg[1])
            except Exception:
                pass

            try:
                gpu_id, results_list = result_queue.get(timeout=0.1)
                for global_idx, proba_col in results_list:
                    predictions[:, global_idx] = proba_col
                results_collected += 1
            except Exception:
                pass

        # Drain remaining progress
        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                if msg[0] == _PROGRESS_MSG:
                    pbar.update(msg[1])
            except Exception:
                break
        pbar.close()

        for p in processes:
            p.join()

        return predictions
