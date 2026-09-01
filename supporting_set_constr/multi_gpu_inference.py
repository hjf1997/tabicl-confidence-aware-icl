import inspect
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

# Fixed outer query-chunk size for the non-TabICL families (their internal
# batching handles fine-grained memory; the outer chunk just drives progress
# and bounds transfer sizes). The TabICL chunk comes from the profiled estimator.
TABPFN_QUERY_CHUNK = 5000

MODEL_FAMILIES = ("tabicl", "tabpfn", "tabpfn_v25", "tabpfn_v3", "tabdpt")


def _filter_ctor_kwargs(cls, kwargs: dict) -> dict:
    """Drop kwargs the constructor does not accept (version drift across the
    tabpfn 2.x / 8.x lines and tabdpt releases). Keeps everything if the
    signature takes **kwargs."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class _TabDPTAdapter:
    """sklearn-style facade over tabdpt.TabDPTClassifier.

    TabDPT takes its runtime knobs (context_size, n_ensembles, ...) at predict
    time; this adapter binds them at construction so workers can call the same
    fit(X, y) / predict_proba(X) contract as every other family. context_size
    is clamped to the fitted support-set size (TabDPT retrieves its per-query
    context from whatever was passed to fit).
    """

    def __init__(self, device: str, model_path: Optional[str] = None,
                 n_ensembles: int = 8, context_size: int = 2048,
                 seed: int = 42):
        from tabdpt import TabDPTClassifier

        ctor = {"device": device}
        if model_path:
            # Upstream has renamed this across releases; try the known names.
            for key in ("model_path", "path", "checkpoint_path"):
                probe = _filter_ctor_kwargs(TabDPTClassifier, {key: model_path})
                if probe:
                    ctor[key] = model_path
                    break
            else:
                raise ValueError(
                    "This tabdpt version exposes no checkpoint-path argument; "
                    "pre-populate the HuggingFace cache instead of --model-path.")
        self._model = TabDPTClassifier(**_filter_ctor_kwargs(TabDPTClassifier, ctor))
        self._n_ensembles = n_ensembles
        self._context_size = context_size
        self._seed = seed
        self._n_fitted = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_fitted = len(X)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        kwargs = {
            "n_ensembles": self._n_ensembles,
            "context_size": min(self._context_size, self._n_fitted),
            "seed": self._seed,
        }
        # Tolerate signature drift: retry without the kwargs it rejects.
        while True:
            try:
                return self._model.predict_proba(X, **kwargs)
            except TypeError as e:
                dropped = next((k for k in list(kwargs) if k in str(e)), None)
                if dropped is None:
                    if kwargs:
                        kwargs = {}
                        continue
                    raise
                kwargs.pop(dropped)


def _make_classifier(model_family: str, device: str, model_kwargs: dict):
    """Instantiate the in-context classifier for a worker process."""
    if model_family == "tabpfn":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device=device, **model_kwargs)
    if model_family in ("tabpfn_v25", "tabpfn_v3"):
        import tabpfn as _tabpfn_pkg
        from tabpfn import TabPFNClassifier

        _ver = str(getattr(_tabpfn_pkg, "__version__", "0"))
        if int(_ver.split(".")[0] or 0) < 8:
            # Never fall through to the 2.x line: it would silently serve a
            # v2 model under a v2.5/v3 label.
            raise RuntimeError(
                f"model family '{model_family}' needs tabpfn>=8.0.0 "
                f"(installed: {_ver}); run it from the tabpfn-8 conda env.")
        kwargs = _filter_ctor_kwargs(TabPFNClassifier, dict(model_kwargs, device=device))
        if model_family == "tabpfn_v3" or "model_path" in kwargs:
            # v3 is the tabpfn>=8 default; an explicit checkpoint pins the
            # version regardless of family.
            return TabPFNClassifier(**kwargs)
        # v2.5 without an explicit checkpoint: require the version pin rather
        # than silently running whatever the installed default is.
        try:
            from tabpfn.constants import ModelVersion
            version = getattr(ModelVersion, "V2_5", None)
        except ImportError:
            version = None
        if version is not None and hasattr(TabPFNClassifier, "create_default_for_version"):
            return TabPFNClassifier.create_default_for_version(version, **kwargs)
        raise RuntimeError(
            "tabpfn_v25 needs tabpfn>=8.0.0 with ModelVersion.V2_5 "
            "(or pass an explicit v2.5 checkpoint via --model-path).")
    if model_family == "tabdpt":
        return _TabDPTAdapter(device=device, **model_kwargs)
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
            # tabpfn v2 line (pip tabpfn==2.2.1): near-mirror of the TabICL API.
            # fit_with_cache is the kv_cache analog (caches train-side state
            # for repeated predict_proba calls on one fitted context).
            self.model_kwargs = {
                "n_estimators": tabicl_config.n_estimators,
                "softmax_temperature": tabicl_config.softmax_temperature,
                "random_state": tabicl_config.random_state,
                "fit_mode": "fit_with_cache",
                "memory_saving_mode": "auto",
                "ignore_pretraining_limits": True,
                # Our loader ordinal-encodes everything to float64; declare no
                # categoricals so the model cannot auto-infer a different set
                # per family (identical input contract across families).
                "categorical_features_indices": [],
            }
            if tabicl_config.model_path:
                self.model_kwargs["model_path"] = tabicl_config.model_path
        elif self.model_family in ("tabpfn_v25", "tabpfn_v3"):
            # tabpfn>=8.0.0 line; unsupported kwargs are filtered by signature
            # in _make_classifier, so version drift degrades gracefully.
            self.model_kwargs = {
                "n_estimators": tabicl_config.n_estimators,
                "softmax_temperature": tabicl_config.softmax_temperature,
                "random_state": tabicl_config.random_state,
                "ignore_pretraining_limits": True,
                "categorical_features_indices": [],
            }
            if tabicl_config.model_path:
                self.model_kwargs["model_path"] = tabicl_config.model_path
        elif self.model_family == "tabdpt":
            self.model_kwargs = {
                "n_ensembles": tabicl_config.n_estimators,
                "context_size": tabicl_config.tabdpt_context_size,
                "seed": tabicl_config.random_state,
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
        if self.model_family != "tabicl":
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
