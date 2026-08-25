from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "exp" / "20260717_0807_importance" / "consensus_top_features.csv"

DEFAULT_MODEL_PATH = PROJECT_ROOT / "tabicl-main" / "checkpoints" / "tabicl-classifier-v2-20260212.ckpt"

# TabPFN v2 checkpoint (download token-free from
# huggingface.co/Prior-Labs/TabPFN-v2-clf and place it here).
DEFAULT_TABPFN_MODEL_PATH = (PROJECT_ROOT / "tabpfn-checkpoints"
                             / "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt")


@dataclass
class DataConfig:
    setting: str = "setting1"
    label_col: str = "label"
    id_col: str = "ar_case_no"
    positive_class: int = 0  # bogus (reliable)
    negative_class: int = 1  # fraud (noisy)
    top_features: int = 150

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / self.setting

    @property
    def train_file(self) -> str:
        return "tabular_dataset_train.csv"

    @property
    def val_file(self) -> str:
        return "tabular_dataset_validation.csv"

    @property
    def test_file(self) -> str:
        return "tabular_dataset_test.csv"


@dataclass
class TabICLConfig:
    # "tabicl" (default) or "tabpfn" (pip tabpfn==2.2.1, v2 weights from
    # Prior-Labs/TabPFN-v2-clf on HuggingFace — no token needed for v2).
    # Shared knobs (n_estimators, softmax_temperature, random_state,
    # model_path) apply to both; batch_size/kv_cache/verbose are TabICL-only.
    model_family: str = "tabicl"
    model_path: Optional[str] = str(DEFAULT_MODEL_PATH)
    n_estimators: int = 8
    batch_size: int = 8
    softmax_temperature: float = 0.9
    kv_cache: bool = True
    random_state: int = 42
    verbose: bool = False


@dataclass
class MultiGPUConfig:
    num_gpus: int = 4
    devices: List[str] = field(default_factory=lambda: [f"cuda:{i}" for i in range(4)])
    mp_start_method: str = "spawn"


@dataclass
class ReliabilityConfig:
    K: int = 20
    support_set_size: int = 500
    probe_design: str = "random"  # "random" or "anchored" (M fixed bogus anchors x K/M fraud draws)
    n_anchors: int = 4  # anchored only; K must be divisible by n_anchors
    w_mean_prob: float = 0.30
    w_stability: float = 0.20
    w_entropy: float = 0.15
    w_agreement: float = 0.20
    w_density: float = 0.15
    n_neighbors: int = 15
    similarity_metric: str = "cosine"
    threshold_method: str = "percentile"  # "fixed" or "percentile"
    # Optional learned-weights JSON (from run_optimize_score.py). When set, the
    # score is the learned linear combination of z-scored components instead of
    # the fixed mixture below; n_neighbors is taken from the file.
    score_weights_file: Optional[str] = None
    reliable_threshold: float = 0.75  # used when threshold_method="fixed"
    uncertain_threshold: float = 0.45  # used when threshold_method="fixed"
    reliable_percentile: float = 30.0  # top X% classified as reliable
    suspect_percentile: float = 30.0  # bottom X% classified as suspect


@dataclass
class SupportSetConfig:
    target_size: int = None  # None = use all positives + equal negatives
    neg_sampling_strategy: str = "random"


@dataclass
class PipelineConfig:
    max_iterations: int = 5
    convergence_threshold: float = 0.005
    eval_metric: str = "pr_auc"
    output_dir: Path = Path("./exp")
    data: DataConfig = field(default_factory=DataConfig)
    tabicl: TabICLConfig = field(default_factory=TabICLConfig)
    gpu: MultiGPUConfig = field(default_factory=MultiGPUConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    support_set: SupportSetConfig = field(default_factory=SupportSetConfig)
