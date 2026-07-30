from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    data_dir: Path = Path("./data")
    train_file: str = "training.csv"
    val_file: str = "validation.csv"
    test_file: str = "test.csv"
    label_col: str = "label"
    positive_class: int = 0  # bogus (reliable)
    negative_class: int = 1  # fraud (noisy)


@dataclass
class TabICLConfig:
    model_path: Optional[str] = None  # local checkpoint path (offline env)
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
    w_mean_prob: float = 0.30
    w_stability: float = 0.20
    w_entropy: float = 0.15
    w_agreement: float = 0.20
    w_density: float = 0.15
    n_neighbors: int = 15
    similarity_metric: str = "cosine"
    reliable_threshold: float = 0.75
    uncertain_threshold: float = 0.45


@dataclass
class SupportSetConfig:
    target_size: int = 1000
    positive_ratio: float = 0.50
    negative_reliable_ratio: float = 0.375  # 375 out of 1000
    negative_boundary_ratio: float = 0.125  # 125 out of 1000
    diversity_lambda: float = 0.3


@dataclass
class PipelineConfig:
    max_iterations: int = 5
    convergence_threshold: float = 0.005
    eval_metric: str = "pr_auc"
    output_dir: Path = Path("./output")
    data: DataConfig = field(default_factory=DataConfig)
    tabicl: TabICLConfig = field(default_factory=TabICLConfig)
    gpu: MultiGPUConfig = field(default_factory=MultiGPUConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    support_set: SupportSetConfig = field(default_factory=SupportSetConfig)
