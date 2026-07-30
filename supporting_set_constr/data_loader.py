import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from config import DataConfig, FEATURE_IMPORTANCE_PATH


def load_selected_features(top_n: int) -> List[str]:
    """Load top N features from the consensus feature importance file."""
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return df["feature"].tolist()[:top_n]


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.selected_features = load_selected_features(config.top_features)

    def load_all(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray, pd.Series]]:
        """Load all splits.

        Returns dict with 'train', 'val', 'test' keys.
        Each value is (X: DataFrame of selected features, y: ndarray, ids: Series of ar_case_no).
        """
        splits = {}
        for name, filename in [
            ("train", self.config.train_file),
            ("val", self.config.val_file),
            ("test", self.config.test_file),
        ]:
            path = self.config.data_dir / filename
            df = pd.read_csv(path)
            y = df[self.config.label_col].values
            ids = df[self.config.id_col]
            X = df[self.selected_features]
            splits[name] = (X, y, ids)
        return splits

    def get_positive_samples(
        self, X: pd.DataFrame, y: np.ndarray, ids: pd.Series
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
        mask = y == self.config.positive_class
        return X[mask].reset_index(drop=True), y[mask], ids[mask].reset_index(drop=True)

    def get_negative_samples(
        self, X: pd.DataFrame, y: np.ndarray, ids: pd.Series
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
        mask = y == self.config.negative_class
        return X[mask].reset_index(drop=True), y[mask], ids[mask].reset_index(drop=True)
