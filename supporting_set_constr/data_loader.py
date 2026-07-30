import pandas as pd
import numpy as np
from typing import Dict, Tuple

from config import DataConfig


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config

    def load_all(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        splits = {}
        for name, filename in [
            ("train", self.config.train_file),
            ("val", self.config.val_file),
            ("test", self.config.test_file),
        ]:
            path = self.config.data_dir / filename
            df = pd.read_csv(path)
            y = df[self.config.label_col].values
            X = df.drop(columns=[self.config.label_col])
            splits[name] = (X, y)
        return splits

    def get_positive_samples(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        mask = y == self.config.positive_class
        return X[mask].reset_index(drop=True), y[mask]

    def get_negative_samples(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        mask = y == self.config.negative_class
        return X[mask].reset_index(drop=True), y[mask]
