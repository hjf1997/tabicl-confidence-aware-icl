import logging
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from typing import Dict, List, Tuple

from config import DataConfig, FEATURE_IMPORTANCE_PATH

logger = logging.getLogger(__name__)

# Value used for categories not seen in train, and for missing values.
UNSEEN_CODE = -1

# Categorical columns above this cardinality get a warning: ordinal codes
# impose a false ordering that is more harmful the more levels there are.
HIGH_CARDINALITY_WARN = 50


def load_selected_features(top_n: int) -> List[str]:
    """Load top N features from the consensus feature importance file."""
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return df["feature"].tolist()[:top_n]


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.selected_features = load_selected_features(config.top_features)
        # Populated by load_all(): {column: {category: code}}, fit on train only.
        self.categorical_encodings_: Dict[str, Dict] = {}

    def load_all(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray, pd.Series]]:
        """Load all splits.

        Returns dict with 'train', 'val', 'test' keys.
        Each value is (X: DataFrame of selected features, y: ndarray, ids: Series of ar_case_no).

        Non-numeric columns are ordinal-encoded so that X.values is float64.
        TabICL rejects object-dtype arrays; the encoding is fit on train only and
        applied unchanged to val/test, with unseen categories and NaN mapped to -1.
        """
        raw = {}
        for name, filename in [
            ("train", self.config.train_file),
            ("val", self.config.val_file),
            ("test", self.config.test_file),
        ]:
            path = self.config.data_dir / filename
            df = pd.read_csv(path)
            raw[name] = (
                df[self.selected_features].copy(),
                df[self.config.label_col].values,
                df[self.config.id_col],
            )

        X_train = raw["train"][0]
        self.categorical_encodings_ = self._fit_categorical_encodings(X_train)

        splits = {}
        for name, (X, y, ids) in raw.items():
            splits[name] = (self._apply_categorical_encodings(X, name), y, ids)
        return splits

    def _fit_categorical_encodings(self, X_train: pd.DataFrame) -> Dict[str, Dict]:
        """Build {column: {category: code}} for every non-numeric column, using train only."""
        encodings = {}
        for col in X_train.columns:
            if is_numeric_dtype(X_train[col]):
                continue
            categories = sorted(X_train[col].dropna().unique(), key=str)
            encodings[col] = {cat: code for code, cat in enumerate(categories)}

        if encodings:
            logger.info("Ordinal-encoding %d non-numeric column(s): %s",
                        len(encodings), ", ".join(sorted(encodings)))
            for col, mapping in sorted(encodings.items()):
                if len(mapping) > HIGH_CARDINALITY_WARN:
                    logger.warning(
                        "Column '%s' has %d categories; ordinal codes impose a false "
                        "ordering on high-cardinality nominal features.",
                        col, len(mapping),
                    )
        return encodings

    def _apply_categorical_encodings(self, X: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Apply the train-fit encodings and cast to float64."""
        for col, mapping in self.categorical_encodings_.items():
            encoded = X[col].map(mapping)
            n_unseen = int(encoded.isna().sum() - X[col].isna().sum())
            if n_unseen > 0:
                logger.warning(
                    "Split '%s' column '%s': %d value(s) not present in train, mapped to %d.",
                    split_name, col, n_unseen, UNSEEN_CODE,
                )
            X[col] = encoded.fillna(UNSEEN_CODE)
        return X.astype(np.float64)

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
