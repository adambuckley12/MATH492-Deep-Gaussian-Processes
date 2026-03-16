"""
data/data_preprocessor.py — Preprocessing for Kin8nm.

All 8 features are already numeric. We just standardise to zero mean, unit variance.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._scaler = StandardScaler()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(
            self._scaler.fit_transform(X),
            columns=X.columns, index=X.index,
        )
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(
            self._scaler.transform(X),
            columns=X.columns, index=X.index,
        )
        return X_out
