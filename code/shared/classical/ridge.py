"""
classical/ridge.py
==================
Ridge regression baseline for regression benchmarks.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from base import BaseModel


class RidgeRegressionModel(BaseModel):
    """
    Thin wrapper around ``sklearn.linear_model.Ridge``.

    Parameters
    ----------
    alpha : float, default 1.0
        Regularisation strength. Larger values shrink coefficients more.
    """

    name = "Ridge Regression"

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)

    # Public API

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "RidgeRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """Return absolute coefficient values as a feature-importance proxy."""
        coef = np.abs(self.model.coef_)
        names = feature_names or [f"f{i}" for i in range(len(coef))]
        df = pd.DataFrame({"feature": names, "abs_coef": coef})
        return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    # Persistence

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str) -> "RidgeRegressionModel":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj
