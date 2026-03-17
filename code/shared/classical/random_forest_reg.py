"""
classical/random_forest_reg.py
================================
Random Forest regressor baseline.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from base import BaseModel


class RandomForestRegressionModel(BaseModel):
    """
    Thin wrapper around ``sklearn.ensemble.RandomForestRegressor``.

    Parameters
    ----------
    n_estimators : int, default 500
    random_state : int, default 42
    """

    name = "Random Forest"

    def __init__(
        self,
        n_estimators: int = 500,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )

    # Public API

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "RandomForestRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """Return mean decrease in impurity (Gini importance)."""
        imp = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Persistence

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str) -> "RandomForestRegressionModel":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj
