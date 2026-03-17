"""
classical/xgboost_reg.py
=========================
XGBoost gradient-boosted tree regressor.

Design choices
--------------
* Early stopping on a validation set to find the optimal number of rounds,
  then refit on the full train+val set using that round count.
* ``objective='reg:squarederror'``: standard squared-error loss for regression.
* ``tree_method='hist'``: fast histogram-based algorithm.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from base import BaseModel


class XGBoostRegressionModel(BaseModel):
    """
    XGBoost regressor with automatic early-stopping and full-data refit.

    Call ``fit(X_train, y_train, X_val, y_val)`` to tune the number of rounds
    on a held-out validation set and then refit on train+val.  If no validation
    data is provided, trains for the full ``n_estimators`` rounds.

    Parameters
    ----------
    n_estimators : int, default 1000
        Maximum boosting rounds (early stopping may terminate sooner).
    learning_rate : float, default 0.05
    max_depth : int, default 6
    subsample : float, default 0.8
    colsample_bytree : float, default 0.8
    early_stopping_rounds : int, default 20
    random_state : int, default 42
    """

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ) -> None:
        self._params = {
            "objective": "reg:squarederror",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "seed": random_state,
        }
        self._n_estimators = n_estimators
        self._early_stopping_rounds = early_stopping_rounds
        self._booster: xgb.Booster | None = None
        self._feature_names: list[str] = []

    # Public API

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "XGBoostRegressionModel":
        """
        If ``X_val`` / ``y_val`` are provided:
          1. Train with early stopping to find the best round count.
          2. Refit on the combined train+val set for that many rounds.
        Otherwise train for the full ``n_estimators`` rounds.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        dtrain = xgb.DMatrix(X, label=y)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            tuner = xgb.train(
                self._params,
                dtrain,
                num_boost_round=self._n_estimators,
                evals=[(dval, "val")],
                early_stopping_rounds=self._early_stopping_rounds,
                verbose_eval=False,
            )
            best_rounds = max(tuner.best_iteration + 1, 10)
            print(f"  [{self.name}] Best iteration: {best_rounds}")

            X_full = (
                pd.concat([X, X_val]).reset_index(drop=True)
                if isinstance(X, pd.DataFrame)
                else np.concatenate([X, X_val])
            )
            y_full = (
                pd.concat([y, y_val]).reset_index(drop=True)
                if isinstance(y, pd.Series)
                else np.concatenate([y, y_val])
            )
            dfull = xgb.DMatrix(X_full, label=y_full)
            self._booster = xgb.train(
                self._params, dfull, num_boost_round=best_rounds, verbose_eval=False,
            )
        else:
            self._booster = xgb.train(
                self._params, dtrain,
                num_boost_round=self._n_estimators, verbose_eval=False,
            )

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._booster.predict(xgb.DMatrix(X))

    def feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """Return gain-based feature importance."""
        names = feature_names or self._feature_names
        importance = self._booster.get_score(importance_type="gain")
        gains = [importance.get(f, 0.0) for f in names] if names else list(importance.values())
        if not names:
            names = list(importance.keys())
        df = pd.DataFrame({"feature": names, "gain": gains})
        return df.sort_values("gain", ascending=False).reset_index(drop=True)

    # Persistence

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"booster": self._booster, "feature_names": self._feature_names}, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostRegressionModel":
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj._booster = data["booster"]
        obj._feature_names = data["feature_names"]
        return obj
