"""
classical/logistic_regression.py
=================================
Logistic Regression baseline for credit default prediction.

Design choices
--------------
* ``class_weight='balanced'``: compensates for the ~22%/78% class imbalance
  by up-weighting the minority (default) class during training.
* Solver ``lbfgs`` with L2 regularisation: fast, numerically stable, works
  well out-of-the-box on tabular data.
* ``max_iter=1000``: UCI credit dataset converges well within this budget.
* Threshold 0.5 for hard predictions (can be tuned via precision-recall curve).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from classical.metrics import evaluate_classifier


class LogisticRegressionModel:
    """
    Thin wrapper around sklearn LogisticRegression with a consistent interface
    shared by all classical models in this project.

    Parameters
    ----------
    C : float, default 1.0
        Inverse regularisation strength (smaller = stronger regularisation).
    max_iter : int, default 1000
    random_state : int, default 42
    """

    name = "Logistic Regression"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str | dict | None = "balanced",
    ):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=random_state,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame | np.ndarray,
            y_train: pd.Series  | np.ndarray) -> "LogisticRegressionModel":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probability of the positive class (default=1)."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """Run predictions and return a metrics dict."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        return evaluate_classifier(self.name, np.asarray(y_test),
                                   y_pred, y_prob, verbose=verbose)

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Return a DataFrame of |coefficient| values sorted descending.
        For a standardised feature matrix these are directly comparable.
        """
        coef = np.abs(self.model.coef_[0])
        names = feature_names or [f"f{i}" for i in range(len(coef))]
        df = pd.DataFrame({"feature": names, "abs_coef": coef})
        return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* using joblib."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        """Deserialise a previously saved model from *path*."""
        import joblib
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        print(f"  [{cls.name}] weights loaded ← {path}")
        return obj
