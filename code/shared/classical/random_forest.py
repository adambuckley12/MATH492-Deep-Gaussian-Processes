"""
classical/random_forest.py
===========================
Random Forest classifier for credit default prediction.

Design choices
--------------
* ``class_weight='balanced_subsample'``: re-weights each bootstrap sample
  independently — generally better than ``'balanced'`` for forests.
* 500 trees: good bias-variance trade-off; diminishing returns beyond ~300.
* ``max_features='sqrt'``: standard for classification forests.
* ``n_jobs=-1``: uses all available CPU cores.
* ``oob_score=True``: out-of-bag ROC-AUC estimate at no extra cost.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classical.metrics import evaluate_classifier


class RandomForestModel:
    """
    Thin wrapper around sklearn RandomForestClassifier.

    Parameters
    ----------
    n_estimators : int, default 500
    max_depth    : int | None, default None (grow until pure leaves)
    min_samples_leaf : int, default 4
    random_state : int, default 42
    """

    name = "Random Forest"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 4,
        random_state: int = 42,
        class_weight: str | dict | None = "balanced_subsample",
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight=class_weight,
            oob_score=True,
            n_jobs=-1,
            random_state=random_state,
        )

    # Public API

    def fit(self, X_train: pd.DataFrame | np.ndarray,
            y_train: pd.Series  | np.ndarray) -> "RandomForestModel":
        self.model.fit(X_train, y_train)
        print(f"  [{self.name}] OOB score: {self.model.oob_score_:.4f}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        verbose: bool = True,
    ) -> dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        return evaluate_classifier(self.name, np.asarray(y_test),
                                   y_pred, y_prob, verbose=verbose)

    # Feature importance

    def feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Return mean decrease in impurity (Gini importance) sorted descending.
        """
        imp = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Persistence

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* using joblib."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        """Deserialise a previously saved model from *path*."""
        import joblib
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        print(f"  [{cls.name}] weights loaded ← {path}")
        return obj
