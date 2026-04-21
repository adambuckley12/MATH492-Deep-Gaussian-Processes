"""
classical/xgboost_model.py
===========================
XGBoost gradient-boosted tree classifier for credit default prediction.

Design choices
--------------
* ``scale_pos_weight``: set to ``(n_negative / n_positive)`` at fit-time to
  handle class imbalance — XGBoost's native equivalent of class_weight.
* ``eval_metric='aucpr'``: optimises average precision during early stopping,
  which is more informative than AUC under imbalance.
* ``early_stopping_rounds=50``: stops training when AUC-PR on a validation
  split stops improving, preventing overfitting without manual tuning.
* ``tree_method='hist'``: fast histogram-based algorithm (same as LightGBM).
* Sensible defaults for ``n_estimators``, ``max_depth``, ``learning_rate``,
  ``subsample``, ``colsample_bytree`` — good starting point before any HPO.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Run: pip install xgboost"
    ) from e

from classical.metrics import evaluate_classifier


class XGBoostModel:
    """
    Thin wrapper around xgboost.XGBClassifier.

    Parameters
    ----------
    n_estimators    : int, default 1000
        Maximum number of boosting rounds (early stopping may terminate sooner).
    max_depth       : int, default 6
    learning_rate   : float, default 0.05
    subsample       : float, default 0.8
    colsample_bytree: float, default 0.8
    early_stopping_rounds : int, default 50
    random_state    : int, default 42
    """

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        use_scale_pos_weight: bool = True,
    ):
        self._early_stopping_rounds = early_stopping_rounds
        self._use_scale_pos_weight = use_scale_pos_weight
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=random_state,
            verbosity=0,
        )

    # Public API

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val:   pd.DataFrame | np.ndarray | None = None,
        y_val:   pd.Series    | np.ndarray | None = None,
    ) -> "XGBoostModel":
        """
        Fit the model.  If ``X_val`` / ``y_val`` are provided, early stopping
        is enabled using AUC-PR on the validation set.  Otherwise the model
        trains for the full ``n_estimators`` rounds.

        ``scale_pos_weight`` is computed from ``y_train`` automatically.
        """
        y_arr = np.asarray(y_train)
        n_neg = int((y_arr == 0).sum())
        n_pos = int((y_arr == 1).sum())
        if self._use_scale_pos_weight:
            self.model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
        else:
            self.model.set_params(scale_pos_weight=1.0)

        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=self._early_stopping_rounds)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            print(f"  [{self.name}] Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train, verbose=False)

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
        Return gain-based feature importance sorted descending.
        Gain is the average improvement in loss brought by a feature's splits.
        """
        imp = self.model.feature_importances_
        names = (feature_names
                 or list(self.model.get_booster().feature_names)
                 or [f"f{i}" for i in range(len(imp))])
        df = pd.DataFrame({"feature": names, "gain": imp})
        return df.sort_values("gain", ascending=False).reset_index(drop=True)

    # Persistence

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* using joblib."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """Deserialise a previously saved model from *path*."""
        import joblib
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        print(f"  [{cls.name}] weights loaded ← {path}")
        return obj
