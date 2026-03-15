"""
data_preprocessor.py
====================
Preprocessing for the Freddie Mac mortgage dataset.

Steps
-----
1. One-hot encode categorical features (loan_purpose, property_type,
   occupancy, channel, first_time_buyer, vintage_era, refi_bucket,
   ltv_bucket, region).  ``drop_first=True`` avoids the dummy-variable trap.
2. Standard-scale all numeric features (credit_score, orig_ltv, orig_dti,
   orig_rate, orig_upb, mi_pct, rate_spread, num_borrowers, num_units,
   orig_term, orig_year).
3. Stratified multi-split protocol producing ``train / validation /
   test`` sets.  The scaler is fit on the *train* split only
   to prevent leakage.

Usage
-----
    from data.data_preprocessor import DataPreprocessor

    pre = DataPreprocessor()
    splits = pre.fit_transform_splits(X, y)
    X_train, y_train = splits["train"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.data_cleaner import DataCleaner


class DataPreprocessor:
    """
    Fit-transform pipeline for the Freddie Mac mortgage dataset.

    Parameters
    ----------
    test_size : float
        Fraction of data held out as the final test set.
    validation_size : float
        Fraction reserved for early-stopping / model selection.
    random_state : int
        RNG seed.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.15,
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

        self._scaler: StandardScaler | None = None
        self._encoded_cols: list[str] = []
        self._numeric_cols: list[str] = []
        self.split_summary_: dict[str, dict[str, float | int]] = {}

    # -- Public API ------------------------------------------------------------

    def fit_transform_splits(
        self, X: pd.DataFrame, y: pd.Series,
    ) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
        """
        Build stratified train / validation / test splits.

        The scaler is fit on *train* only, then applied to remaining splits.
        """
        raw = self._split_raw(X, y)

        transformed: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
        X_train_raw, y_train = raw["train"]
        transformed["train"] = (
            self.fit_transform(X_train_raw),
            y_train.reset_index(drop=True),
        )

        for name in ["validation", "test"]:
            X_part, y_part = raw[name]
            transformed[name] = (
                self.transform(X_part),
                y_part.reset_index(drop=True),
            )

        self.split_summary_ = {
            name: {
                "rows": int(len(yp)),
                "default_rate": float(np.asarray(yp).mean()),
            }
            for name, (_, yp) in transformed.items()
        }
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on X, encode categoricals, and return transformed DataFrame."""
        X_enc = self._encode_categoricals(X)
        X_enc = self._fit_scale(X_enc)
        return X_enc

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted encoding + scaling to new data."""
        if self._scaler is None:
            raise RuntimeError("Call fit_transform before transform.")
        X_enc = self._encode_categoricals(X)
        X_enc = self._apply_scale(X_enc)
        return X_enc

    # -- Splitting -------------------------------------------------------------

    def _split_raw(
        self, X: pd.DataFrame, y: pd.Series,
    ) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
        total_holdout = self.test_size + self.validation_size
        if total_holdout >= 1.0:
            raise ValueError("test + validation must sum < 1.")

        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y,
        )

        if self.validation_size > 0:
            val_frac = self.validation_size / (1.0 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_dev, y_dev, test_size=val_frac,
                random_state=self.random_state, stratify=y_dev,
            )
        else:
            X_train, y_train = X_dev, y_dev
            X_val = X_dev.iloc[0:0].copy()
            y_val = y_dev.iloc[0:0].copy()

        return {
            "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
            "validation": (X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
            "test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        }

    # -- Encoding --------------------------------------------------------------

    @staticmethod
    def _encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns, drop_first to avoid collinearity."""
        X = X.copy()
        cat_cols = DataCleaner.categorical_cols(X)
        # Ensure all categorical columns are string type for get_dummies
        for col in cat_cols:
            X[col] = X[col].astype(str)
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
        return X

    # -- Scaling ---------------------------------------------------------------

    def _fit_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Identify numeric columns, fit StandardScaler, transform."""
        cat_dummies = [c for c in X.columns if any(
            c.startswith(p + "_") for p in DataCleaner.CATEGORICAL_FEATURES
        )]
        self._numeric_cols = [c for c in X.columns if c not in cat_dummies]
        self._encoded_cols = list(X.columns)
        self._scaler = StandardScaler()
        X_out = X.copy()
        if self._numeric_cols:
            X_out[self._numeric_cols] = self._scaler.fit_transform(X[self._numeric_cols])
        return X_out

    def _apply_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler. Handles unseen dummy columns gracefully."""
        X_out = X.reindex(columns=self._encoded_cols, fill_value=0).copy()
        if self._numeric_cols:
            X_out[self._numeric_cols] = self._scaler.transform(X_out[self._numeric_cols])
        return X_out
