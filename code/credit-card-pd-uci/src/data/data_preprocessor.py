"""
data_preprocessor.py
====================
Handles all preprocessing steps after DataCleaner.clean():

  1. Categorical encoding
      - SEX       : binary int (0=male, 1=female) — already ordinal
      - EDUCATION : one-hot (4 levels → drop first to avoid dummy trap)
      - MARRIAGE  : one-hot (3 levels → drop first)
  2. Scaling of numeric features (StandardScaler)
      Includes: LIMIT_BAL, AGE, PAY_SEP–PAY_APR (ordinal ints),
              BILL_SEP–BILL_APR, PAY_AMT_SEP–PAY_AMT_APR
  3. Stratified multi-split protocol
      - train      : model fitting
      - validation : model-selection / early stopping
      - test       : untouched final evaluation
  4. Optional PCA applied to the *scaled continuous* columns only
      (not on the one-hot dummies — PCA on binary indicators is not useful)

Usage
-----
    from data.data_preprocessor import DataPreprocessor

    pre = DataPreprocessor(test_size=0.2, validation_size=0.15,
                           random_state=42)
    splits = pre.fit_transform_splits(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Column name constants (readable names after DataCleaner.clean()) ──────────
_SEX_COL       = "SEX"
_EDU_COL       = "EDUCATION"
_MAR_COL       = "MARRIAGE"
_CATEGORICAL   = [_SEX_COL, _EDU_COL, _MAR_COL]

_MONTHS        = ["SEP", "AUG", "JUL", "JUN", "MAY", "APR"]
_PAY_STATUS    = [f"PAY_{m}" for m in _MONTHS]   # ordinal repayment codes
_BILL_COLS     = [f"BILL_{m}" for m in _MONTHS]
_PAY_AMT_COLS  = [f"PAY_AMT_{m}" for m in _MONTHS]
_NUMERIC       = ["LIMIT_BAL", "AGE"] + _PAY_STATUS + _BILL_COLS + _PAY_AMT_COLS


class DataPreprocessor:
    """
    Fit-transform pipeline for the UCI Credit Default dataset.

    Parameters
    ----------
    test_size : float, default 0.2
        Fraction of data held out as the test set.
    random_state : int, default 42
        RNG seed for reproducibility.
    apply_pca : bool, default False
        If True, apply PCA to the scaled continuous columns after encoding.
    pca_variance : float, default 0.95
        Cumulative explained-variance ratio target when ``apply_pca=True``.
        The number of components is chosen automatically.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.15,
        random_state: int = 42,
        apply_pca: bool = False,
        pca_variance: float = 0.95,
    ):
        self.test_size    = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.apply_pca    = apply_pca
        self.pca_variance = pca_variance

        # Fitted objects (populated by fit / fit_transform_split)
        self._scaler:         StandardScaler | None = None
        self._pca:            PCA            | None = None
        self._encoded_cols:   list[str]             = []   # cols after OHE
        self._numeric_cols:   list[str]             = []   # numeric cols in X
        self._pca_target_cols: list[str]            = []   # cols PCA is applied to
        self.n_components_:   int            | None = None # PCA components used
        self.split_summary_:  dict[str, dict[str, float | int]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fit_transform_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Stratified train/test split, then fit on train set and transform both.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_train = self.fit_transform(X_train_raw)
        X_test  = self.transform(X_test_raw)

        return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    def fit_transform_splits(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
        """
        Build stratified train / validation / test splits.

        The preprocessor is fit on the *train* split only, then applied to the
        remaining splits to avoid leakage.
        """
        raw_splits = self.split_raw(X, y)

        transformed = {}
        X_train_raw, y_train = raw_splits["train"]
        transformed["train"] = (
            self.fit_transform(X_train_raw),
            y_train.reset_index(drop=True),
        )

        for split_name in ["validation", "test"]:
            X_part, y_part = raw_splits[split_name]
            transformed[split_name] = (
                self.transform(X_part),
                y_part.reset_index(drop=True),
            )

        self.split_summary_ = {
            split_name: {
                "rows": int(len(y_part)),
                "default_rate": float(np.asarray(y_part).mean()),
            }
            for split_name, (_, y_part) in transformed.items()
        }
        return transformed

    def split_raw(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
        """
        Create raw stratified train / validation / test splits.
        """
        total_holdout = self.test_size + self.validation_size
        if total_holdout >= 1.0:
            raise ValueError("test_size + validation_size must be < 1.")

        X_dev, X_test, y_dev, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        if self.validation_size > 0:
            val_fraction_of_dev = self.validation_size / (1.0 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_dev,
                y_dev,
                test_size=val_fraction_of_dev,
                random_state=self.random_state,
                stratify=y_dev,
            )
        else:
            X_train, y_train = X_dev, y_dev
            X_val = X_dev.iloc[0:0].copy()
            y_val = y_model.iloc[0:0].copy()

        return {
            "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
            "validation": (X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
            "test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        }

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit all encoders/scalers on *X* and return the transformed DataFrame."""
        X_enc = self._encode_categoricals(X)
        X_enc = self._fit_scale(X_enc)
        if self.apply_pca:
            X_enc = self._fit_pca(X_enc)
        self._encoded_cols = list(X_enc.columns)
        return X_enc

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted encoders/scalers to a new DataFrame."""
        if self._scaler is None:
            raise RuntimeError("Call fit_transform (or fit_transform_split) before transform.")
        X_enc = self._encode_categoricals(X)
        # Align columns to match training set (missing dummies → 0)
        X_enc = X_enc.reindex(columns=self._encoded_cols, fill_value=0)
        X_enc = self._apply_scale(X_enc)
        if self.apply_pca:
            X_enc = self._apply_pca(X_enc)
        return X_enc

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns in-place:

        * SEX        : 1/2  →  0/1  (0 = male, 1 = female)
        * EDUCATION  : one-hot,  drop_first=True  → EDUCATION_2, _3, _4
        * MARRIAGE   : one-hot,  drop_first=True  → MARRIAGE_2, _3
        """
        X = X.copy()

        # SEX: remap 1→0, 2→1
        if _SEX_COL in X.columns:
            X[_SEX_COL] = X[_SEX_COL].map({1: 0, 2: 1}).astype(int)

        # EDUCATION & MARRIAGE: one-hot
        ohe_cols = [c for c in [_EDU_COL, _MAR_COL] if c in X.columns]
        if ohe_cols:
            X = pd.get_dummies(X, columns=ohe_cols, drop_first=True, dtype=int)

        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Scaling
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit a StandardScaler on numeric columns and scale them."""
        num_cols = [c for c in _NUMERIC if c in X.columns]
        self._numeric_cols = num_cols
        self._scaler = StandardScaler()
        X = X.copy()
        X[num_cols] = self._scaler.fit_transform(X[num_cols].astype(float))
        return X

    def _apply_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted StandardScaler."""
        num_cols = self._numeric_cols
        X = X.copy()
        X[num_cols] = self._scaler.transform(X[num_cols].astype(float))
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # PCA  (optional — continuous columns only)
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA on the scaled numeric columns, retaining enough components to
        explain ``pca_variance`` of the variance.  One-hot columns are passed
        through unchanged.
        """
        num_cols = self._numeric_cols
        self._pca_target_cols = num_cols

        self._pca = PCA(n_components=self.pca_variance, svd_solver="full",
                        random_state=self.random_state)
        pca_values = self._pca.fit_transform(X[num_cols].values)
        self.n_components_ = self._pca.n_components_

        # Build output: keep non-PCA columns, append PC columns
        non_pca_cols = [c for c in X.columns if c not in num_cols]
        pc_names = [f"PC{i+1}" for i in range(self.n_components_)]
        pc_df    = pd.DataFrame(pca_values, columns=pc_names, index=X.index)

        return pd.concat([X[non_pca_cols].reset_index(drop=True),
                          pc_df.reset_index(drop=True)], axis=1)

    def _apply_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted PCA."""
        num_cols     = self._pca_target_cols
        non_pca_cols = [c for c in X.columns if c not in num_cols]
        pc_names     = [f"PC{i+1}" for i in range(self.n_components_)]
        pca_values   = self._pca.transform(X[num_cols].values)
        pc_df        = pd.DataFrame(pca_values, columns=pc_names, index=X.index)
        return pd.concat([X[non_pca_cols].reset_index(drop=True),
                          pc_df.reset_index(drop=True)], axis=1)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def explained_variance_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame showing how many PCA components are needed to reach
        common explained-variance thresholds (80 %, 90 %, 95 %, 99 %).
        Only available after ``apply_pca=True`` and fitting.
        """
        if self._pca is None:
            raise RuntimeError("PCA has not been fitted yet.")
        cum_var = np.cumsum(self._pca.explained_variance_ratio_)
        rows = []
        for k, ratio in enumerate(cum_var, start=1):
            rows.append({"n_components": k,
                         "cumulative_variance": round(ratio, 4)})
        df = pd.DataFrame(rows)
        milestones = {}
        for thresh in [0.80, 0.90, 0.95, 0.99]:
            idx = np.searchsorted(cum_var, thresh)
            milestones[f">= {int(thresh*100)}%"] = int(idx) + 1
        print("Components needed per variance threshold:")
        for k, v in milestones.items():
            print(f"  {k} : {v} components")
        return df

    def feature_names_out(self) -> list[str]:
        """Return column names of the last transformed output."""
        if self._pca is not None:
            num_cols     = self._pca_target_cols
            non_pca_cols = [c for c in self._encoded_cols if c not in num_cols]
            return non_pca_cols + [f"PC{i+1}" for i in range(self.n_components_)]
        return self._encoded_cols

    def split_summary(self) -> pd.DataFrame:
        """Return row counts and default rates for the latest multi-split run."""
        if not self.split_summary_:
            raise RuntimeError("No multi-split summary is available yet.")
        df = pd.DataFrame(self.split_summary_).T
        df.index.name = "split"
        return df.reset_index()
