"""
data_cleaner.py
===============
Feature engineering and cleaning for the Freddie Mac mortgage dataset.

Cleaning steps
--------------
1. Handle missing values (median imputation for numeric, mode for categorical).
2. Engineer derived features:
   - ``vintage_era``: pre-crisis / crisis / post-crisis / covid
   - ``refi_bucket``: in / at / out of the money
   - ``region``: US Census region from property state
   - ``ltv_bucket``: low / medium / high
   - ``rate_spread``: loan rate minus cohort median
3. Log-transform skewed monetary columns (``orig_upb``).
4. Drop identifier and redundant columns.

The output is a feature DataFrame ``X`` and binary target ``y`` suitable
for the shared preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# US Census regions (duplicated here for standalone use)
STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
    "NH": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast", "RI": "Northeast", "VT": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest",
    "KS": "Midwest", "MI": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest",
    "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",
    "OH": "Midwest", "WI": "Midwest",
    "AL": "South", "AR": "South", "DE": "South", "FL": "South",
    "GA": "South", "KY": "South", "LA": "South", "MD": "South",
    "MS": "South", "NC": "South", "OK": "South", "SC": "South",
    "TN": "South", "TX": "South", "VA": "South", "WV": "South",
    "DC": "South",
    "AK": "West", "AZ": "West", "CA": "West", "CO": "West",
    "HI": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "OR": "West", "UT": "West", "WA": "West",
    "WY": "West",
}


class DataCleaner:
    """
    Clean and engineer features for Freddie Mac loan-level data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw merged DataFrame from ``FreddieMacLoader.load()``.
        Must contain a ``default`` column (binary target).
    """

    # Columns expected in the final feature set
    NUMERIC_FEATURES = [
        "credit_score", "orig_ltv", "orig_dti", "orig_rate",
        "orig_upb", "mi_pct", "num_borrowers", "num_units",
        "orig_term", "rate_spread", "orig_year",
    ]
    CATEGORICAL_FEATURES = [
        "loan_purpose", "property_type", "occupancy", "channel",
        "first_time_buyer", "vintage_era", "refi_bucket",
        "ltv_bucket", "region",
    ]

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def clean(self, apply_log_transforms: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """
        Run the full cleaning pipeline.

        Returns
        -------
        X : pd.DataFrame   — feature matrix
        y : pd.Series       — binary default target
        """
        df = self.data.copy()

        # ── 1. Extract target ─────────────────────────────────────────────────
        assert "default" in df.columns, "Expected 'default' column"
        y = df["default"].astype(int).rename("default")
        df = df.drop(columns=["default"])

        # ── 2. Ensure derived features exist ──────────────────────────────────
        # These are normally computed by FreddieMacLoader._engineer_strata,
        # but guard against missing columns defensively.
        if "vintage_era" not in df.columns:
            df["vintage_era"] = self._compute_vintage_era(df)
        if "region" not in df.columns:
            df["region"] = df["property_state"].map(STATE_TO_REGION).fillna("South")
        if "rate_spread" not in df.columns:
            cohort_med = df.groupby("orig_year")["orig_rate"].transform("median")
            df["rate_spread"] = df["orig_rate"] - cohort_med
        if "refi_bucket" not in df.columns:
            df["refi_bucket"] = df["rate_spread"].apply(self._refi_bucket)
        if "ltv_bucket" not in df.columns:
            df["ltv_bucket"] = pd.cut(
                df["orig_ltv"], bins=[0, 60, 80, 200],
                labels=["low", "medium", "high"],
            ).astype(str)
        if "orig_year" not in df.columns:
            df["orig_year"] = pd.to_numeric(
                df.get("first_payment_date", pd.Series(dtype=str)).astype(str).str[:4],
                errors="coerce",
            ).fillna(2015).astype(int)

        # ── 3. Impute missing values ─────────────────────────────────────────
        for col in self.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")

        # ── 4. Log-transform skewed monetary columns ──────────────────────────
        if apply_log_transforms:
            if "orig_upb" in df.columns:
                df["orig_upb"] = np.log1p(df["orig_upb"].clip(lower=0))

        # ── 5. Select feature columns ─────────────────────────────────────────
        keep_cols = [c for c in self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES if c in df.columns]
        X = df[keep_cols].copy()

        return X, y.reset_index(drop=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_vintage_era(df: pd.DataFrame) -> pd.Series:
        """Assign vintage era from origination date."""
        if "orig_year" in df.columns:
            year = df["orig_year"]
        elif "first_payment_date" in df.columns:
            year = pd.to_numeric(
                df["first_payment_date"].astype(str).str[:4], errors="coerce"
            ).fillna(2015)
        else:
            return pd.Series("post_crisis", index=df.index)
        return year.apply(
            lambda y: "pre_crisis" if y < 2007
            else "crisis" if y <= 2009
            else "covid" if y >= 2020
            else "post_crisis"
        )

    @staticmethod
    def _refi_bucket(spread: float) -> str:
        if spread > 1.0:
            return "in_the_money"
        elif spread < -1.0:
            return "out_of_the_money"
        return "at_the_money"

    # ── Column-type helpers (for preprocessing) ───────────────────────────────

    @staticmethod
    def categorical_cols(X: pd.DataFrame) -> list[str]:
        return [c for c in DataCleaner.CATEGORICAL_FEATURES if c in X.columns]

    @staticmethod
    def continuous_cols(X: pd.DataFrame) -> list[str]:
        cat = set(DataCleaner.categorical_cols(X))
        return [c for c in X.columns if c not in cat]
