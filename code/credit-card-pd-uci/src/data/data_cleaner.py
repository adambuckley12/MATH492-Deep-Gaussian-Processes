import numpy as np
import pandas as pd

# -- Column name constants (ucimlrepo uses X1–X23) ----------------------------
COL_LIMIT_BAL = "X1"   # Credit limit (NT dollar)
COL_SEX       = "X2"   # Gender (1=male, 2=female)
COL_EDUCATION = "X3"   # Education level (1=grad, 2=uni, 3=high school, 4=other)
COL_MARRIAGE  = "X4"   # Marital status (1=married, 2=single, 3=other)
COL_AGE       = "X5"   # Age (years)

# Repayment status Sep–Apr 2005 (X6=Sep … X11=Apr)
COLS_PAY  = ["X6", "X7", "X8", "X9", "X10", "X11"]
# Bill statement amounts Sep–Apr 2005 (X12=Sep … X17=Apr)
COLS_BILL = [f"X{i}" for i in range(12, 18)]
# Previous payment amounts Sep–Apr 2005 (X18=Sep … X23=Apr)
COLS_PAY_AMT = [f"X{i}" for i in range(18, 24)]

# Categorical feature columns (treated as nominal, not ordinal)
CATEGORICAL_COLS = [COL_SEX, COL_EDUCATION, COL_MARRIAGE]

# Readable rename mapping applied inside clean()
RENAME = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6":  "PAY_SEP",  "X7":  "PAY_AUG",  "X8":  "PAY_JUL",
    "X9":  "PAY_JUN",  "X10": "PAY_MAY",   "X11": "PAY_APR",
    "X12": "BILL_SEP", "X13": "BILL_AUG",  "X14": "BILL_JUL",
    "X15": "BILL_JUN", "X16": "BILL_MAY",  "X17": "BILL_APR",
    "X18": "PAY_AMT_SEP", "X19": "PAY_AMT_AUG", "X20": "PAY_AMT_JUL",
    "X21": "PAY_AMT_JUN", "X22": "PAY_AMT_MAY", "X23": "PAY_AMT_APR",
}


class DataCleaner:
    """
    Cleans the UCI Default of Credit Card Clients dataset.

    The raw DataFrame is expected to come from DataLoader, which fetches it via
    ucimlrepo.  That package returns feature columns named X1–X23 and a target
    column named ``y`` (default.payment.next.month: 1 = default, 0 = no default).

    Cleaning steps performed by :meth:`clean`:

    1. Drop the row-ID column if present.
    2. Assert no missing values exist.
    3. Collapse undocumented EDUCATION codes 0, 5, 6 → 4 (Other).
    4. Collapse undocumented MARRIAGE code 0 → 3 (Other).
    5. Rename X1–X23 to human-readable names (see ``RENAME``).
    6. Separate features *X* and target *y*.
    7. Optionally apply log1p-based transforms to skewed monetary columns.

    Parameters
    ----------
    data : pd.DataFrame
        Raw dataframe from DataLoader.  Must contain columns X1–X23 and ``y``.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    # -- Public API ------------------------------------------------------------

    def clean(self, apply_log_transforms: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        """
        Run the full cleaning pipeline.

        Parameters
        ----------
        apply_log_transforms : bool, default False
            When True, apply log1p-based transforms to skewed monetary columns
            before returning:

            * ``LIMIT_BAL``, ``PAY_AMT_*``  →  ``log1p(clip(x, lower=0))``
            * ``BILL_*`` (can be negative)   →  ``sign(x) · log1p(|x|)``

        Returns
        -------
        X : pd.DataFrame
            Feature matrix with readable column names.
        y : pd.Series
            Binary target (1 = default next month, 0 = no default).
        """
        df = self.data.copy()

        # 1. Drop ID column (ucimlrepo sometimes prepends an 'ID' column)
        df = self._drop_id_column(df)

        # 2. Assert no missing values
        missing = df.isnull().sum().sum()
        assert missing == 0, (
            f"Data contains {missing} missing value(s). "
            "Handle them before cleaning."
        )

        # 3. Split target before further column ops
        assert "y" in df.columns, "Expected a 'y' target column in the dataframe."
        y = df["y"].squeeze().astype(int).rename("default")
        df = df.drop(columns=["y"])

        # 4. Fix undocumented categorical codes
        df = self._fix_education(df)
        df = self._fix_marriage(df)

        # 5. Rename X1–X23 to readable names
        df = df.rename(columns=RENAME)

        # 6. Optionally transform skewed monetary columns
        if apply_log_transforms:
            df = self._apply_log_transforms(df)

        return df, y

    # -- Column-type helpers ---------------------------------------------------

    @staticmethod
    def categorical_cols(X: pd.DataFrame) -> list[str]:
        """Return the readable names of categorical feature columns present in X."""
        readable = [RENAME.get(c, c) for c in CATEGORICAL_COLS]
        return [c for c in readable if c in X.columns]

    @staticmethod
    def continuous_cols(X: pd.DataFrame) -> list[str]:
        """Return the readable names of continuous feature columns present in X."""
        cat = DataCleaner.categorical_cols(X)
        return [c for c in X.columns if c not in cat]

    # -- Private helpers -------------------------------------------------------

    @staticmethod
    def _drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
        """Drop an 'ID' column if present (ucimlrepo may include one)."""
        id_candidates = [c for c in df.columns if c.strip().upper() == "ID"]
        if id_candidates:
            df = df.drop(columns=id_candidates)
        return df

    @staticmethod
    def _fix_education(df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse undocumented EDUCATION codes into the 'Other' category (4).

        The dataset documentation only defines values 1–4.  Codes 0, 5, and 6
        appear in the data but are not defined; they are treated as 'Other'.
        """
        df[COL_EDUCATION] = df[COL_EDUCATION].replace({0: 4, 5: 4, 6: 4})
        return df

    @staticmethod
    def _fix_marriage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse undocumented MARRIAGE code 0 into the 'Other' category (3).

        The dataset documentation only defines values 1–3.  Code 0 appears in
        the data but is not defined; it is treated as 'Other'.
        """
        df[COL_MARRIAGE] = df[COL_MARRIAGE].replace({0: 3})
        return df

    @staticmethod
    def _apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p-based transforms to right-skewed monetary columns.

        * ``LIMIT_BAL`` and ``PAY_AMT_*`` are always non-negative →
          ``log1p(clip(x, lower=0))``
        * ``BILL_*`` can be negative (overpayment credits) →
          signed transform: ``sign(x) · log1p(|x|)``
        """
        limit_col   = RENAME["X1"]                        # LIMIT_BAL
        bill_cols   = [RENAME[f"X{i}"] for i in range(12, 18)]
        pay_amt_cols = [RENAME[f"X{i}"] for i in range(18, 24)]

        # Non-negative columns: plain log1p (after clipping any stray negatives)
        for col in [limit_col] + pay_amt_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        # Bill columns: signed log1p to preserve directionality of negative values
        for col in bill_cols:
            if col in df.columns:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

        return df
