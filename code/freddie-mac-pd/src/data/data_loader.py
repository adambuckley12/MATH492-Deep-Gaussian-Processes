"""
data_loader.py
==============
Load Freddie Mac Single-Family Loan-Level data from real quarterly files.

The Freddie Mac Single-Family Loan Performance dataset is available (free
registration) at https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset.
Place origination and monthly performance files under ``dataset/``:

    dataset/
      historical_data_2025Q1.txt       (origination)
      historical_data_time_2025Q1.txt  (performance)
      ...

Each file is pipe-delimited with no header.  Column definitions follow the
Freddie Mac User Guide (v37, December 2023).

References
----------
* Freddie Mac (2023). Single-Family Loan-Level Dataset User Guide.
* Frame et al. (2015). The Rescue of Fannie Mae and Freddie Mac.
  Journal of Economic Perspectives 29(2), 25-52.
"""

from __future__ import annotations

import os
import glob

import numpy as np
import pandas as pd


# -- Freddie Mac column definitions -------------------------------------------

ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_buyer",
    "maturity_date", "msa", "mi_pct", "num_units", "occupancy",
    "orig_cltv", "orig_dti", "orig_upb", "orig_ltv", "orig_rate",
    "channel", "ppm_flag", "amort_type", "property_state",
    "property_type", "postal_code", "loan_seq_no", "loan_purpose",
    "orig_term", "num_borrowers", "seller", "servicer",
    "super_conforming", "pre_harp_seq", "program_indicator",
    "harp_indicator", "valuation_method", "io_indicator",
    "mi_type",
]

PERF_COLS = [
    "loan_seq_no", "report_period", "current_upb", "delinq_status",
    "loan_age", "remaining_maturity", "repurchase_flag", "mod_flag",
    "zero_bal_code", "zero_bal_date", "current_rate", "deferred_upb",
    "last_paid_date", "mi_recoveries", "net_proceeds",
    "non_mi_recoveries", "expenses", "legal_costs", "maint_costs",
    "taxes_insurance", "misc_expenses", "actual_loss", "mod_cost",
    "step_mod_flag", "deferred_plan", "est_ltv",
    "zero_bal_removal_upb", "delinq_interest", "disaster_delinq",
    "borrower_assist", "current_mod_cost", "interest_bearing_upb",
]

# US Census regions for geographic stratification
STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
    "NH": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast", "RI": "Northeast", "VT": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest",
    "KS": "Midwest", "MI": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest",
    "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",
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


class FreddieMacLoader:
    """
    Load and merge Freddie Mac origination + performance files.

    Parameters
    ----------
    dataset_dir : str
        Directory containing Freddie Mac quarterly text files.
    observation_months : int
        Window (months after origination) for defining the default label.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        dataset_dir: str,
        observation_months: int = 24,
        seed: int = 42,
    ):
        self.dataset_dir = dataset_dir
        self.observation_months = observation_months
        self.seed = seed

    # -- Public API ------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load data from Freddie Mac files.

        Returns a DataFrame with one row per loan, including engineered
        stratification features and the binary ``default`` target.
        """
        orig_files = sorted(
            glob.glob(os.path.join(self.dataset_dir, "historical_data_*.txt"))
        )
        orig_files = [f for f in orig_files if "_time_" not in os.path.basename(f)]
        perf_files = sorted(
            glob.glob(os.path.join(self.dataset_dir, "historical_data_time_*.txt"))
        )

        if not orig_files or not perf_files:
            raise FileNotFoundError(
                f"No Freddie Mac data files found in {self.dataset_dir}. "
                "Place origination (historical_data_*.txt) and performance "
                "(historical_data_time_*.txt) files in the dataset/ directory."
            )

        # Pair origination ↔ performance files by quarter tag
        pairs = self._pair_files(orig_files, perf_files)
        print(f"  Found {len(pairs)} paired quarters")

        df = self._load_and_merge(pairs)
        df = self._engineer_strata(df)
        return df

    # -- File pairing ----------------------------------------------------------

    @staticmethod
    def _quarter_tag(path: str) -> str:
        """Extract quarter tag like '2023Q1' from a filename."""
        base = os.path.basename(path)
        # historical_data_2023Q1.txt or historical_data_time_2023Q1.txt
        tag = base.replace("historical_data_time_", "").replace("historical_data_", "")
        return tag.replace(".txt", "")

    def _pair_files(
        self,
        orig_files: list[str],
        perf_files: list[str],
    ) -> list[tuple[str, str, str]]:
        """Return (quarter_tag, orig_path, perf_path) for matched pairs."""
        perf_by_q = {self._quarter_tag(f): f for f in perf_files}
        pairs = []
        for of in orig_files:
            q = self._quarter_tag(of)
            if q in perf_by_q:
                pairs.append((q, of, perf_by_q[q]))
        return sorted(pairs)

    # -- Loading ---------------------------------------------------------------

    def _load_and_merge(
        self,
        pairs: list[tuple[str, str, str]],
    ) -> pd.DataFrame:
        """Load paired files, keeping only sufficiently seasoned quarters."""
        orig_dfs = []
        perf_dfs = []
        used_quarters = []

        for quarter, orig_path, perf_path in pairs:
            # Quick check: is this quarter seasoned enough?
            perf_age = pd.read_csv(
                perf_path, sep="|", header=None, names=PERF_COLS,
                dtype=str, low_memory=False, usecols=["loan_age"],
            )
            max_age = pd.to_numeric(
                perf_age["loan_age"], errors="coerce"
            ).max()
            if pd.isna(max_age) or max_age < self.observation_months:
                print(f"    Skipping {quarter} (max age {max_age:.0f}m < {self.observation_months}m window)")
                continue

            used_quarters.append(quarter)
            orig_dfs.append(pd.read_csv(
                orig_path, sep="|", header=None, names=ORIG_COLS,
                dtype=str, low_memory=False,
            ))
            perf_dfs.append(pd.read_csv(
                perf_path, sep="|", header=None, names=PERF_COLS,
                dtype=str, low_memory=False,
            ))
            print(f"    Loaded {quarter} (max age {max_age:.0f}m)")

        if not orig_dfs:
            raise ValueError(
                f"No quarters have sufficient seasoning (>= {self.observation_months}m). "
                "Try reducing observation_months or adding older data."
            )

        print(f"  Using quarters: {', '.join(used_quarters)}")
        orig = pd.concat(orig_dfs, ignore_index=True)
        perf = pd.concat(perf_dfs, ignore_index=True)
        print(f"  Origination records: {len(orig):,}")
        print(f"  Performance records: {len(perf):,}")

        # Compute default label: ever 90+ DPD or foreclosure/short-sale
        perf["delinq_num"] = pd.to_numeric(
            perf["delinq_status"], errors="coerce"
        ).fillna(0)
        perf["zbc"] = pd.to_numeric(
            perf["zero_bal_code"], errors="coerce"
        ).fillna(0)
        perf["is_d90"] = (perf["delinq_num"] >= 3).astype(int)
        perf["is_credit_event"] = perf["zbc"].isin([2, 3, 9, 15]).astype(int)
        perf["is_default"] = (
            (perf["is_d90"] == 1) | (perf["is_credit_event"] == 1)
        ).astype(int)

        # Filter to observation window (loan_age <= observation_months)
        perf["loan_age_num"] = pd.to_numeric(
            perf["loan_age"], errors="coerce"
        ).fillna(999)
        perf_window = perf.loc[perf["loan_age_num"] <= self.observation_months]

        default_flag = (
            perf_window.groupby("loan_seq_no")["is_default"]
            .max()
            .reset_index()
            .rename(columns={"is_default": "default"})
        )

        # Merge
        df = orig.merge(default_flag, on="loan_seq_no", how="inner")
        print(f"  Merged loans: {len(df):,}")

        # Convert numeric columns
        for col in [
            "credit_score", "orig_ltv", "orig_cltv", "orig_dti",
            "orig_upb", "orig_rate", "mi_pct", "num_units",
            "orig_term", "num_borrowers",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        default_rate = df["default"].mean()
        print(f"  Default rate: {default_rate:.2%}")
        return df

    # -- Feature engineering (for stratification) ------------------------------

    @staticmethod
    def _engineer_strata(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns needed by StratifiedSampler and DataCleaner."""
        # Origination year from first_payment_date (YYYYMM format)
        df["orig_year"] = pd.to_numeric(
            df["first_payment_date"].astype(str).str[:4], errors="coerce"
        ).fillna(2020).astype(int)

        # Vintage era
        df["vintage_era"] = df["orig_year"].apply(
            lambda y: "pre_crisis" if y < 2007
            else "crisis" if y <= 2009
            else "covid" if y >= 2020
            else "post_crisis"
        )

        # Region from property state
        df["region"] = df["property_state"].map(STATE_TO_REGION).fillna("South")

        # Rate spread (loan rate minus annual cohort median)
        cohort_median = df.groupby("orig_year")["orig_rate"].transform("median")
        df["rate_spread"] = df["orig_rate"] - cohort_median

        # Refi bucket from rate spread
        df["refi_bucket"] = df["rate_spread"].apply(
            lambda s: "in_the_money" if s > 1.0
            else "out_of_the_money" if s < -1.0
            else "at_the_money"
        )

        # LTV bucket
        df["ltv_bucket"] = pd.cut(
            df["orig_ltv"],
            bins=[0, 60, 80, 200],
            labels=["low", "medium", "high"],
        ).astype(str)

        return df


class StratifiedSampler:
    """
    Stratified sampler for reducing a large Freddie Mac dataset to a
    manageable size while preserving representation across key strata.

    Stratification variables
    ------------------------
    * Vintage era (pre-crisis, crisis, post-crisis, COVID)
    * Refinancing incentive bucket (in/out/at the money)
    * Geographic region (Northeast, Midwest, South, West)

    The sampler uses proportional allocation across strata up to the
    target n_samples.  Under-represented strata are fully included;
    surplus is redistributed proportionally among larger strata.
    """

    STRATA_COLS = ["vintage_era", "refi_bucket", "region"]

    def __init__(self, n_samples: int, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a stratified subsample of ``df``."""
        if len(df) <= self.n_samples:
            print(
                f"\n  Dataset ({len(df):,} loans) <= target "
                f"({self.n_samples:,}), using all data"
            )
            return df.reset_index(drop=True)

        rng = np.random.default_rng(self.seed)

        # Compute target per stratum (proportional)
        groups = df.groupby(self.STRATA_COLS)
        strata_sizes = groups.size()
        total = len(df)
        target_per = (strata_sizes / total * self.n_samples).round().astype(int)
        target_per = target_per.clip(lower=1)

        sampled = []
        for key, grp in groups:
            t = min(target_per.loc[key], len(grp))
            sampled.append(
                grp.sample(n=t, random_state=int(rng.integers(1e9)))
            )

        result = pd.concat(sampled, ignore_index=True)

        # If under target, sample more from remaining rows
        if len(result) < self.n_samples:
            remaining = df.drop(result.index, errors="ignore")
            extra = remaining.sample(
                n=min(self.n_samples - len(result), len(remaining)),
                random_state=int(rng.integers(1e9)),
            )
            result = pd.concat([result, extra], ignore_index=True)

        # If over target, trim
        if len(result) > self.n_samples:
            result = result.sample(
                n=self.n_samples, random_state=int(rng.integers(1e9))
            )

        result = result.reset_index(drop=True)

        # Print strata summary
        print(f"\n  Stratified sample: {len(result):,} loans")
        for col in self.STRATA_COLS:
            counts = result[col].value_counts().sort_index()
            print(f"    {col}: {dict(counts)}")
        default_rate = result["default"].mean()
        print(f"    Default rate: {default_rate:.2%}")

        return result
