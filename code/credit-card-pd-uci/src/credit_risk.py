"""
credit_risk.py
==============
Entry point for the UCI credit-default prediction pipeline.

Usage
-----
    python credit_risk.py                  # run small + medium
    python credit_risk.py --small          # small only  (500 train)
    python credit_risk.py --medium         # medium only (5 000 train)
    python credit_risk.py --large          # large only  (full dataset)
    python credit_risk.py --no-viz         # skip EDA
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")

import signal
import sys

signal.signal(signal.SIGINT, lambda *_: (print("\n[interrupted]"), os._exit(1)))

# -- Path setup ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "..", "shared"))
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.data_cleaner import DataCleaner
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.data_visulisations import DataVisualisations
from pipeline import PipelineConfig, run_pipeline
from figures import generate_all_figures

SEED = 42
NO_VIZ = "--no-viz" in sys.argv
DATASET_CSV = os.path.join(_HERE, "dataset", "credit_risk.csv")

ROOT = _HERE
RESULTS_DIR = os.path.join(ROOT, "results")
EDA_DIR = os.path.join(RESULTS_DIR, "eda")

# Size tiers: dev (train+val) sample size, or None for full dev set.
# Test set is always 20 % of the full dataset (~6 k rows) for all tiers.
SIZE_TIERS = {
    "small":  500,
    "medium": 5_000,
    "large":  None,
}

VAL_FRAC = 0.15  # fraction of dev used for validation


def _dirs(label: str):
    base = os.path.join(RESULTS_DIR, label)
    return {
        "tables":      os.path.join(base, "tables"),
        "predictions": os.path.join(base, "predictions"),
        "weights":     os.path.join(base, "weights"),
        "figures":     os.path.join(base, "figures"),
    }


def _subsample(X: pd.DataFrame, y: pd.Series, n: int):
    """Stratified subsample to *n* total rows."""
    if n >= len(y):
        return X, y
    _, X_s, _, y_s = train_test_split(
        X, y, test_size=n, stratify=y, random_state=SEED,
    )
    return X_s.reset_index(drop=True), y_s.reset_index(drop=True)


def run_tier(
    X_dev: pd.DataFrame, y_dev: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    preprocessor: DataPreprocessor,
    label: str, n_dev: int | None,
) -> None:
    """Subsample dev, split train/val, preprocess, run pipeline."""
    # Subsample dev if needed
    if n_dev is not None:
        Xd, yd = _subsample(X_dev, y_dev, min(n_dev, len(y_dev)))
    else:
        Xd, yd = X_dev, y_dev

    # Split dev → train / val
    X_train, X_val, y_train, y_val = train_test_split(
        Xd, yd, test_size=VAL_FRAC, stratify=yd, random_state=SEED,
    )

    # Preprocess (fit on train only)
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p   = preprocessor.transform(X_val)
    X_test_p  = preprocessor.transform(X_test)

    splits = {
        "train":      (X_train_p, y_train.reset_index(drop=True)),
        "validation": (X_val_p,   y_val.reset_index(drop=True)),
        "test":       (X_test_p,  y_test.reset_index(drop=True)),
    }

    dirs = _dirs(label)
    cfg = PipelineConfig(
        seed=SEED,
        tables_dir=dirs["tables"],
        predictions_dir=dirs["predictions"],
        weights_dir=dirs["weights"],
    )
    run_pipeline(splits, cfg, dataset_name="UCI Credit Default", size_label=label)
    generate_all_figures(dirs["tables"], dirs["predictions"], dirs["figures"])


def main() -> None:
    # Load & clean
    df = DataLoader(data_path=DATASET_CSV).load_data()
    X_raw, y_raw = DataCleaner(df).clean(apply_log_transforms=True)

    if not NO_VIZ:
        os.makedirs(EDA_DIR, exist_ok=True)
        viz = DataVisualisations(output_dir=EDA_DIR)
        viz.target_distribution(y_raw)
        viz.correlation_heatmap(X_raw)
        viz.feature_distributions(X_raw)
        viz.categorical_distributions(X_raw, y_raw)
        viz.boxplots_by_target(X_raw, y_raw)
        viz.temporal_trends(X_raw, y_raw)
        viz.repayment_status_heatmap(X_raw, y_raw)

    # Split off fixed test set (20 % of full data)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED,
    )

    preprocessor = DataPreprocessor(random_state=SEED)

    # Determine which tiers to run
    explicit = [k for k in SIZE_TIERS if f"--{k}" in sys.argv]
    tiers = explicit if explicit else ["small", "medium"]

    for label in tiers:
        run_tier(X_dev, y_dev, X_test, y_test, preprocessor, label, SIZE_TIERS[label])


if __name__ == "__main__":
    main()
