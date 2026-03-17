"""
run.py — Kin8nm regression benchmark.

Usage
-----
    python run.py                  # run small + medium
    python run.py --small          # small only  (500 dev)
    python run.py --medium         # medium only (3 000 dev)
    python run.py --large          # large only  (full dataset)
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "..", "shared"))
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

import pandas as pd
from sklearn.model_selection import train_test_split

from data.data_loader import load_kin8nm
from data.data_preprocessor import DataPreprocessor
from base import PipelineConfig
from regression_pipeline import run_regression_pipeline
from regression_figures import generate_all_regression_figures

SEED = 42
DATASET_DIR = os.path.join(_HERE, "dataset")
RESULTS_DIR = os.path.join(_HERE, "results")

SIZE_TIERS = {
    "small":  500,
    "medium": 3_000,
    "large":  None,   # full dev set
}

VAL_FRAC = 0.15


def _dirs(label: str):
    base = os.path.join(RESULTS_DIR, label)
    return {
        "tables":      os.path.join(base, "tables"),
        "predictions": os.path.join(base, "predictions"),
        "weights":     os.path.join(base, "weights"),
        "figures":     os.path.join(base, "figures"),
    }


def run_tier(
    X_dev: pd.DataFrame, y_dev: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    preprocessor: DataPreprocessor,
    label: str, n_dev: int | None,
) -> None:
    if n_dev is not None and n_dev < len(y_dev):
        _, Xd, _, yd = train_test_split(
            X_dev, y_dev, test_size=n_dev, random_state=SEED,
        )
        Xd, yd = Xd.reset_index(drop=True), yd.reset_index(drop=True)
    else:
        Xd, yd = X_dev, y_dev

    X_train, X_val, y_train, y_val = train_test_split(
        Xd, yd, test_size=VAL_FRAC, random_state=SEED,
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

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
        figures_dir=dirs["figures"],
    )
    run_regression_pipeline(splits, cfg, dataset_name="Kin8nm", size_label=label)
    generate_all_regression_figures(dirs["tables"], dirs["predictions"], dirs["figures"])


def main() -> None:
    df = load_kin8nm(DATASET_DIR)
    X = df.drop(columns=["y"])
    y = df["y"]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
    )

    preprocessor = DataPreprocessor(random_state=SEED)

    explicit = [k for k in SIZE_TIERS if f"--{k}" in sys.argv]
    tiers = explicit if explicit else ["small", "medium"]

    for label in tiers:
        run_tier(X_dev, y_dev, X_test, y_test, preprocessor, label, SIZE_TIERS[label])


if __name__ == "__main__":
    main()
