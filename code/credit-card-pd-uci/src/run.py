"""
credit_risk.py  (refactored)
=============
Entry point for the UCI credit-card default prediction pipeline.
Loads and preprocesses the dataset, then delegates to the shared pipeline.

Usage
-----
    python credit_risk.py
    python credit_risk.py --quick
    python credit_risk.py --no-viz
    python credit_risk.py --results
    python credit_risk.py --small
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

# ── Path setup: add shared package and local src to sys.path ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "..", "shared"))
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

from data.data_cleaner import DataCleaner
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.data_visulisations import DataVisualisations
from pipeline import PipelineConfig, run_pipeline

SEED = 42
QUICK = "--quick" in sys.argv
NO_VIZ = "--no-viz" in sys.argv
RESULTS_ONLY = "--results" in sys.argv
SMALL = "--small" in sys.argv

ROOT = _HERE
RESULTS_DIR = os.path.join(ROOT, "results")
EDA_DIR = os.path.join(RESULTS_DIR, "eda")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "small") if SMALL else os.path.join(RESULTS_DIR, "tables")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions", "small") if SMALL else os.path.join(RESULTS_DIR, "predictions")
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "models", "weights_small") if SMALL else os.path.join(RESULTS_DIR, "models", "weights")
DATASET_CSV = os.path.join(ROOT, "dataset", "credit_risk.csv")


def main() -> None:
    os.makedirs(EDA_DIR, exist_ok=True)

    # ── 1. Load and clean ─────────────────────────────────────────────────────
    data_loader = DataLoader(data_path=DATASET_CSV)
    df = data_loader.load_data()

    data_cleaner = DataCleaner(df)
    X_raw, y_raw = data_cleaner.clean(apply_log_transforms=True)

    # ── 2. EDA visualisations (dataset-specific) ──────────────────────────────
    if not NO_VIZ:
        viz = DataVisualisations(output_dir=EDA_DIR)
        viz.target_distribution(y_raw)
        viz.correlation_heatmap(X_raw)
        viz.feature_distributions(X_raw)
        viz.categorical_distributions(X_raw, y_raw)
        viz.boxplots_by_target(X_raw, y_raw)
        viz.temporal_trends(X_raw, y_raw)
        viz.repayment_status_heatmap(X_raw, y_raw)

    # ── 3. Preprocess (encode, scale, split) ──────────────────────────────────
    preprocessor = DataPreprocessor(
        test_size=0.2, calibration_size=0.2,
        validation_size=0.1, random_state=SEED,
    )
    splits = preprocessor.fit_transform_splits(X_raw, y_raw)

    # ── 4. Run shared pipeline ────────────────────────────────────────────────
    cfg = PipelineConfig(
        seed=SEED, quick=QUICK, small=SMALL,
        results_only=RESULTS_ONLY,
        tables_dir=TABLES_DIR,
        predictions_dir=PREDICTIONS_DIR,
        weights_dir=WEIGHTS_DIR,
    )
    run_pipeline(splits, cfg, dataset_name="UCI Credit Card Default")


if __name__ == "__main__":
    main()
