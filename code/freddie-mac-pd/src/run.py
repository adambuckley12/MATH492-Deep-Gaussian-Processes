"""
run.py
======
Entry point for the Freddie Mac mortgage PD pipeline.

Usage
-----
    python run.py                  # run small + medium
    python run.py --small          # small only  (500 total)
    python run.py --medium         # medium only (5 000 total)
    python run.py --large          # large only  (30 000 total)
    python run.py --no-viz         # skip EDA
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

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "..", "shared"))
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

import pandas as pd

from data.data_cleaner import DataCleaner
from data.data_loader import FreddieMacLoader, StratifiedSampler
from data.data_preprocessor import DataPreprocessor
from pipeline import PipelineConfig, run_pipeline
from figures import generate_all_figures

SEED = 42
NO_VIZ = "--no-viz" in sys.argv
FIGURES_ONLY = "--figures-only" in sys.argv

ROOT = _HERE
RESULTS_DIR = os.path.join(ROOT, "results")
EDA_DIR = os.path.join(RESULTS_DIR, "eda")
DATASET_DIR = os.path.join(ROOT, "dataset")

SIZE_TIERS = {
    "small":  500,
    "medium": 5_000,
    "large":  30_000,
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


def eda_plots(df_raw: pd.DataFrame, y: pd.Series, eda_dir: str) -> None:
    """Dataset-specific EDA visualisations for Freddie Mac."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    })
    os.makedirs(eda_dir, exist_ok=True)

    if "vintage_era" in df_raw.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        era_order = ["pre_crisis", "crisis", "post_crisis", "covid"]
        rates = df_raw.groupby("vintage_era")["default"].mean().reindex(era_order)
        rates.plot.bar(ax=ax, color=["#D65F5F", "#C4AD66", "#4878CF", "#6ACC65"],
                       edgecolor="white", linewidth=0.6)
        ax.set_ylabel("Default rate"); ax.set_title("Default rate by vintage era")
        ax.set_xticklabels(era_order, rotation=0)
        fig.tight_layout(); fig.savefig(os.path.join(eda_dir, "default_by_era.png")); plt.close(fig)

    if "credit_score" in df_raw.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_raw.loc[df_raw["default"] == 0, "credit_score"].hist(ax=ax, bins=40, alpha=0.6, label="No default", color="#4878CF", density=True)
        df_raw.loc[df_raw["default"] == 1, "credit_score"].hist(ax=ax, bins=40, alpha=0.6, label="Default", color="#D65F5F", density=True)
        ax.set_xlabel("Credit score (FICO)"); ax.set_title("Credit score distribution"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(eda_dir, "credit_score_dist.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    y.value_counts().sort_index().plot.bar(ax=ax, color=["#4878CF", "#D65F5F"], edgecolor="white", linewidth=0.6)
    ax.set_xticklabels(["No default", "Default"], rotation=0); ax.set_ylabel("Count"); ax.set_title("Target distribution")
    fig.tight_layout(); fig.savefig(os.path.join(eda_dir, "target_distribution.png")); plt.close(fig)
    print(f"  EDA figures → {eda_dir}")


def run_tier(
    X_dev: pd.DataFrame, y_dev: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    preprocessor: DataPreprocessor,
    label: str, n_dev: int,
) -> None:
    """Subsample dev, split train/val, preprocess, run pipeline."""
    from sklearn.model_selection import train_test_split

    # Subsample dev if needed
    if n_dev < len(y_dev):
        _, Xd, _, yd = train_test_split(
            X_dev, y_dev, test_size=n_dev, stratify=y_dev, random_state=SEED,
        )
        Xd, yd = Xd.reset_index(drop=True), yd.reset_index(drop=True)
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
    run_pipeline(splits, cfg, dataset_name="Freddie Mac Mortgage Default", size_label=label)
    generate_all_figures(dirs["tables"], dirs["predictions"], dirs["figures"],
                         calibration_strategy="uniform")


def main() -> None:
    from sklearn.model_selection import train_test_split

    explicit = [k for k in SIZE_TIERS if f"--{k}" in sys.argv]

    if FIGURES_ONLY:
        # Regenerate figures from saved predictions without retraining
        for label in (explicit if explicit else list(SIZE_TIERS.keys())):
            dirs = _dirs(label)
            if not os.path.exists(os.path.join(dirs["predictions"], "predictions.csv")):
                print(f"  [{label}] No saved predictions found, skipping.")
                continue
            print(f"  [{label}] Regenerating figures from saved predictions...")
            generate_all_figures(dirs["tables"], dirs["predictions"], dirs["figures"],
                                 calibration_strategy="uniform")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    loader = FreddieMacLoader(dataset_dir=DATASET_DIR, seed=SEED)
    df_raw = loader.load()

    # Sample 30 k loans (the "large" working set) and clean once
    sampler = StratifiedSampler(n_samples=min(30_000, len(df_raw)), seed=SEED)
    df_sampled = sampler.sample(df_raw)

    if not NO_VIZ:
        eda_plots(df_sampled, df_sampled["default"], EDA_DIR)

    cleaner = DataCleaner(df_sampled)
    X_raw, y_raw = cleaner.clean(apply_log_transforms=True)

    # Fixed test set: 20 % of the 30 k sample (~6 k rows), shared by all tiers
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED,
    )

    preprocessor = DataPreprocessor(random_state=SEED)

    tiers = explicit if explicit else ["small", "medium"]

    for label in tiers:
        run_tier(X_dev, y_dev, X_test, y_test, preprocessor, label, SIZE_TIERS[label])


if __name__ == "__main__":
    main()
