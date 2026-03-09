"""Generate dissertation figures for the UCI credit-card dataset."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "..", "shared"))
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data.data_cleaner import DataCleaner
from data.data_loader import DataLoader
from figures import apply_style, generate_all_figures

SMALL = "--small" in sys.argv
RESULTS_DIR = os.path.join(_HERE, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "small") if SMALL else os.path.join(RESULTS_DIR, "tables")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions", "small") if SMALL else os.path.join(RESULTS_DIR, "predictions")
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "small") if SMALL else os.path.join(RESULTS_DIR, "figures")
DATASET_CSV = os.path.join(_HERE, "dataset", "credit_risk.csv")


def plot_eda_overview() -> None:
    """Dataset-specific EDA figure."""
    loader = DataLoader(data_path=DATASET_CSV)
    raw_df = loader.load_data()
    cleaner = DataCleaner(raw_df)
    X_all, y_all = cleaner.clean(apply_log_transforms=True)

    apply_style()
    base_rate = float(y_all.mean())
    counts = y_all.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    labels = [f"No Default\n({1 - base_rate:.1%})", f"Default\n({base_rate:.1%})"]
    colors = ["#6ACC65", "#D65F5F"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.0, width=0.55)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150, f"{val:,}", ha="center", va="bottom")
    ax.set_title("(a) Target class distribution")
    ax.set_ylabel("Number of clients")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax = axes[1]
    pay_cols = [c for c in X_all.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
    pay_df = X_all[pay_cols].copy()
    pay_df["default"] = y_all.values
    positive = [pay_df.loc[pay_df["default"].eq(1), c].median() for c in pay_cols]
    negative = [pay_df.loc[pay_df["default"].eq(0), c].median() for c in pay_cols]
    months = [c.replace("PAY_", "") for c in pay_cols]
    x = np.arange(len(months))
    w = 0.35
    ax.bar(x - w / 2, negative, width=w, color="#6ACC65", edgecolor="white", label="No Default")
    ax.bar(x + w / 2, positive, width=w, color="#D65F5F", edgecolor="white", label="Default")
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_title("(b) Repayment status by default outcome")
    ax.set_ylabel("Median repayment-status code")
    ax.axhline(0, color="0.4", linewidth=0.8, linestyle=":")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_overview.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    if not SMALL:
        plot_eda_overview()
    generate_all_figures(TABLES_DIR, PREDICTIONS_DIR, FIG_DIR, small=SMALL)


if __name__ == "__main__":
    main()
