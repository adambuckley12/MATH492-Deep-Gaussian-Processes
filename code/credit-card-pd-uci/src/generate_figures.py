"""Generate dissertation figures from exported experiment tables."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc as sk_auc,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.data_cleaner import DataCleaner
from data.data_loader import DataLoader
from classical.metrics import expected_calibration_error

DATASET_CSV = os.path.join(_HERE, "dataset", "credit_risk.csv")
RESULTS_DIR = os.path.join(_HERE, "results")
SMALL = "--small" in sys.argv
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "small") if SMALL else os.path.join(RESULTS_DIR, "tables")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions", "small") if SMALL else os.path.join(RESULTS_DIR, "predictions")
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "small") if SMALL else os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MAIN_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "SVGP",
    "DSVGP-2L",
    "DSVGP-3L",
]
CLASSICAL_MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]
PALETTE = {
    "Logistic Regression": "#4878CF",
    "Random Forest": "#6ACC65",
    "XGBoost": "#D65F5F",
    "SVGP": "#B47CC7",
    "DSVGP-2L": "#C4AD66",
    "DSVGP-3L": "#77BEDB",
    "Logistic Regression (unweighted)": "#96B9F2",
    "Random Forest (unweighted)": "#9EDB98",
    "XGBoost (unweighted)": "#F29A9A",
    "Logistic Regression + Platt": "#1D3A7A",
    "Random Forest + Platt": "#327A2B",
    "XGBoost + Platt": "#8F2C2C",
    "Logistic Regression + Isotonic": "#2748A8",
    "Random Forest + Isotonic": "#429437",
    "XGBoost + Isotonic": "#B23A3A",
}

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(os.path.join(TABLES_DIR, "all_metrics.csv"))
    predictions = pd.read_csv(os.path.join(PREDICTIONS_DIR, "test_predictions.csv"))
    thresholds = pd.read_csv(os.path.join(TABLES_DIR, "threshold_sweep.csv"))
    threshold_optima = pd.read_csv(os.path.join(TABLES_DIR, "threshold_optima.csv"))
    runtime = pd.read_csv(os.path.join(TABLES_DIR, "runtime_summary.csv"))
    feature_importance = pd.read_csv(os.path.join(TABLES_DIR, "feature_importance.csv"))
    return metrics, predictions, thresholds, threshold_optima, runtime, feature_importance


def prediction_slice(predictions: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    subset = predictions.loc[predictions["model"].eq(model)].sort_values("row_id")
    return subset["y_true"].to_numpy(), subset["y_prob"].to_numpy()


def plot_eda_overview() -> None:
    loader = DataLoader(data_path=DATASET_CSV)
    raw_df = loader.load_data()
    cleaner = DataCleaner(raw_df)
    X_all, y_all = cleaner.clean(apply_log_transforms=True)

    base_rate = float(y_all.mean())
    counts = y_all.value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    labels = [f"No Default\n({1 - base_rate:.1%})", f"Default\n({base_rate:.1%})"]
    colors = ["#6ACC65", "#D65F5F"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.0, width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150, f"{value:,}", ha="center", va="bottom")
    ax.set_title("(a) Target class distribution")
    ax.set_ylabel("Number of clients")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax = axes[1]
    pay_cols = [c for c in X_all.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
    pay_df = X_all[pay_cols].copy()
    pay_df["default"] = y_all.values
    positive = [pay_df.loc[pay_df["default"].eq(1), col].median() for col in pay_cols]
    negative = [pay_df.loc[pay_df["default"].eq(0), col].median() for col in pay_cols]
    months = [col.replace("PAY_", "") for col in pay_cols]
    x = np.arange(len(months))
    width = 0.35
    ax.bar(x - width / 2, negative, width=width, color="#6ACC65", edgecolor="white", label="No Default")
    ax.bar(x + width / 2, positive, width=width, color="#D65F5F", edgecolor="white", label="Default")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_title("(b) Repayment status by default outcome")
    ax.set_ylabel("Median repayment-status code")
    ax.axhline(0, color="0.4", linewidth=0.8, linestyle=":")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_overview.png"), bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for model in MAIN_MODELS:
        y_true, y_prob = prediction_slice(predictions, model)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color=PALETTE[model], label=f"{model} (AUC={sk_auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_title("(a) ROC curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    prevalence = None
    for model in MAIN_MODELS:
        y_true, y_prob = prediction_slice(predictions, model)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        prevalence = float(y_true.mean())
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=PALETTE[model], label=f"{model} (AP={ap:.4f})")
    ax.axhline(prevalence, color="0.4", linestyle="--", lw=0.8, label=f"Prevalence ({prevalence:.2f})")
    ax.set_title("(b) Precision-Recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_pr.png"), bbox_inches="tight")
    plt.close(fig)


def plot_raw_calibration(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    groups = [CLASSICAL_MODELS, ["SVGP", "DSVGP-2L", "DSVGP-3L"]]
    titles = ["(a) Classical baselines", "(b) GP and DGP models"]

    for ax, group, title in zip(axes, groups, titles):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="0.9", alpha=0.5, label="±0.05 band")
        for model in group:
            y_true, y_prob = prediction_slice(predictions, model)
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy="quantile")
            ece = expected_calibration_error(y_true, y_prob, n_bins=15)
            ax.plot(mean_pred, frac_pos, marker="o", color=PALETTE[model], label=f"{model} (ECE={ece:.4f})")
        ax.set_title(title)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration.png"), bbox_inches="tight")
    plt.close(fig)


def plot_classical_calibration_comparison(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    variants = {
        "Logistic Regression": [
            "Logistic Regression",
            "Logistic Regression (unweighted)",
            "Logistic Regression + Platt",
            "Logistic Regression + Isotonic",
        ],
        "Random Forest": [
            "Random Forest",
            "Random Forest (unweighted)",
            "Random Forest + Platt",
            "Random Forest + Isotonic",
        ],
        "XGBoost": [
            "XGBoost",
            "XGBoost (unweighted)",
            "XGBoost + Platt",
            "XGBoost + Isotonic",
        ],
    }

    for ax, (base_model, model_names) in zip(axes, variants.items()):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        for model in model_names:
            y_true, y_prob = prediction_slice(predictions, model)
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="o", color=PALETTE[model], label=model)
        ax.set_title(base_model)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=6.5)
    axes[0].set_ylabel("Fraction of positives")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "classical_calibration_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.ravel()
    for ax, model in zip(axes_flat, MAIN_MODELS):
        subset = predictions.loc[predictions["model"].eq(model)].sort_values("row_id")
        cm = confusion_matrix(subset["y_true"], subset["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", text_kw={"fontsize": 10})
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    specs = [
        ("Logistic Regression", "abs_coef", "#4878CF", "Standardised |coefficient|"),
        ("Random Forest", "importance", "#6ACC65", "Mean Gini decrease"),
        ("XGBoost", "gain", "#D65F5F", "Mean gain per split"),
    ]
    for ax, (model, value_col, color, xlabel) in zip(axes, specs):
        top = (
            feature_importance.loc[feature_importance["model"].eq(model), ["feature", value_col]]
            .nlargest(10, value_col)
            .sort_values(value_col, ascending=True)
        )
        bars = ax.barh(top["feature"], top[value_col], color=color, edgecolor="white", linewidth=0.6)
        x_max = top[value_col].max()
        for bar, value in zip(bars, top[value_col]):
            ax.text(value + x_max * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7)
        ax.set_title(model, fontweight="bold", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xlim(0, x_max * 1.22)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "feature_importance.png"), bbox_inches="tight")
    plt.close(fig)


def plot_runtime_tradeoff(metrics: pd.DataFrame, runtime: pd.DataFrame) -> None:
    merged = metrics.merge(
        runtime,
        on=["model", "base_model", "family", "stage", "calibration_method", "train_time_sec"],
        how="left",
    )
    merged = merged.loc[merged["model"].isin(MAIN_MODELS)].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for _, row in merged.iterrows():
        ax.scatter(row["train_time_sec"], row["ece"], s=90, color=PALETTE[row["model"]], edgecolor="white", linewidth=0.8)
        ax.text(row["train_time_sec"] * 1.02, row["ece"] + 0.002, row["model"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (seconds, log scale)")
    ax.set_ylabel("ECE")
    ax.set_title("Runtime vs calibration trade-off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "runtime_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(thresholds: pd.DataFrame) -> None:
    subset = thresholds.loc[thresholds["model"].isin(MAIN_MODELS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    for model in MAIN_MODELS:
        rows = subset.loc[subset["model"].eq(model)].sort_values("threshold")
        axes[0].plot(rows["threshold"], rows["f1"], color=PALETTE[model], label=model)
        axes[1].plot(rows["threshold"], rows["balanced_accuracy"], color=PALETTE[model], label=model)

    axes[0].set_title("(a) Threshold vs F1")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("F1")
    axes[1].set_title("(b) Threshold vs balanced accuracy")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "threshold_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


def plot_calibrated_threshold_tradeoff(thresholds: pd.DataFrame) -> None:
    focus_models = [
        "Logistic Regression",
        "Logistic Regression + Platt",
        "Random Forest",
        "Random Forest + Platt",
        "XGBoost",
        "XGBoost + Platt",
    ]
    subset = thresholds.loc[thresholds["model"].isin(focus_models)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    for model in focus_models:
        rows = subset.loc[subset["model"].eq(model)].sort_values("threshold")
        axes[0].plot(rows["threshold"], rows["f1"], color=PALETTE[model], label=model)
        axes[1].plot(rows["threshold"], rows["recall"], color=PALETTE[model], label=model)

    axes[0].set_title("(a) F1 before and after Platt calibration")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("F1")
    axes[1].set_title("(b) Recall before and after Platt calibration")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Recall")
    axes[1].legend(loc="lower left", fontsize=6.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibrated_threshold_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics, predictions, thresholds, threshold_optima, runtime, feature_importance = load_inputs()
    if SMALL:
        # For the small-data regime, only generate the comparison figures
        plot_roc_pr(predictions)
        plot_raw_calibration(predictions)
        print("Small-data figures written to:", FIG_DIR)
        return
    plot_eda_overview()
    plot_roc_pr(predictions)
    plot_raw_calibration(predictions)
    plot_classical_calibration_comparison(predictions)
    plot_confusion_matrices(predictions)
    plot_feature_importance(feature_importance)
    plot_runtime_tradeoff(metrics, runtime)
    plot_threshold_tradeoff(thresholds)
    plot_calibrated_threshold_tradeoff(thresholds)
    print("Figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()