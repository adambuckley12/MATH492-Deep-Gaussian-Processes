"""
figures.py — Shared visualisations for PD modelling.

Each function takes DataFrames and an output directory, producing PNG figures.
Dataset-specific EDA lives in each project's own code.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc as sk_auc,
)

from classical.metrics import expected_calibration_error

MODELS = [
    "Logistic Regression", "Random Forest", "XGBoost",
    "SVGP", "DSVGP-2L", "DSVGP-3L",
]

PALETTE = {
    "Logistic Regression": "#4878CF",
    "Random Forest": "#6ACC65",
    "XGBoost": "#D65F5F",
    "SVGP": "#B47CC7",
    "DSVGP-2L": "#C4AD66",
    "DSVGP-3L": "#77BEDB",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    })


def _slice(preds: pd.DataFrame, model: str):
    s = preds.loc[preds["model"].eq(model)].sort_values("row_id")
    return s["y_true"].to_numpy(), s["y_prob"].to_numpy()


def load_results(tables_dir: str, predictions_dir: str):
    metrics = pd.read_csv(os.path.join(tables_dir, "metrics.csv"))
    preds = pd.read_csv(os.path.join(predictions_dir, "predictions.csv"))
    thresh = pd.read_csv(os.path.join(tables_dir, "threshold_sweep.csv"))
    fi = pd.read_csv(os.path.join(tables_dir, "feature_importance.csv"))
    runtime = pd.read_csv(os.path.join(tables_dir, "runtime.csv"))
    return metrics, preds, thresh, fi, runtime


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_pr(preds: pd.DataFrame, fig_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in MODELS:
        yt, yp = _slice(preds, m)
        fpr, tpr, _ = roc_curve(yt, yp)
        ax1.plot(fpr, tpr, color=PALETTE[m], label=f"{m} ({sk_auc(fpr, tpr):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(title="ROC", xlabel="FPR", ylabel="TPR")
    ax1.legend(loc="lower right", fontsize=7)

    prev = None
    for m in MODELS:
        yt, yp = _slice(preds, m)
        prec, rec, _ = precision_recall_curve(yt, yp)
        prev = float(yt.mean())
        ax2.plot(rec, prec, color=PALETTE[m],
                 label=f"{m} ({average_precision_score(yt, yp):.3f})")
    ax2.axhline(prev, color="0.4", ls="--", lw=0.8, label=f"Prevalence ({prev:.2f})")
    ax2.set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "roc_pr.png"), bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    preds: pd.DataFrame,
    fig_dir: str,
    strategy: str = "quantile",
) -> None:
    """Plot reliability diagrams.

    Parameters
    ----------
    strategy : {"quantile", "uniform"}
        Bin strategy passed to sklearn's ``calibration_curve``.
        Use ``"uniform"`` for low-prevalence datasets where quantile
        bins collapse into a narrow probability range.
    """
    classical = [m for m in MODELS[:3]]
    gp = [m for m in MODELS[3:]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, grp, title in [(ax1, classical, "Classical"), (ax2, gp, "GP / DGP")]:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                        color="0.9", alpha=0.5, label="±0.05")
        for m in grp:
            yt, yp = _slice(preds, m)
            frac, mp = calibration_curve(yt, yp, n_bins=15, strategy=strategy)
            ece = expected_calibration_error(yt, yp, n_bins=15)
            ax.plot(mp, frac, "o-", color=PALETTE[m], label=f"{m} (ECE={ece:.4f})")
        ax.set(title=title, xlabel="Mean predicted prob.", ylabel="Fraction of positives",
               xlim=(0, 1), ylim=(-0.02, 1.02))
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "calibration.png"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion(preds: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, m in zip(axes.ravel(), MODELS):
        s = preds.loc[preds["model"].eq(m)].sort_values("row_id")
        cm = confusion_matrix(s["y_true"], s["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No Def.", "Default"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(m, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion.png"), bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(fi: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    specs = [
        ("Logistic Regression", "abs_coef", "#4878CF"),
        ("Random Forest", "importance", "#6ACC65"),
        ("XGBoost", "gain", "#D65F5F"),
    ]
    for ax, (model, col, color) in zip(axes, specs):
        sub = fi.loc[fi["model"].eq(model)]
        if col not in sub.columns:
            continue
        top = sub[["feature", col]].nlargest(10, col).sort_values(col, ascending=True)
        ax.barh(top["feature"], top[col], color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(model, fontweight="bold", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_importance.png"), bbox_inches="tight")
    plt.close(fig)


def plot_runtime(metrics: pd.DataFrame, fig_dir: str) -> None:
    sub = metrics.loc[metrics["model"].isin(MODELS)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in sub.iterrows():
        ax.scatter(r["train_time_sec"], r["ece"], s=90,
                   color=PALETTE.get(r["model"]), edgecolor="white", linewidth=0.8)
        ax.text(r["train_time_sec"] * 1.05, r["ece"] + 0.002, r["model"], fontsize=8)
    ax.set(xscale="log", xlabel="Training time (s)", ylabel="ECE",
           title="Runtime vs calibration")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "runtime.png"), bbox_inches="tight")
    plt.close(fig)


def plot_threshold(thresh: pd.DataFrame, fig_dir: str) -> None:
    sub = thresh.loc[thresh["model"].isin(MODELS)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for m in MODELS:
        rows = sub.loc[sub["model"].eq(m)].sort_values("threshold")
        ax1.plot(rows["threshold"], rows["f1"], color=PALETTE[m], label=m)
        ax2.plot(rows["threshold"], rows["balanced_accuracy"], color=PALETTE[m], label=m)
    ax1.set(title="F1 vs Threshold", xlabel="Threshold", ylabel="F1")
    ax2.set(title="Balanced Acc. vs Threshold", xlabel="Threshold", ylabel="Balanced accuracy")
    ax2.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "threshold.png"), bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    tables_dir: str,
    predictions_dir: str,
    fig_dir: str,
    calibration_strategy: str = "quantile",
) -> None:
    apply_style()
    os.makedirs(fig_dir, exist_ok=True)
    metrics, preds, thresh, fi, _ = load_results(tables_dir, predictions_dir)
    plot_roc_pr(preds, fig_dir)
    plot_calibration(preds, fig_dir, strategy=calibration_strategy)
    plot_confusion(preds, fig_dir)
    plot_feature_importance(fi, fig_dir)
    plot_runtime(metrics, fig_dir)
    plot_threshold(thresh, fig_dir)
    print(f"  Figures → {fig_dir}")
