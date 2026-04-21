"""
classical/metrics.py
====================
Shared evaluation utilities for binary classification models.

Metrics reported (chosen for class-imbalanced credit default prediction):
  - ROC-AUC         : primary discrimination metric
  - Average Precision (AUC-PR): better than ROC-AUC under heavy imbalance
  - F1 (class 1)    : harmonic mean of precision/recall for the minority class
  - Precision / Recall (class 1)
  - Brier Score     : calibration quality - important for GP comparison
  - ECE             : Expected Calibration Error - weighted |acc - conf| per bin
  - Accuracy        : reported for completeness (misleading under imbalance)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
    log_loss,
    matthews_corrcoef,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Expected Calibration Error

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Partitions predicted probabilities into ``n_bins`` equal-width bins and
    computes the weighted average of |accuracy - confidence| per bin.

        ECE = Σ_m  (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : ground-truth binary labels
    y_prob : predicted probability for the positive class
    n_bins : number of equally-spaced probability bins (default 15)
    """
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    n     = len(y_true)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += abs(acc - conf) * mask.sum() / n
    return float(ece)


def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Compute and optionally print a standard suite of binary classification
    metrics.

    Parameters
    ----------
    model_name : str
    y_true     : ground-truth labels (0/1)
    y_pred     : hard predictions (0/1)
    y_prob     : predicted probability for class 1
    verbose    : print a formatted table

    Returns
    -------
    dict of metric_name → float
    """
    metrics = evaluate_predictions(
        model_name=model_name,
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        verbose=False,
    )

    if verbose:
        print(f"\n{'-' * 45}")
        print(f"  {model_name}")
        print(f"{'-' * 45}")
        print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision    : {metrics['avg_prec']:.4f}")
        print(f"  F1  (class 1)    : {metrics['f1']:.4f}")
        print(f"  Precision (cl.1) : {metrics['precision']:.4f}")
        print(f"  Recall    (cl.1) : {metrics['recall']:.4f}")
        print(f"  Specificity      : {metrics['specificity']:.4f}")
        print(f"  Balanced Acc.    : {metrics['balanced_accuracy']:.4f}")
        print(f"  MCC              : {metrics['mcc']:.4f}")
        print(f"  Brier Score      : {metrics['brier']:.4f}")
        print(f"  Log Loss         : {metrics['log_loss']:.4f}")
        print(f"  ECE              : {metrics['ece']:.4f}")
        print(f"  Accuracy         : {metrics['accuracy']:.4f}")
        print(f"{'-' * 45}")
        print(classification_report(y_true, y_pred,
                                    target_names=["No Default", "Default"]))

    return metrics


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Evaluate probability predictions at a given decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / max(tn + fp, 1)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_prec": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6)),
        "ece": expected_calibration_error(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": threshold,
    }

    if verbose:
        evaluate_classifier(model_name, y_true, y_pred, y_prob, verbose=True)
    return metrics


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing multiple models.

    Parameters
    ----------
    results : { model_name: metrics_dict }
    """
    df = pd.DataFrame(results).T
    preferred = [
        "roc_auc", "avg_prec", "f1", "precision", "recall",
        "specificity", "balanced_accuracy", "mcc", "brier",
        "log_loss", "ece", "accuracy", "threshold",
    ]
    col_order = [col for col in preferred if col in df.columns]
    df = df[col_order].round(4)
    df.index.name = "model"
    return df


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return threshold-wise precision/recall trade-offs for a probability vector."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(
            model_name="threshold_sweep",
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
            verbose=False,
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_threshold_optima(
    threshold_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Return the threshold achieving the best value of each requested metric."""
    if metrics is None:
        metrics = ["f1", "precision", "recall", "balanced_accuracy", "mcc", "specificity"]

    group_cols = [c for c in ["model", "family"] if c in threshold_df.columns]
    rows = []
    for group_key, group_df in threshold_df.groupby(group_cols, dropna=False):
        group_df = group_df.sort_values("threshold").reset_index(drop=True)
        row_base = dict(zip(group_cols, group_key))
        for metric_name in metrics:
            idx = group_df[metric_name].idxmax()
            best_row = group_df.loc[idx]
            rows.append(
                {
                    **row_base,
                    "target_metric": metric_name,
                    "best_threshold": float(best_row["threshold"]),
                    "best_value": float(best_row[metric_name]),
                    "precision": float(best_row["precision"]),
                    "recall": float(best_row["recall"]),
                    "f1": float(best_row["f1"]),
                    "balanced_accuracy": float(best_row["balanced_accuracy"]),
                    "mcc": float(best_row["mcc"]),
                    "specificity": float(best_row["specificity"]),
                    "accuracy": float(best_row["accuracy"]),
                }
            )
    return pd.DataFrame(rows)


def plot_roc_curves(
    models_probs: dict[str, tuple],
    y_true: np.ndarray,
    save_path: str = "results/models/roc_curves.png",
):
    """
    Overlay ROC curves for multiple models on one axes and save to disk.

    Parameters
    ----------
    models_probs : { model_name: (y_prob,) }
    y_true       : true labels
    save_path    : output file path
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_prob,) in models_probs.items():
        RocCurveDisplay.from_predictions(y_true, y_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [metrics] saved → {save_path}")


def plot_pr_curves(
    models_probs: dict[str, tuple],
    y_true: np.ndarray,
    save_path: str = "results/models/pr_curves.png",
):
    """
    Overlay Precision-Recall curves for multiple models and save to disk.

    Parameters
    ----------
    models_probs : { model_name: (y_prob,) }
    y_true       : true labels
    save_path    : output file path
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_prob,) in models_probs.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_prob,
                                                name=name, ax=ax)
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=0.8,
               label=f"Baseline (prevalence={baseline:.2f})")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [metrics] saved → {save_path}")


def plot_confusion_matrices(
    models_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: str = "results/models/confusion_matrices.png",
):
    """
    Grid of confusion matrices for all models.

    Parameters
    ----------
    models_preds : { model_name: y_pred }  (hard 0/1 predictions)
    y_true       : true labels
    save_path    : output file path
    """
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    n     = len(models_preds)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, (name, y_pred) in zip(axes_flat, models_preds.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            cm, display_labels=["No Default", "Default"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [metrics] saved → {save_path}")


def plot_calibration_curves(
    models_probs: dict[str, tuple],
    y_true: np.ndarray,
    n_bins: int = 15,
    save_path: str = "results/models/calibration_curves.png",
):
    """
    Reliability (calibration) diagram for multiple models.

    For each model plots fraction-of-positives vs. mean predicted probability
    per bin and annotates the ECE in the legend.

    Parameters
    ----------
    models_probs : { model_name: (y_prob,) }
    y_true       : true labels
    n_bins       : calibration bins (default 15)
    save_path    : output file path
    """
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    for name, (y_prob,) in models_probs.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, marker="o",
                label=f"{name} (ECE={ece:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves — All Models")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [metrics] saved → {save_path}")
