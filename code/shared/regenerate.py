"""
regenerate.py — Recompute metrics & figures from saved predictions.

Loads predictions.csv (which has y_prob), recomputes:
  - metrics at t=0.5  (backwards-compat)
  - metrics at F1-optimal threshold per model
  - threshold sweep
  - confusion matrices at optimal threshold
  - all figures

Does NOT retrain any models.

Usage
-----
    python regenerate.py --dataset credit-card --tier large
    python regenerate.py --dataset freddie-mac --tier small medium large
    python regenerate.py --dataset all --tier all          # everything
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

# -- Path setup --------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from classical.metrics import expected_calibration_error

# -- Constants ---------------------------------------------------------------------
MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "SVGP",
    "DSVGP-2L",
    "DSVGP-3L",
]
CLASSICAL = MODELS[:3]
GP_MODELS = MODELS[3:]

PALETTE = {
    "Logistic Regression": "#4878CF",
    "Random Forest": "#6ACC65",
    "XGBoost": "#D65F5F",
    "SVGP": "#B47CC7",
    "DSVGP-2L": "#C4AD66",
    "DSVGP-3L": "#77BEDB",
}

DATASETS = {
    "credit-card": os.path.join(_HERE, "..", "credit-card-pd-uci", "src", "results"),
    "freddie-mac": os.path.join(_HERE, "..", "freddie-mac-pd", "src", "results"),
}
TIERS = ["small", "medium", "large"]

# Where to copy final figures for the dissertation
ASSET_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "sections", "4-modeling", "assets"))
ASSET_MAP = {
    "credit-card": "credit-card-default",
    "freddie-mac": "freddie-mac",
}


def _apply_style():
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


def _slice(preds: pd.DataFrame, model: str):
    s = preds.loc[preds["model"].eq(model)].sort_values("row_id")
    return s["y_true"].to_numpy(), s["y_prob"].to_numpy()


# -- Metric computation ------------------------------------------------------------

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_prec": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6)),
        "ece": expected_calibration_error(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": threshold,
    }


def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """Find threshold in [0.01, 0.99] maximising the given metric."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_val = 0.5, -1
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, t)
        if m[metric] > best_val:
            best_val = m[metric]
            best_t = t
    # Fine-grained search around best
    for t in np.linspace(max(0.01, best_t - 0.05), min(0.99, best_t + 0.05), 50):
        m = compute_metrics(y_true, y_prob, t)
        if m[metric] > best_val:
            best_val = m[metric]
            best_t = t
    return round(best_t, 3)


def threshold_sweep(y_true, y_prob):
    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        m = compute_metrics(y_true, y_prob, float(t))
        rows.append(m)
    return pd.DataFrame(rows)


# -- Figure generation -------------------------------------------------------------

def plot_roc_pr(preds, fig_dir):
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


def plot_calibration(preds, fig_dir, strategy="quantile"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, grp, title in [(ax1, CLASSICAL, "Classical"), (ax2, GP_MODELS, "GP / DGP")]:
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


def plot_confusion(preds, fig_dir, optimal_thresholds):
    """Plot confusion matrices at each model's F1-optimal threshold."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, m in zip(axes.ravel(), MODELS):
        yt, yp = _slice(preds, m)
        t = optimal_thresholds.get(m, 0.5)
        y_pred = (yp >= t).astype(int)
        cm = confusion_matrix(yt, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["No Def.", "Default"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{m}\n($t^* = {t:.2f}$)", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion.png"), bbox_inches="tight")
    plt.close(fig)


def plot_runtime(metrics_df, fig_dir):
    sub = metrics_df.loc[metrics_df["model"].isin(MODELS)]
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


def plot_threshold(thresh_df, fig_dir, optimal_thresholds):
    """Threshold sweep with optimal points marked."""
    sub = thresh_df.loc[thresh_df["model"].isin(MODELS)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for m in MODELS:
        rows = sub.loc[sub["model"].eq(m)].sort_values("threshold")
        ax1.plot(rows["threshold"], rows["f1"], color=PALETTE[m], label=m)
        ax2.plot(rows["threshold"], rows["balanced_accuracy"], color=PALETTE[m], label=m)
        # Mark optimal
        t_opt = optimal_thresholds.get(m, 0.5)
        best_row = rows.iloc[(rows["threshold"] - t_opt).abs().argsort()[:1]]
        if not best_row.empty:
            ax1.scatter([t_opt], best_row["f1"].values, color=PALETTE[m],
                        s=60, zorder=5, edgecolor="black", linewidth=0.8)
    ax1.set(title="F1 vs Threshold", xlabel="Threshold", ylabel="F1")
    ax2.set(title="Balanced Acc. vs Threshold", xlabel="Threshold", ylabel="Balanced accuracy")
    ax2.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "threshold.png"), bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(fi_df, fig_dir):
    if fi_df is None or fi_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    specs = [
        ("Logistic Regression", "abs_coef", "#4878CF"),
        ("Random Forest", "importance", "#6ACC65"),
        ("XGBoost", "gain", "#D65F5F"),
    ]
    for ax, (model, col, color) in zip(axes, specs):
        sub = fi_df.loc[fi_df["model"].eq(model)]
        if col not in sub.columns:
            continue
        top = sub[["feature", col]].nlargest(10, col).sort_values(col, ascending=True)
        ax.barh(top["feature"], top[col], color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(model, fontweight="bold", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_importance.png"), bbox_inches="tight")
    plt.close(fig)


# -- Main regeneration logic -------------------------------------------------------

def regenerate_tier(dataset_key, tier, results_root):
    """Regenerate metrics and figures for one dataset-tier pair."""
    tier_dir = os.path.join(results_root, tier)
    tables_dir = os.path.join(tier_dir, "tables")
    preds_dir = os.path.join(tier_dir, "predictions")
    fig_dir = os.path.join(tier_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    preds_path = os.path.join(preds_dir, "predictions.csv")
    if not os.path.exists(preds_path):
        print(f"  [{dataset_key}/{tier}] No predictions found, skipping.")
        return

    print(f"\n{'═' * 60}")
    print(f"  Regenerating: {dataset_key} / {tier}")
    print(f"{'═' * 60}")

    preds = pd.read_csv(preds_path)

    # Determine calibration strategy
    cal_strategy = "uniform" if dataset_key == "freddie-mac" else "quantile"

    # Load runtime from existing metrics if available
    old_metrics_path = os.path.join(tables_dir, "metrics.csv")
    runtime_map = {}
    if os.path.exists(old_metrics_path):
        old_m = pd.read_csv(old_metrics_path)
        if "train_time_sec" in old_m.columns:
            runtime_map = dict(zip(old_m["model"], old_m["train_time_sec"]))

    # -- Compute metrics ------------------------------------------
    metric_rows_fixed = []
    metric_rows_optimal = []
    optimal_thresholds = {}
    all_thresh_rows = []

    for model in MODELS:
        yt, yp = _slice(preds, model)
        family = "classical" if model in CLASSICAL else ("gp" if model == "SVGP" else "dgp")

        # Fixed threshold = 0.5
        m_fixed = compute_metrics(yt, yp, 0.5)
        m_fixed["model"] = model
        m_fixed["family"] = family
        m_fixed["train_time_sec"] = runtime_map.get(model, 0)
        metric_rows_fixed.append(m_fixed)

        # Optimal F1 threshold
        t_opt = find_optimal_threshold(yt, yp, metric="f1")
        optimal_thresholds[model] = t_opt
        m_opt = compute_metrics(yt, yp, t_opt)
        m_opt["model"] = model
        m_opt["family"] = family
        m_opt["train_time_sec"] = runtime_map.get(model, 0)
        m_opt["optimal_threshold"] = t_opt
        metric_rows_optimal.append(m_opt)

        print(f"  {model:25s}  t*={t_opt:.3f}  F1={m_opt['f1']:.4f}  "
              f"ROC-AUC={m_opt['roc_auc']:.4f}  ECE={m_opt['ece']:.4f}")

        # Threshold sweep
        sw = threshold_sweep(yt, yp)
        sw.insert(0, "model", model)
        sw.insert(1, "family", family)
        all_thresh_rows.append(sw)

    metrics_fixed = pd.DataFrame(metric_rows_fixed)
    metrics_optimal = pd.DataFrame(metric_rows_optimal)
    thresholds_df = pd.concat(all_thresh_rows, ignore_index=True)

    # Update y_pred in predictions to use optimal thresholds
    preds_updated = preds.copy()
    for model in MODELS:
        mask = preds_updated["model"] == model
        t = optimal_thresholds[model]
        preds_updated.loc[mask, "y_pred"] = (
            preds_updated.loc[mask, "y_prob"] >= t
        ).astype(int)

    # Save
    metrics_fixed.to_csv(os.path.join(tables_dir, "metrics.csv"), index=False)
    metrics_optimal.to_csv(os.path.join(tables_dir, "metrics_optimal.csv"), index=False)
    thresholds_df.to_csv(os.path.join(tables_dir, "threshold_sweep.csv"), index=False)
    preds_updated.to_csv(os.path.join(preds_dir, "predictions.csv"), index=False)

    # Threshold optima summary
    optima_rows = []
    for model in MODELS:
        yt, yp = _slice(preds, model)
        family = "classical" if model in CLASSICAL else ("gp" if model == "SVGP" else "dgp")
        for target_metric in ["f1", "balanced_accuracy", "mcc"]:
            t = find_optimal_threshold(yt, yp, metric=target_metric)
            m = compute_metrics(yt, yp, t)
            optima_rows.append({
                "model": model, "family": family,
                "target_metric": target_metric,
                "best_threshold": t, "best_value": m[target_metric],
                **{k: m[k] for k in ["f1", "precision", "recall",
                                       "balanced_accuracy", "mcc", "specificity", "accuracy"]},
            })
    pd.DataFrame(optima_rows).to_csv(
        os.path.join(tables_dir, "threshold_optima.csv"), index=False)
    metrics_fixed[["model", "family", "train_time_sec"]].to_csv(
        os.path.join(tables_dir, "runtime.csv"), index=False)

    # Load feature importance if exists
    fi_path = os.path.join(tables_dir, "feature_importance.csv")
    fi_df = pd.read_csv(fi_path) if os.path.exists(fi_path) else None

    # -- Generate figures -----------------------------------------
    _apply_style()
    plot_roc_pr(preds, fig_dir)
    plot_calibration(preds, fig_dir, strategy=cal_strategy)
    plot_confusion(preds_updated, fig_dir, optimal_thresholds)
    plot_runtime(metrics_fixed, fig_dir)
    plot_threshold(thresholds_df, fig_dir, optimal_thresholds)
    if fi_df is not None:
        plot_feature_importance(fi_df, fig_dir)
    print(f"  Figures → {fig_dir}")

    # -- Copy figures to dissertation assets -----------------------
    asset_dir = os.path.join(ASSET_ROOT, ASSET_MAP[dataset_key], tier)
    os.makedirs(asset_dir, exist_ok=True)
    for fname in os.listdir(fig_dir):
        if fname.endswith(".png"):
            shutil.copy2(os.path.join(fig_dir, fname), os.path.join(asset_dir, fname))
    # Also copy key CSVs
    for csv_name in ["metrics.csv", "metrics_optimal.csv", "threshold_optima.csv", "split_summary.csv"]:
        src = os.path.join(tables_dir, csv_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(asset_dir, csv_name))
    print(f"  Assets → {asset_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--tier", nargs="+", default=["all"])
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if "all" in args.dataset else args.dataset
    tiers = TIERS if "all" in args.tier else args.tier

    _apply_style()
    for ds in datasets:
        root = os.path.normpath(DATASETS[ds])
        for tier in tiers:
            regenerate_tier(ds, tier, root)

    print("\n✓ Regeneration complete.")


if __name__ == "__main__":
    main()
