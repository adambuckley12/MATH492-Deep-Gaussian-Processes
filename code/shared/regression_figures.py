"""
regression_figures.py — Visualisations for regression experiments.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = [
    "Ridge Regression", "Random Forest", "XGBoost",
    "SVGP", "DSVGP-2L", "DSVGP-3L",
]

PALETTE = {
    "Ridge Regression": "#4878CF",
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
    return s["y_true"].to_numpy(), s["y_mean"].to_numpy()


def plot_predicted_vs_actual(preds: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, m in zip(axes.ravel(), MODELS):
        if m not in preds["model"].values:
            ax.set_visible(False)
            continue
        yt, ym = _slice(preds, m)
        ax.scatter(yt, ym, alpha=0.15, s=8, color=PALETTE[m], rasterized=True)
        lo, hi = min(yt.min(), ym.min()), max(yt.max(), ym.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        rmse = np.sqrt(np.mean((yt - ym) ** 2))
        ax.set_title(f"{m} (RMSE={rmse:.4f})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "predicted_vs_actual.png"), bbox_inches="tight")
    plt.close(fig)


def plot_residuals(preds: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, m in zip(axes.ravel(), MODELS):
        if m not in preds["model"].values:
            ax.set_visible(False)
            continue
        yt, ym = _slice(preds, m)
        residuals = yt - ym
        ax.hist(residuals, bins=50, color=PALETTE[m], alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(m, fontsize=9, fontweight="bold")
        ax.set_xlabel("Residual")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "residuals.png"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics: pd.DataFrame, fig_dir: str) -> None:
    sub = metrics.loc[metrics["model"].isin(MODELS)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # RMSE
    for _, r in sub.iterrows():
        axes[0].barh(r["model"], r["rmse"], color=PALETTE.get(r["model"], "gray"),
                      edgecolor="white", linewidth=0.6)
    axes[0].set_title("RMSE (lower is better)")
    axes[0].invert_yaxis()

    # R²
    for _, r in sub.iterrows():
        axes[1].barh(r["model"], r["r2"], color=PALETTE.get(r["model"], "gray"),
                      edgecolor="white", linewidth=0.6)
    axes[1].set_title("R² (higher is better)")
    axes[1].invert_yaxis()

    # NLPD (only GP models have this)
    gp_sub = sub.dropna(subset=["nlpd"]) if "nlpd" in sub.columns else sub[0:0]
    if len(gp_sub) > 0:
        for _, r in gp_sub.iterrows():
            axes[2].barh(r["model"], r["nlpd"], color=PALETTE.get(r["model"], "gray"),
                          edgecolor="white", linewidth=0.6)
        axes[2].set_title("NLPD (lower is better)")
        axes[2].invert_yaxis()
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "metrics_bar.png"), bbox_inches="tight")
    plt.close(fig)


def plot_runtime(metrics: pd.DataFrame, fig_dir: str) -> None:
    sub = metrics.loc[metrics["model"].isin(MODELS)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in sub.iterrows():
        ax.scatter(r["train_time_sec"], r["rmse"], s=90,
                   color=PALETTE.get(r["model"]), edgecolor="white", linewidth=0.8)
        ax.text(r["train_time_sec"] * 1.05, r["rmse"] + 0.001, r["model"], fontsize=8)
    ax.set(xscale="log", xlabel="Training time (s)", ylabel="RMSE",
           title="Runtime vs RMSE")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "runtime.png"), bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(fi: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    specs = [
        ("Ridge Regression", "abs_coef", "#4878CF"),
        ("Random Forest", "importance", "#6ACC65"),
        ("XGBoost", "gain", "#D65F5F"),
    ]
    for ax, (model, col, color) in zip(axes, specs):
        sub = fi.loc[fi["model"].eq(model)]
        if col not in sub.columns or len(sub) == 0:
            ax.set_visible(False)
            continue
        top = sub[["feature", col]].nlargest(10, col).sort_values(col, ascending=True)
        ax.barh(top["feature"], top[col], color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(model, fontweight="bold", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_importance.png"), bbox_inches="tight")
    plt.close(fig)


def generate_all_regression_figures(
    tables_dir: str,
    predictions_dir: str,
    fig_dir: str,
) -> None:
    apply_style()
    os.makedirs(fig_dir, exist_ok=True)
    metrics = pd.read_csv(os.path.join(tables_dir, "metrics.csv"))
    preds = pd.read_csv(os.path.join(predictions_dir, "predictions.csv"))
    fi = pd.read_csv(os.path.join(tables_dir, "feature_importance.csv"))

    plot_predicted_vs_actual(preds, fig_dir)
    plot_residuals(preds, fig_dir)
    plot_metrics_bar(metrics, fig_dir)
    plot_runtime(metrics, fig_dir)
    plot_feature_importance(fi, fig_dir)
    print(f"  Figures → {fig_dir}")
