"""
regression_pipeline.py — Shared regression pipeline.

Trains six models (Ridge, RF, XGBoost, SVGP, DSVGP-2L, DSVGP-3L) on
train/validation/test splits and exports metrics, predictions, and
feature importances.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from base import PipelineConfig
from classical.ridge import RidgeRegressionModel
from classical.random_forest_reg import RandomForestRegressionModel
from classical.xgboost_reg import XGBoostRegressionModel
from dgp.dsvgp_regression import DSVGPRegressionModel
from gp.svgp_regression import SVGPRegressionModel
from utils import ensure_dirs, seed_everything, timed


# 
# Metrics
# 

def nlpd(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    """Negative log predictive density under a Gaussian predictive distribution."""
    y_var = np.maximum(y_var, 1e-6)
    nll = 0.5 * np.log(2 * np.pi * y_var) + 0.5 * (y_true - y_mean) ** 2 / y_var
    return float(np.mean(nll))


def evaluate_regression(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_var: np.ndarray | None = None,
) -> dict:
    """Return a dict of RMSE, MAE, R², and optionally NLPD."""
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_mean))),
        "mae":  float(mean_absolute_error(y_true, y_mean)),
        "r2":   float(r2_score(y_true, y_mean)),
    }
    if y_var is not None:
        metrics["nlpd"] = nlpd(y_true, y_mean, y_var)
    return metrics


# Pipeline

def run_regression_pipeline(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    cfg: PipelineConfig,
    dataset_name: str = "Dataset",
    size_label: str = "medium",
) -> pd.DataFrame:
    """
    Train all models and export results.

    Parameters
    ----------
    splits : dict with keys ``"train"``, ``"validation"``, ``"test"``
    cfg    : PipelineConfig — hyperparameters and output directories
    """
    seed_everything(cfg.seed)
    ensure_dirs(cfg.tables_dir, cfg.predictions_dir, cfg.weights_dir, cfg.figures_dir)

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["validation"]
    X_test,  y_test  = splits["test"]
    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    y_test_np = np.asarray(y_test, dtype=float)
    features  = list(X_train.columns)

    print(f"\n{'═' * 64}")
    print(f"  {dataset_name} — {size_label}")
    print(f"{'═' * 64}\n")
    for name, (_, y) in splits.items():
        print(f"  {name:>10}: {len(y):>6,} rows  mean={np.mean(y):.4f}  std={np.std(y):.4f}")
    print()

    pd.DataFrame([
        {"split": name, "rows": len(y), "y_mean": float(np.mean(y)),
         "y_std": float(np.std(y)), "size": size_label}
        for name, (_, y) in splits.items()
    ]).to_csv(os.path.join(cfg.tables_dir, "split_summary.csv"), index=False)

    # Classical models 
    print(f"{'═' * 64}")
    print("  Classical Models")
    print(f"{'═' * 64}")

    classical_models = [
        RidgeRegressionModel(alpha=1.0),
        RandomForestRegressionModel(n_estimators=cfg.rf_estimators, random_state=cfg.seed),
        XGBoostRegressionModel(random_state=cfg.seed),
    ]

    results = []
    feature_frames = []

    for model in classical_models:
        if isinstance(model, XGBoostRegressionModel):
            fitted, elapsed = timed(
                lambda m=model: m.fit(X_train, y_train, X_val, y_val)
            )
        else:
            fitted, elapsed = timed(lambda m=model: m.fit(X_full, y_full))

        y_pred = fitted.predict(X_test)
        fitted.save(os.path.join(cfg.weights_dir, f"{fitted.name.lower().replace(' ', '_')}.joblib"))

        fi = fitted.feature_importance(features)
        fi["model"] = fitted.name
        feature_frames.append(fi)

        results.append({
            "model": fitted.name, "family": "classical",
            "y_mean": y_pred, "y_var": None, "time": elapsed,
        })

    # GP / DGP models 
    print(f"\n{'═' * 64}")
    print("  GP / DGP Models")
    print(f"{'═' * 64}")

    pe = 5  # print_every for training logs

    svgp, svgp_t = timed(
        lambda: SVGPRegressionModel(
            num_inducing=cfg.svgp_inducing,
            n_epochs=cfg.svgp_epochs,
            batch_size=cfg.svgp_batch_size,
            print_every=pe,
        ).fit(X_full, y_full)
    )
    svgp_mean, svgp_var = svgp.predict(X_test)
    svgp.save(os.path.join(cfg.weights_dir, "svgp.pt"))
    results.append({
        "model": svgp.name, "family": "gp",
        "y_mean": svgp_mean, "y_var": svgp_var, "time": svgp_t,
    })

    for n_layers in (2, 3):
        dgp, dgp_t = timed(
            lambda nl=n_layers: DSVGPRegressionModel(
                n_layers=nl,
                hidden_dim=cfg.dgp_hidden_dim,
                num_inducing=cfg.dgp_inducing,
                n_epochs=cfg.dgp_epochs,
                batch_size=cfg.dgp_batch_size,
                num_samples=cfg.dgp_num_samples,
                print_every=pe,
            ).fit(X_full, y_full)
        )
        dgp_mean, dgp_var = dgp.predict(X_test)
        dgp.save(os.path.join(cfg.weights_dir, f"dsvgp_{n_layers}l.pt"))
        results.append({
            "model": dgp.name, "family": "dgp",
            "y_mean": dgp_mean, "y_var": dgp_var, "time": dgp_t,
        })

    # Metrics & export
    metric_rows = []
    prediction_rows = []

    for r in results:
        y_mean = np.asarray(r["y_mean"], dtype=float)
        y_var  = np.asarray(r["y_var"],  dtype=float) if r["y_var"] is not None else None
        m = evaluate_regression(y_test_np, y_mean, y_var)
        m["model"] = r["model"]
        m["family"] = r["family"]
        m["train_time_sec"] = round(r["time"], 4)
        metric_rows.append(m)

        for i, (yt, ym) in enumerate(zip(y_test_np, y_mean), 1):
            row = {
                "model": r["model"], "family": r["family"],
                "row_id": i, "y_true": float(yt), "y_mean": float(ym),
            }
            if y_var is not None:
                row["y_var"] = float(y_var[i - 1])
            prediction_rows.append(row)

    metrics_df     = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    feature_df     = pd.concat(feature_frames, ignore_index=True)

    metrics_df.to_csv(os.path.join(cfg.tables_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(cfg.predictions_dir, "predictions.csv"), index=False)
    feature_df.to_csv(os.path.join(cfg.tables_dir, "feature_importance.csv"), index=False)
    metrics_df[["model", "family", "train_time_sec"]].to_csv(
        os.path.join(cfg.tables_dir, "runtime.csv"), index=False,
    )

    # Print summary
    print(f"\n{'═' * 64}")
    print("  Results")
    print(f"{'═' * 64}")
    cols = ["model", "rmse", "mae", "r2", "train_time_sec"]
    if "nlpd" in metrics_df.columns:
        cols.insert(4, "nlpd")
    print(metrics_df[cols].to_string(index=False))
    print(f"\n  Saved → {cfg.tables_dir}")

    return metrics_df
