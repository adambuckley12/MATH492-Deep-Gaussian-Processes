"""
regression_pipeline.py — Shared regression pipeline.

Trains six models (Ridge, RF, XGBoost, SVGP, DSVGP-2L, DSVGP-3L) on
train/validation/test splits and exports metrics, predictions, and
feature importances.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from gp.svgp_regression import SVGPRegressionModel
from dgp.dsvgp_regression import DSVGPRegressionModel


@dataclass
class RegressionPipelineConfig:
    seed: int = 42
    tables_dir: str = ""
    predictions_dir: str = ""
    weights_dir: str = ""
    figures_dir: str = ""
    # GP / DGP
    svgp_inducing: int = 512
    svgp_epochs: int = 150
    svgp_batch_size: int = 512
    dgp_hidden_dim: int = 5
    dgp_inducing: int = 128
    dgp_epochs: int = 150
    dgp_batch_size: int = 512
    dgp_num_samples: int = 30
    # Classical
    rf_estimators: int = 500


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs(cfg: RegressionPipelineConfig) -> None:
    for d in (cfg.tables_dir, cfg.predictions_dir, cfg.weights_dir, cfg.figures_dir):
        if d:
            os.makedirs(d, exist_ok=True)


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def nlpd(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    """Negative log predictive density under a Gaussian predictive distribution."""
    y_var = np.maximum(y_var, 1e-6)
    nll = 0.5 * np.log(2 * np.pi * y_var) + 0.5 * (y_true - y_mean) ** 2 / y_var
    return float(np.mean(nll))


def evaluate_regression(y_true: np.ndarray, y_mean: np.ndarray,
                        y_var: np.ndarray | None = None) -> dict:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_mean)))
    mae = float(mean_absolute_error(y_true, y_mean))
    r2 = float(r2_score(y_true, y_mean))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    if y_var is not None:
        metrics["nlpd"] = nlpd(y_true, y_mean, y_var)
    return metrics


def _fit_xgb_regressor(X_train, y_train, X_val, y_val, seed):
    """Fit XGBoost regressor with early stopping, then refit on full data."""
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dval, "val")], early_stopping_rounds=20,
        verbose_eval=False,
    )
    best_rounds = model.best_iteration + 1

    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    dfull = xgb.DMatrix(X_full, label=y_full)
    final = xgb.train(params, dfull, num_boost_round=max(best_rounds, 10),
                       verbose_eval=False)
    return final


def run_regression_pipeline(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    cfg: RegressionPipelineConfig,
    dataset_name: str = "Dataset",
    size_label: str = "medium",
) -> pd.DataFrame:
    """Train all models and export results."""
    import xgboost as xgb

    _seed_everything(cfg.seed)
    _ensure_dirs(cfg)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    y_test_np = np.asarray(y_test, dtype=float)
    features = list(X_train.columns)

    print(f"\n{'═' * 64}")
    print(f"  {dataset_name} — {size_label}")
    print(f"{'═' * 64}\n")

    for name, (X, y) in splits.items():
        print(f"  {name:>10}: {len(y):>6,} rows, mean={np.mean(y):.4f}, std={np.std(y):.4f}")
    print()

    # Save split summary
    rows = []
    for name, (X, y) in splits.items():
        rows.append({"split": name, "rows": len(y),
                      "y_mean": float(np.mean(y)), "y_std": float(np.std(y)),
                      "size": size_label})
    pd.DataFrame(rows).to_csv(
        os.path.join(cfg.tables_dir, "split_summary.csv"), index=False,
    )

    results = []
    feature_frames = []

    pe = 5  # print_every for GP

    # -- Classical models --
    print(f"{'═' * 64}")
    print("  Classical Models")
    print(f"{'═' * 64}")

    # Ridge Regression
    ridge, ridge_t = _timed(
        lambda: Ridge(alpha=1.0).fit(X_full, y_full)
    )
    ridge_pred = ridge.predict(X_test)
    fi_ridge = pd.DataFrame({
        "feature": features,
        "abs_coef": np.abs(ridge.coef_),
        "model": "Ridge Regression",
    })
    feature_frames.append(fi_ridge)
    results.append({
        "model": "Ridge Regression", "family": "classical",
        "y_mean": ridge_pred, "y_var": None, "time": ridge_t,
    })

    # Random Forest
    rf, rf_t = _timed(
        lambda: RandomForestRegressor(
            n_estimators=cfg.rf_estimators, random_state=cfg.seed, n_jobs=-1,
        ).fit(X_full, y_full)
    )
    rf_pred = rf.predict(X_test)
    fi_rf = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
        "model": "Random Forest",
    })
    feature_frames.append(fi_rf)
    results.append({
        "model": "Random Forest", "family": "classical",
        "y_mean": rf_pred, "y_var": None, "time": rf_t,
    })

    # XGBoost
    xgb_model, xgb_t = _timed(
        lambda: _fit_xgb_regressor(X_train, y_train, X_val, y_val, cfg.seed)
    )
    dtest = xgb.DMatrix(X_test)
    xgb_pred = xgb_model.predict(dtest)
    importance = xgb_model.get_score(importance_type="gain")
    fi_xgb = pd.DataFrame({
        "feature": features,
        "gain": [importance.get(f, 0.0) for f in features],
        "model": "XGBoost",
    })
    feature_frames.append(fi_xgb)
    results.append({
        "model": "XGBoost", "family": "classical",
        "y_mean": xgb_pred, "y_var": None, "time": xgb_t,
    })

    # -- GP / DGP models --
    print(f"\n{'═' * 64}")
    print("  GP / DGP Models")
    print(f"{'═' * 64}")

    svgp, svgp_t = _timed(
        lambda: SVGPRegressionModel(
            num_inducing=cfg.svgp_inducing, n_epochs=cfg.svgp_epochs,
            batch_size=cfg.svgp_batch_size, print_every=pe,
        ).fit(X_full, y_full)
    )
    svgp_mean, svgp_var = svgp.predict(X_test)
    svgp.save(os.path.join(cfg.weights_dir, "svgp.pt"))
    results.append({
        "model": svgp.name, "family": "gp",
        "y_mean": svgp_mean, "y_var": svgp_var, "time": svgp_t,
    })

    for nl in (2, 3):
        dgp, dgp_t = _timed(
            lambda nl=nl: DSVGPRegressionModel(
                n_layers=nl, hidden_dim=cfg.dgp_hidden_dim,
                num_inducing=cfg.dgp_inducing, n_epochs=cfg.dgp_epochs,
                batch_size=cfg.dgp_batch_size,
                num_samples=cfg.dgp_num_samples, print_every=pe,
            ).fit(X_full, y_full)
        )
        dgp_mean, dgp_var = dgp.predict(X_test)
        dgp.save(os.path.join(cfg.weights_dir, f"dsvgp_{nl}l.pt"))
        results.append({
            "model": dgp.name, "family": "dgp",
            "y_mean": dgp_mean, "y_var": dgp_var, "time": dgp_t,
        })

    # -- Metrics & export --
    metric_rows = []
    prediction_rows = []

    for r in results:
        y_mean = np.asarray(r["y_mean"], dtype=float)
        y_var = np.asarray(r["y_var"], dtype=float) if r["y_var"] is not None else None
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

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    feature_df = pd.concat(feature_frames, ignore_index=True)

    metrics_df.to_csv(os.path.join(cfg.tables_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(cfg.predictions_dir, "predictions.csv"), index=False)
    feature_df.to_csv(os.path.join(cfg.tables_dir, "feature_importance.csv"), index=False)
    metrics_df[["model", "family", "train_time_sec"]].to_csv(
        os.path.join(cfg.tables_dir, "runtime.csv"), index=False,
    )

    # -- Print summary --
    print(f"\n{'═' * 64}")
    print("  Results")
    print(f"{'═' * 64}")
    cols = ["model", "rmse", "mae", "r2", "train_time_sec"]
    if "nlpd" in metrics_df.columns:
        cols.insert(4, "nlpd")
    print(metrics_df[cols].to_string(index=False))
    print(f"\n  Saved → {cfg.tables_dir}")

    return metrics_df
