"""
pipeline.py — Shared binary-classification PD pipeline.

Trains six models (LR, RF, XGBoost, SVGP, DSVGP-2L, DSVGP-3L) on
train/validation/test splits and exports metrics, predictions, and
feature importances.

Classical models use class_weight="balanced" (LR, RF) or
scale_pos_weight (XGBoost) to handle class imbalance — this is the
standard approach for known-imbalanced credit scoring data.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from classical import LogisticRegressionModel, RandomForestModel, XGBoostModel
from classical.metrics import (
    comparison_table,
    evaluate_predictions,
    summarize_threshold_optima,
    threshold_sweep,
)
from dgp import DSVGPModel
from gp import SVGPModel


# -- Configuration -------------------------------------------------------------

@dataclass
class PipelineConfig:
    seed: int = 42
    tables_dir: str = ""
    predictions_dir: str = ""
    weights_dir: str = ""
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
    xgb_estimators: int = 1000


# -- Helpers -------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs(cfg: PipelineConfig) -> None:
    for d in (cfg.tables_dir, cfg.predictions_dir, cfg.weights_dir):
        os.makedirs(d, exist_ok=True)


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def _fit_xgboost(X_train, y_train, X_val, y_val, n_est, seed):
    """Tune early-stopping rounds on val, then refit on train+val."""
    tuner = XGBoostModel(
        n_estimators=n_est, learning_rate=0.05,
        random_state=seed, use_scale_pos_weight=True,
    )
    tuner.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    best = getattr(tuner.model, "best_iteration", None)
    rounds = max((int(best) + 1) if best is not None else n_est, 10)

    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    final = XGBoostModel(
        n_estimators=rounds, learning_rate=0.05,
        random_state=seed, use_scale_pos_weight=True,
    )
    final.fit(X_full, y_full)
    return final


# -- Main pipeline -------------------------------------------------------------

def run_pipeline(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    cfg: PipelineConfig,
    dataset_name: str = "Dataset",
    size_label: str = "medium",
) -> pd.DataFrame:
    """
    Train all models and export results.

    splits : dict with keys "train", "validation", "test"
    """
    _seed_everything(cfg.seed)
    _ensure_dirs(cfg)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    y_test_np = y_test.to_numpy()
    features = list(X_train.columns)

    print(f"\n{'═' * 64}")
    print(f"  {dataset_name} — {size_label}")
    print(f"{'═' * 64}\n")

    for name, (_, y) in splits.items():
        ya = np.asarray(y)
        print(f"  {name:>10}: {len(y):>6,} rows, {ya.mean():.2%} default rate")
    print()

    # Save split summary
    rows = []
    for name, (_, y) in splits.items():
        rows.append({"split": name, "rows": len(y),
                      "default_rate": float(np.asarray(y).mean()),
                      "size": size_label})
    pd.DataFrame(rows).to_csv(
        os.path.join(cfg.tables_dir, "split_summary.csv"), index=False,
    )

    results = []  # list of dicts
    feature_frames = []

    pe = 5  # print_every for GP training logs

    # -- Classical models ------------------------------------------------------
    print(f"{'═' * 64}")
    print("  Classical Models")
    print(f"{'═' * 64}")

    # Logistic Regression
    lr_path = os.path.join(cfg.weights_dir, "logistic_regression.joblib")
    lr, lr_t = _timed(
        lambda: LogisticRegressionModel(
            C=1.0, random_state=cfg.seed, class_weight="balanced",
        ).fit(X_full, y_full)
    )
    lr.save(lr_path)
    lr.name = "Logistic Regression"
    fi = lr.feature_importance(features); fi["model"] = lr.name
    feature_frames.append(fi)
    results.append({"model": lr.name, "family": "classical",
                     "y_prob": lr.predict_proba(X_test), "time": lr_t})

    # Random Forest
    rf_path = os.path.join(cfg.weights_dir, "random_forest.joblib")
    rf, rf_t = _timed(
        lambda: RandomForestModel(
            n_estimators=cfg.rf_estimators, random_state=cfg.seed,
            class_weight="balanced_subsample",
        ).fit(X_full, y_full)
    )
    rf.save(rf_path)
    rf.name = "Random Forest"
    fi = rf.feature_importance(features); fi["model"] = rf.name
    feature_frames.append(fi)
    results.append({"model": rf.name, "family": "classical",
                     "y_prob": rf.predict_proba(X_test), "time": rf_t})

    # XGBoost
    xgb_path = os.path.join(cfg.weights_dir, "xgboost.joblib")
    xgb, xgb_t = _timed(
        lambda: _fit_xgboost(
            X_train, y_train, X_val, y_val,
            cfg.xgb_estimators, cfg.seed,
        )
    )
    xgb.save(xgb_path)
    xgb.name = "XGBoost"
    fi = xgb.feature_importance(features); fi["model"] = xgb.name
    feature_frames.append(fi)
    results.append({"model": xgb.name, "family": "classical",
                     "y_prob": xgb.predict_proba(X_test), "time": xgb_t})

    # -- GP / DGP models ------------------------------------------------------
    print(f"\n{'═' * 64}")
    print("  GP / DGP Models")
    print(f"{'═' * 64}")

    svgp_path = os.path.join(cfg.weights_dir, "svgp.pt")
    svgp, svgp_t = _timed(
        lambda: SVGPModel(
            num_inducing=cfg.svgp_inducing, n_epochs=cfg.svgp_epochs,
            batch_size=cfg.svgp_batch_size, print_every=pe,
        ).fit(X_full, y_full)
    )
    svgp.save(svgp_path)
    results.append({"model": svgp.name, "family": "gp",
                     "y_prob": svgp.predict_proba(X_test), "time": svgp_t})

    for nl in (2, 3):
        dgp_path = os.path.join(cfg.weights_dir, f"dsvgp_{nl}l.pt")
        dgp, dgp_t = _timed(
            lambda nl=nl: DSVGPModel(
                n_layers=nl, hidden_dim=cfg.dgp_hidden_dim,
                num_inducing=cfg.dgp_inducing, n_epochs=cfg.dgp_epochs,
                batch_size=cfg.dgp_batch_size,
                num_samples=cfg.dgp_num_samples, print_every=pe,
            ).fit(X_full, y_full)
        )
        dgp.save(dgp_path)
        results.append({"model": dgp.name, "family": "dgp",
                         "y_prob": dgp.predict_proba(X_test), "time": dgp_t})

    # -- Metrics & export ------------------------------------------------------
    metric_rows = []
    prediction_rows = []
    threshold_rows = []

    for r in results:
        yp = np.asarray(r["y_prob"], dtype=float)
        m = evaluate_predictions(r["model"], y_test_np, yp, threshold=0.5, verbose=False)
        m["model"] = r["model"]
        m["family"] = r["family"]
        m["train_time_sec"] = round(r["time"], 4)
        metric_rows.append(m)

        ypred = (yp >= 0.5).astype(int)
        for i, (yt, pp, pd_) in enumerate(zip(y_test_np, yp, ypred), 1):
            prediction_rows.append({
                "model": r["model"], "family": r["family"],
                "row_id": i, "y_true": int(yt),
                "y_prob": float(pp), "y_pred": int(pd_),
            })

        sw = threshold_sweep(y_test_np, yp)
        sw.insert(0, "model", r["model"])
        sw.insert(1, "family", r["family"])
        threshold_rows.extend(sw.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    threshold_optima_df = summarize_threshold_optima(thresholds_df)
    feature_df = pd.concat(feature_frames, ignore_index=True)

    metrics_df.to_csv(os.path.join(cfg.tables_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(cfg.predictions_dir, "predictions.csv"), index=False)
    thresholds_df.to_csv(os.path.join(cfg.tables_dir, "threshold_sweep.csv"), index=False)
    threshold_optima_df.to_csv(os.path.join(cfg.tables_dir, "threshold_optima.csv"), index=False)
    feature_df.to_csv(os.path.join(cfg.tables_dir, "feature_importance.csv"), index=False)
    metrics_df[["model", "family", "train_time_sec"]].to_csv(
        os.path.join(cfg.tables_dir, "runtime.csv"), index=False,
    )

    # -- Print summary ---------------------------------------------------------
    print(f"\n{'═' * 64}")
    print("  Results")
    print(f"{'═' * 64}")
    print(metrics_df[["model", "roc_auc", "avg_prec", "f1", "brier",
                       "log_loss", "ece", "train_time_sec"]].to_string(index=False))
    print(f"\n  Saved → {cfg.tables_dir}")

    return metrics_df
