"""
pipeline.py — Shared binary-classification pipeline.

Trains six models (LR, RF, XGBoost, SVGP, DSVGP-2L, DSVGP-3L) on
train/validation/test splits and exports metrics, predictions, and
feature importances.

Classical models use class_weight="balanced" (LR, RF) or scale_pos_weight
(XGBoost) to handle class imbalance — the standard approach for
known-imbalanced credit-scoring data.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from base import PipelineConfig
from classical import LogisticRegressionModel, RandomForestModel, XGBoostModel
from classical.metrics import (
    evaluate_predictions,
    summarize_threshold_optima,
    threshold_sweep,
)
from dgp import DSVGPModel
from gp import SVGPModel
from utils import ensure_dirs, seed_everything, timed


# Internal helpers

def _find_val_threshold(y_val: np.ndarray, y_val_prob: np.ndarray) -> float:
    """Find F1-optimal threshold on validation predictions."""
    sw = threshold_sweep(y_val, y_val_prob)
    best_idx = sw["f1"].idxmax()
    return float(sw.loc[best_idx, "threshold"])


def _fit_xgboost(
    X_train, y_train, X_val, y_val,
    n_estimators: int, seed: int,
) -> tuple[XGBoostModel, XGBoostModel]:
    """Tune early-stopping rounds on val, then refit on train+val.

    Returns (tuner, final) — tuner is trained on X_train only (used for
    val-set threshold selection), final is retrained on train+val.
    """
    tuner = XGBoostModel(
        n_estimators=n_estimators, learning_rate=0.05,
        random_state=seed, use_scale_pos_weight=True,
    )
    tuner.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    best = getattr(tuner.model, "best_iteration", None)
    rounds = max((int(best) + 1) if best is not None else n_estimators, 10)

    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    final = XGBoostModel(
        n_estimators=rounds, learning_rate=0.05,
        random_state=seed, use_scale_pos_weight=True,
    )
    final.fit(X_full, y_full)
    return tuner, final

 
# Pipeline

def run_pipeline(
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
    ensure_dirs(cfg.tables_dir, cfg.predictions_dir, cfg.weights_dir)

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["validation"]
    X_test,  y_test  = splits["test"]
    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    y_test_np = y_test.to_numpy()
    features  = list(X_train.columns)

    print(f"\n{'═' * 64}")
    print(f"  {dataset_name} — {size_label}")
    print(f"{'═' * 64}\n")
    for name, (_, y) in splits.items():
        ya = np.asarray(y)
        print(f"  {name:>10}: {len(y):>6,} rows, {ya.mean():.2%} default rate")
    print()

    pd.DataFrame([
        {"split": name, "rows": len(y),
         "default_rate": float(np.asarray(y).mean()), "size": size_label}
        for name, (_, y) in splits.items()
    ]).to_csv(os.path.join(cfg.tables_dir, "split_summary.csv"), index=False)

    y_val_np = y_val.to_numpy()
    results = []
    feature_frames = []
    val_thresholds = {}
    pe = 5  # print_every for GP training logs

    # ── Stage 1: Train on X_train, predict X_val → find thresholds ──
    print(f"{'═' * 64}")
    print("  Stage 1: Threshold selection on validation set")
    print(f"{'═' * 64}")

    lr_s1 = LogisticRegressionModel(
        C=1.0, random_state=cfg.seed, class_weight="balanced",
    ).fit(X_train, y_train)
    val_thresholds[lr_s1.name] = _find_val_threshold(y_val_np, lr_s1.predict_proba(X_val))

    rf_s1 = RandomForestModel(
        n_estimators=cfg.rf_estimators, random_state=cfg.seed,
        class_weight="balanced_subsample",
    ).fit(X_train, y_train)
    val_thresholds[rf_s1.name] = _find_val_threshold(y_val_np, rf_s1.predict_proba(X_val))

    xgb_tuner = XGBoostModel(
        n_estimators=cfg.xgb_estimators, learning_rate=0.05,
        random_state=cfg.seed, use_scale_pos_weight=True,
    )
    xgb_tuner.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    val_thresholds[xgb_tuner.name] = _find_val_threshold(y_val_np, xgb_tuner.predict_proba(X_val))

    svgp_s1 = SVGPModel(
        num_inducing=cfg.svgp_inducing, n_epochs=cfg.svgp_epochs,
        batch_size=cfg.svgp_batch_size, print_every=pe,
    ).fit(X_train, y_train)
    val_thresholds[svgp_s1.name] = _find_val_threshold(y_val_np, svgp_s1.predict_proba(X_val))

    for n_layers in (2, 3):
        dgp_s1 = DSVGPModel(
            n_layers=n_layers, hidden_dim=cfg.dgp_hidden_dim,
            num_inducing=cfg.dgp_inducing, n_epochs=cfg.dgp_epochs,
            batch_size=cfg.dgp_batch_size,
            num_samples=cfg.dgp_num_samples, print_every=pe,
        ).fit(X_train, y_train)
        val_thresholds[dgp_s1.name] = _find_val_threshold(y_val_np, dgp_s1.predict_proba(X_val))

    print("\n  Validation-set thresholds:")
    for name, t in val_thresholds.items():
        print(f"    {name:25s}  t* = {t:.2f}")

    # Save val thresholds
    pd.DataFrame([
        {"model": k, "val_threshold": v} for k, v in val_thresholds.items()
    ]).to_csv(os.path.join(cfg.tables_dir, "val_thresholds.csv"), index=False)

    # ── Stage 2: Retrain on X_full, predict X_test ──
    print(f"\n{'═' * 64}")
    print("  Stage 2: Final training on train+val")
    print(f"{'═' * 64}")

    # Classical models
    print(f"\n  Classical Models")

    lr, lr_t = timed(
        lambda: LogisticRegressionModel(
            C=1.0, random_state=cfg.seed, class_weight="balanced",
        ).fit(X_full, y_full)
    )
    lr.save(os.path.join(cfg.weights_dir, "logistic_regression.joblib"))
    fi = lr.feature_importance(features); fi["model"] = lr.name
    feature_frames.append(fi)
    results.append({"model": lr.name, "family": "classical",
                    "y_prob": lr.predict_proba(X_test), "time": lr_t})

    rf, rf_t = timed(
        lambda: RandomForestModel(
            n_estimators=cfg.rf_estimators, random_state=cfg.seed,
            class_weight="balanced_subsample",
        ).fit(X_full, y_full)
    )
    rf.save(os.path.join(cfg.weights_dir, "random_forest.joblib"))
    fi = rf.feature_importance(features); fi["model"] = rf.name
    feature_frames.append(fi)
    results.append({"model": rf.name, "family": "classical",
                    "y_prob": rf.predict_proba(X_test), "time": rf_t})

    # XGBoost: use already-computed tuner for early-stopping rounds
    best = getattr(xgb_tuner.model, "best_iteration", None)
    xgb_rounds = max((int(best) + 1) if best is not None else cfg.xgb_estimators, 10)
    xgb, xgb_t = timed(
        lambda: XGBoostModel(
            n_estimators=xgb_rounds, learning_rate=0.05,
            random_state=cfg.seed, use_scale_pos_weight=True,
        ).fit(X_full, y_full)
    )
    xgb.save(os.path.join(cfg.weights_dir, "xgboost.joblib"))
    fi = xgb.feature_importance(features); fi["model"] = xgb.name
    feature_frames.append(fi)
    results.append({"model": xgb.name, "family": "classical",
                    "y_prob": xgb.predict_proba(X_test), "time": xgb_t})

    # GP / DGP models
    print(f"\n  GP / DGP Models")

    svgp, svgp_t = timed(
        lambda: SVGPModel(
            num_inducing=cfg.svgp_inducing, n_epochs=cfg.svgp_epochs,
            batch_size=cfg.svgp_batch_size, print_every=pe,
        ).fit(X_full, y_full)
    )
    svgp.save(os.path.join(cfg.weights_dir, "svgp.pt"))
    results.append({"model": svgp.name, "family": "gp",
                    "y_prob": svgp.predict_proba(X_test), "time": svgp_t})

    for n_layers in (2, 3):
        dgp, dgp_t = timed(
            lambda nl=n_layers: DSVGPModel(
                n_layers=nl, hidden_dim=cfg.dgp_hidden_dim,
                num_inducing=cfg.dgp_inducing, n_epochs=cfg.dgp_epochs,
                batch_size=cfg.dgp_batch_size,
                num_samples=cfg.dgp_num_samples, print_every=pe,
            ).fit(X_full, y_full)
        )
        dgp.save(os.path.join(cfg.weights_dir, f"dsvgp_{n_layers}l.pt"))
        results.append({"model": dgp.name, "family": "dgp",
                        "y_prob": dgp.predict_proba(X_test), "time": dgp_t})

    # Metrics & export — use val-chosen thresholds for threshold-dependent metrics
    metric_rows = []
    prediction_rows = []
    threshold_rows = []

    for r in results:
        yp = np.asarray(r["y_prob"], dtype=float)
        t_val = val_thresholds.get(r["model"], 0.5)
        m = evaluate_predictions(r["model"], y_test_np, yp, threshold=t_val, verbose=False)
        m["model"] = r["model"]
        m["family"] = r["family"]
        m["train_time_sec"] = round(r["time"], 4)
        m["val_threshold"] = t_val
        metric_rows.append(m)

        ypred = (yp >= t_val).astype(int)
        for i, (yt, pp, pd_) in enumerate(zip(y_test_np, yp, ypred), 1):
            prediction_rows.append({
                "model": r["model"], "family": r["family"],
                "row_id": i, "y_true": int(yt),
                "y_prob": float(pp), "y_pred": int(pd_),
            })

        # Test-set threshold sweep kept for diagnostic purposes
        sw = threshold_sweep(y_test_np, yp)
        sw.insert(0, "model", r["model"])
        sw.insert(1, "family", r["family"])
        threshold_rows.extend(sw.to_dict(orient="records"))

    metrics_df          = pd.DataFrame(metric_rows)
    predictions_df      = pd.DataFrame(prediction_rows)
    thresholds_df       = pd.DataFrame(threshold_rows)
    threshold_optima_df = summarize_threshold_optima(thresholds_df)
    feature_df          = pd.concat(feature_frames, ignore_index=True)

    metrics_df.to_csv(os.path.join(cfg.tables_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(cfg.predictions_dir, "predictions.csv"), index=False)
    thresholds_df.to_csv(os.path.join(cfg.tables_dir, "threshold_sweep.csv"), index=False)
    threshold_optima_df.to_csv(os.path.join(cfg.tables_dir, "threshold_optima.csv"), index=False)
    feature_df.to_csv(os.path.join(cfg.tables_dir, "feature_importance.csv"), index=False)
    metrics_df[["model", "family", "train_time_sec"]].to_csv(
        os.path.join(cfg.tables_dir, "runtime.csv"), index=False,
    )

    # Print summary
    print(f"\n{'═' * 64}")
    print("  Results")
    print(f"{'═' * 64}")
    print(metrics_df[["model", "roc_auc", "avg_prec", "f1", "brier",
                       "log_loss", "ece", "train_time_sec"]].to_string(index=False))
    print(f"\n  Saved → {cfg.tables_dir}")

    return metrics_df
