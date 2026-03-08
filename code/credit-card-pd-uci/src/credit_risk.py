"""
credit_risk.py
==============
Entry point for the credit-default prediction pipeline.

Usage
-----
    python credit_risk.py
    python credit_risk.py --quick
    python credit_risk.py --no-viz
    python credit_risk.py --results

Outputs
-------
    results/eda/            — EDA plots
    results/models/weights/ — saved raw model weights
    results/tables/         — metrics, runtimes, thresholds, split summaries
    results/predictions/    — test-set prediction exports for figure generation
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")

import random
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

signal.signal(signal.SIGINT, lambda *_: (print("\n[interrupted]"), os._exit(1)))

from classical import LogisticRegressionModel, RandomForestModel, XGBoostModel
from classical.calibration import fit_probability_calibrator
from classical.metrics import comparison_table, evaluate_predictions, summarize_threshold_optima, threshold_sweep
from data.data_cleaner import DataCleaner
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.data_visulisations import DataVisualisations
from dgp import DSVGPModel
from gp import SVGPModel

SEED = 42

QUICK = "--quick" in sys.argv
NO_VIZ = "--no-viz" in sys.argv
RESULTS_ONLY = "--results" in sys.argv
SMALL = "--small" in sys.argv

QUICK_FRAC = 0.25
# Total non-test data budget shared equally by all models in small-data mode.
# Classical models requiring post-hoc calibration must carve SMALL_N_CAL from this budget,
# leaving only (SMALL_BUDGET - SMALL_N_CAL) for training — the calibration tax.
SMALL_BUDGET = 1600
SMALL_N_CAL  = 400
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
EDA_DIR = os.path.join(RESULTS_DIR, "eda")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "small") if SMALL else os.path.join(RESULTS_DIR, "tables")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions", "small") if SMALL else os.path.join(RESULTS_DIR, "predictions")
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "models", "weights_small") if SMALL else os.path.join(RESULTS_DIR, "models", "weights")
DATASET_CSV = os.path.join(ROOT, "dataset", "credit_risk.csv")


@dataclass
class ExperimentResult:
    label: str
    base_model: str
    family: str
    stage: str
    calibration_method: str
    y_prob: np.ndarray
    train_time_sec: float


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in [EDA_DIR, TABLES_DIR, PREDICTIONS_DIR, WEIGHTS_DIR]:
        os.makedirs(path, exist_ok=True)


def maybe_subsample(
    X_part: pd.DataFrame,
    y_part: pd.Series,
    fraction: float,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(y_part) == 0 or fraction >= 1.0:
        return X_part, y_part
    X_small, _, y_small, _ = train_test_split(
        X_part,
        y_part,
        train_size=fraction,
        random_state=SEED,
        stratify=y_part,
    )
    return X_small.reset_index(drop=True), y_small.reset_index(drop=True)


def fit_with_timer(fit_fn):
    started = time.perf_counter()
    fitted = fit_fn()
    return fitted, time.perf_counter() - started


def maybe_load(path: str, loader):
    if RESULTS_ONLY and os.path.exists(path):
        return loader(path)
    return None


def tune_and_refit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_model_train: pd.DataFrame,
    y_model_train: pd.Series,
    *,
    use_scale_pos_weight: bool,
    quick: bool,
) -> tuple[XGBoostModel, float]:
    started = time.perf_counter()
    tuning_model = XGBoostModel(
        n_estimators=200 if quick else 1000,
        learning_rate=0.05,
        random_state=SEED,
        use_scale_pos_weight=use_scale_pos_weight,
    )
    tuning_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    best_iteration = getattr(tuning_model.model, "best_iteration", None)
    tuned_rounds = (int(best_iteration) + 1) if best_iteration is not None else (200 if quick else 1000)

    final_model = XGBoostModel(
        n_estimators=max(tuned_rounds, 10),
        learning_rate=0.05,
        random_state=SEED,
        use_scale_pos_weight=use_scale_pos_weight,
    )
    final_model.fit(X_model_train, y_model_train)
    return final_model, time.perf_counter() - started


def collect_result_row(result: ExperimentResult, y_true: np.ndarray) -> dict:
    metrics = evaluate_predictions(
        model_name=result.label,
        y_true=y_true,
        y_prob=result.y_prob,
        threshold=0.5,
        verbose=False,
    )
    return {
        "model": result.label,
        "base_model": result.base_model,
        "family": result.family,
        "stage": result.stage,
        "calibration_method": result.calibration_method,
        "train_time_sec": round(result.train_time_sec, 4),
        **metrics,
    }


def main() -> None:
    set_global_seed()
    ensure_dirs()

    mode_tag = " [QUICK MODE]" if QUICK else ""
    mode_tag += f" [SMALL DATA MODE — budget={SMALL_BUDGET}]" if SMALL else ""
    mode_tag += " [RESULTS ONLY WHERE POSSIBLE]" if RESULTS_ONLY else ""
    print(f"\n{'═' * 64}")
    print(f"  Credit Default Prediction Pipeline{mode_tag}")
    print(f"{'═' * 64}\n")

    data_loader = DataLoader(data_path=DATASET_CSV)
    df = data_loader.load_data()

    data_cleaner = DataCleaner(df)
    X_raw, y_raw = data_cleaner.clean(apply_log_transforms=True)

    if not NO_VIZ:
        viz = DataVisualisations(output_dir=EDA_DIR)
        viz.target_distribution(y_raw)
        viz.correlation_heatmap(X_raw)
        viz.feature_distributions(X_raw)
        viz.categorical_distributions(X_raw, y_raw)
        viz.boxplots_by_target(X_raw, y_raw)
        viz.temporal_trends(X_raw, y_raw)
        viz.repayment_status_heatmap(X_raw, y_raw)

    preprocessor = DataPreprocessor(
        test_size=0.2,
        calibration_size=0.2,
        validation_size=0.1,
        random_state=SEED,
    )
    splits = preprocessor.fit_transform_splits(X_raw, y_raw)

    if SMALL:
        # Pool all non-test data from the full split into a single fixed budget.
        # Calibration points are carved from WITHIN this budget — models needing post-hoc
        # calibration must sacrifice SMALL_N_CAL training observations for their calibrator.
        X_pool = pd.concat(
            [splits["train"][0], splits["validation"][0], splits["calibration"][0]], axis=0
        ).reset_index(drop=True)
        y_pool = pd.concat(
            [splits["train"][1], splits["validation"][1], splits["calibration"][1]], axis=0
        ).reset_index(drop=True)
        n_bud = min(SMALL_BUDGET, len(y_pool))
        _, X_bud, _, y_bud = train_test_split(
            X_pool, y_pool, test_size=n_bud, stratify=y_pool, random_state=SEED
        )
        X_bud = X_bud.reset_index(drop=True)
        y_bud = y_bud.reset_index(drop=True)
        # Carve calibration set from the budget (used only for penalized post-hoc classical).
        X_bud_model, X_cal_bud, y_bud_model, y_cal_bud = train_test_split(
            X_bud, y_bud, test_size=SMALL_N_CAL, stratify=y_bud, random_state=SEED
        )
        X_bud_model = X_bud_model.reset_index(drop=True)
        y_bud_model = y_bud_model.reset_index(drop=True)
        # Split remaining budget into train/val for XGBoost inner tuning.
        n_val_inner = max(1, round(len(y_bud_model) * 0.15))
        X_tr_inner, X_va_inner, y_tr_inner, y_va_inner = train_test_split(
            X_bud_model, y_bud_model, test_size=n_val_inner,
            stratify=y_bud_model, random_state=SEED,
        )
        splits["train"]       = (X_tr_inner.reset_index(drop=True), y_tr_inner.reset_index(drop=True))
        splits["validation"]  = (X_va_inner.reset_index(drop=True), y_va_inner.reset_index(drop=True))
        splits["calibration"] = (X_cal_bud.reset_index(drop=True), y_cal_bud.reset_index(drop=True))
        # test is unchanged (full 6 k)
        n_test = len(splits["test"][1])
        n_pen  = len(y_bud_model)
        print(
            f"  [SMALL MODE] Budget={n_bud}; GP/raw/pre-hoc training={n_bud} obs (full budget); "
            f"post-hoc classical training={n_pen} obs (+{SMALL_N_CAL} cal = {n_bud} total); "
            f"Test={n_test} (full)"
        )

    if QUICK:
        for split_name in ["train", "validation", "calibration", "test"]:
            X_part, y_part = splits[split_name]
            fraction = QUICK_FRAC if split_name == "train" else min(max(QUICK_FRAC * 1.5, 0.25), 1.0)
            splits[split_name] = maybe_subsample(X_part, y_part, fraction)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]

    split_rows = []
    for split_name, (_, y_part) in splits.items():
        split_rows.append(
            {
                "split": split_name,
                "rows": int(len(y_part)),
                "default_rate": float(np.asarray(y_part).mean()),
                "mode": "small" if SMALL else ("quick" if QUICK else "full"),
            }
        )
    split_summary_df = pd.DataFrame(split_rows)
    split_summary_df.to_csv(os.path.join(TABLES_DIR, "split_summary.csv"), index=False)
    print(split_summary_df.to_string(index=False))

    X_model_train = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_model_train = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    # In SMALL mode: GP and raw/pre-hoc classical use the full budget (train+val+cal).
    # Post-hoc classical models are retrained on X_model_train (penalized, without cal points)
    # and calibrated on X_cal — this is the calibration tax story.
    if SMALL:
        X_model_train_full = pd.concat([X_train, X_val, X_cal], axis=0).reset_index(drop=True)
        y_model_train_full = pd.concat([y_train, y_val, y_cal], axis=0).reset_index(drop=True)
    else:
        X_model_train_full = X_model_train
        y_model_train_full = y_model_train
    y_test_arr = y_test.to_numpy()
    feature_names = list(X_model_train.columns)

    results: list[ExperimentResult] = []
    prediction_rows: list[dict] = []
    threshold_rows: list[dict] = []
    feature_frames: list[pd.DataFrame] = []
    raw_classical_probs_cal: dict[str, np.ndarray] = {}
    raw_classical_probs_test: dict[str, np.ndarray] = {}

    print(f"\n{'═' * 64}")
    print("  Main Raw Model Suite")
    print(f"{'═' * 64}")

    lr_path = os.path.join(WEIGHTS_DIR, "logistic_regression.joblib")
    lr = maybe_load(lr_path, LogisticRegressionModel.load)
    lr_time = 0.0
    if lr is None:
        lr, lr_time = fit_with_timer(
            lambda: LogisticRegressionModel(C=1.0, random_state=SEED, class_weight="balanced").fit(
                X_model_train_full, y_model_train_full
            )
        )
        lr.save(lr_path) if not SMALL else None
    lr.name = "Logistic Regression"
    # In SMALL mode, cal probs are filled later from the penalized re-trained model.
    if not SMALL:
        raw_classical_probs_cal[lr.name] = lr.predict_proba(X_cal)
    raw_classical_probs_test[lr.name] = lr.predict_proba(X_test)
    results.append(
        ExperimentResult(
            label=lr.name,
            base_model="Logistic Regression",
            family="classical",
            stage="raw",
            calibration_method="none",
            y_prob=raw_classical_probs_test[lr.name],
            train_time_sec=lr_time,
        )
    )
    fi = lr.feature_importance(feature_names)
    fi["model"] = lr.name
    feature_frames.append(fi)

    rf_path = os.path.join(WEIGHTS_DIR, "random_forest.joblib")
    rf = maybe_load(rf_path, RandomForestModel.load)
    rf_time = 0.0
    if rf is None:
        rf, rf_time = fit_with_timer(
            lambda: RandomForestModel(
                n_estimators=100 if QUICK else 500,
                random_state=SEED,
                class_weight="balanced_subsample",
            ).fit(X_model_train_full, y_model_train_full)
        )
        rf.save(rf_path) if not SMALL else None
    rf.name = "Random Forest"
    if not SMALL:
        raw_classical_probs_cal[rf.name] = rf.predict_proba(X_cal)
    raw_classical_probs_test[rf.name] = rf.predict_proba(X_test)
    results.append(
        ExperimentResult(
            label=rf.name,
            base_model="Random Forest",
            family="classical",
            stage="raw",
            calibration_method="none",
            y_prob=raw_classical_probs_test[rf.name],
            train_time_sec=rf_time,
        )
    )
    fi = rf.feature_importance(feature_names)
    fi["model"] = rf.name
    feature_frames.append(fi)

    xgb_path = os.path.join(WEIGHTS_DIR, "xgboost.joblib")
    xgb = maybe_load(xgb_path, XGBoostModel.load)
    xgb_time = 0.0
    if xgb is None:
        xgb, xgb_time = tune_and_refit_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            X_model_train_full,
            y_model_train_full,
            use_scale_pos_weight=True,
            quick=QUICK,
        )
        xgb.save(xgb_path) if not SMALL else None
    xgb.name = "XGBoost"
    if not SMALL:
        raw_classical_probs_cal[xgb.name] = xgb.predict_proba(X_cal)
    raw_classical_probs_test[xgb.name] = xgb.predict_proba(X_test)
    results.append(
        ExperimentResult(
            label=xgb.name,
            base_model="XGBoost",
            family="classical",
            stage="raw",
            calibration_method="none",
            y_prob=raw_classical_probs_test[xgb.name],
            train_time_sec=xgb_time,
        )
    )
    fi = xgb.feature_importance(feature_names)
    fi["model"] = xgb.name
    feature_frames.append(fi)

    print(f"\n{'═' * 64}")
    print("  GP / DGP Raw Model Suite")
    print(f"{'═' * 64}")

    svgp_path = os.path.join(WEIGHTS_DIR, "svgp.pt")
    svgp = maybe_load(svgp_path, SVGPModel.load)
    svgp_time = 0.0
    if svgp is None:
        svgp, svgp_time = fit_with_timer(
            lambda: SVGPModel(
                num_inducing=64 if SMALL else (256 if QUICK else 512),
                n_epochs=75 if (QUICK or SMALL) else 150,
                batch_size=64 if SMALL else (256 if QUICK else 512),
                print_every=5 if QUICK else 10,
            ).fit(X_model_train_full, y_model_train_full)
        )
        if not SMALL:
            svgp.save(svgp_path)
    results.append(
        ExperimentResult(
            label=svgp.name,
            base_model=svgp.name,
            family="gp",
            stage="raw",
            calibration_method="none",
            y_prob=svgp.predict_proba(X_test),
            train_time_sec=svgp_time,
        )
    )

    for n_layers in [2, 3]:
        dgp_path = os.path.join(WEIGHTS_DIR, f"dsvgp_{n_layers}l.pt")
        dgp = maybe_load(dgp_path, DSVGPModel.load)
        dgp_time = 0.0
        if dgp is None:
            dgp, dgp_time = fit_with_timer(
                lambda n_layers=n_layers: DSVGPModel(
                    n_layers=n_layers,
                    hidden_dim=3 if SMALL else (4 if QUICK else 5),
                    num_inducing=32 if SMALL else (64 if QUICK else 128),
                    n_epochs=75 if (QUICK or SMALL) else 150,
                    batch_size=64 if SMALL else (256 if QUICK else 512),
                    num_samples=15 if (QUICK or SMALL) else 30,
                    print_every=5 if QUICK else 10,
                ).fit(X_model_train_full, y_model_train_full)
            )
            if not SMALL:
                dgp.save(dgp_path)
        results.append(
            ExperimentResult(
                label=dgp.name,
                base_model=dgp.name,
                family="dgp",
                stage="raw",
                calibration_method="none",
                y_prob=dgp.predict_proba(X_test),
                train_time_sec=dgp_time,
            )
        )

    print(f"\n{'═' * 64}")
    print("  Classical Pre-Hoc Variants")
    print(f"{'═' * 64}")

    lr_unweighted, lr_unweighted_time = fit_with_timer(
        lambda: LogisticRegressionModel(C=1.0, random_state=SEED, class_weight=None).fit(
            X_model_train_full, y_model_train_full
        )
    )
    lr_unweighted.name = "Logistic Regression (unweighted)"
    results.append(
        ExperimentResult(
            label=lr_unweighted.name,
            base_model="Logistic Regression",
            family="classical",
            stage="pre_hoc",
            calibration_method="unweighted",
            y_prob=lr_unweighted.predict_proba(X_test),
            train_time_sec=lr_unweighted_time,
        )
    )

    rf_unweighted, rf_unweighted_time = fit_with_timer(
        lambda: RandomForestModel(
            n_estimators=100 if QUICK else 500,
            random_state=SEED,
            class_weight=None,
        ).fit(X_model_train_full, y_model_train_full)
    )
    rf_unweighted.name = "Random Forest (unweighted)"
    results.append(
        ExperimentResult(
            label=rf_unweighted.name,
            base_model="Random Forest",
            family="classical",
            stage="pre_hoc",
            calibration_method="unweighted",
            y_prob=rf_unweighted.predict_proba(X_test),
            train_time_sec=rf_unweighted_time,
        )
    )

    xgb_unweighted, xgb_unweighted_time = tune_and_refit_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        X_model_train_full,
        y_model_train_full,
        use_scale_pos_weight=False,
        quick=QUICK,
    )
    xgb_unweighted.name = "XGBoost (unweighted)"
    results.append(
        ExperimentResult(
            label=xgb_unweighted.name,
            base_model="XGBoost",
            family="classical",
            stage="pre_hoc",
            calibration_method="unweighted",
            y_prob=xgb_unweighted.predict_proba(X_test),
            train_time_sec=xgb_unweighted_time,
        )
    )

    print(f"\n{'═' * 64}")
    print("  Classical Post-Hoc Calibration")
    print(f"{'═' * 64}")

    if SMALL:
        # In small mode, post-hoc classical models must be RETRAINED on the penalized budget
        # (X_model_train, which excludes X_cal).  This simulates the real calibration tax:
        # the 400 cal points were taken FROM the training budget, not provided for free.
        print(
            f"  [SMALL] Retraining classicals on penalized budget ({len(y_model_train)} obs) "
            f"+ {SMALL_N_CAL} cal carved from budget = {SMALL_BUDGET} total"
        )
        lr_pen = LogisticRegressionModel(C=1.0, random_state=SEED, class_weight="balanced").fit(
            X_model_train, y_model_train
        )
        raw_classical_probs_cal["Logistic Regression"] = lr_pen.predict_proba(X_cal)
        raw_classical_probs_test["Logistic Regression"] = lr_pen.predict_proba(X_test)

        rf_pen = RandomForestModel(
            n_estimators=100, random_state=SEED, class_weight="balanced_subsample"
        ).fit(X_model_train, y_model_train)
        raw_classical_probs_cal["Random Forest"] = rf_pen.predict_proba(X_cal)
        raw_classical_probs_test["Random Forest"] = rf_pen.predict_proba(X_test)

        xgb_pen, _ = tune_and_refit_xgboost(
            X_train, y_train, X_val, y_val,
            X_model_train, y_model_train,
            use_scale_pos_weight=True, quick=QUICK,
        )
        raw_classical_probs_cal["XGBoost"] = xgb_pen.predict_proba(X_cal)
        raw_classical_probs_test["XGBoost"] = xgb_pen.predict_proba(X_test)

    for base_model in ["Logistic Regression", "Random Forest", "XGBoost"]:
        cal_probs = raw_classical_probs_cal[base_model]
        test_probs = raw_classical_probs_test[base_model]
        for method in ["platt", "isotonic"]:
            calibrator, cal_time = fit_with_timer(
                lambda method=method, cal_probs=cal_probs: fit_probability_calibrator(
                    method,
                    cal_probs,
                    y_cal.to_numpy(),
                )
            )
            results.append(
                ExperimentResult(
                    label=f"{base_model} + {method.title()}",
                    base_model=base_model,
                    family="classical",
                    stage="post_hoc",
                    calibration_method=method,
                    y_prob=calibrator.predict(test_probs),
                    train_time_sec=cal_time,
                )
            )

    metric_rows = [collect_result_row(result, y_test_arr) for result in results]
    metrics_df = pd.DataFrame(metric_rows).sort_values(["stage", "family", "model"]).reset_index(drop=True)

    for result in results:
        y_prob = np.asarray(result.y_prob, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        prediction_rows.extend(
            {
                "model": result.label,
                "base_model": result.base_model,
                "family": result.family,
                "stage": result.stage,
                "calibration_method": result.calibration_method,
                "row_id": idx,
                "y_true": int(y_true_val),
                "y_prob": float(y_prob_val),
                "y_pred": int(y_pred_val),
            }
            for idx, (y_true_val, y_prob_val, y_pred_val) in enumerate(
                zip(y_test_arr, y_prob, y_pred),
                start=1,
            )
        )

        if result.stage in {"raw", "post_hoc"}:
            sweep_df = threshold_sweep(y_test_arr, y_prob)
            sweep_df.insert(0, "model", result.label)
            sweep_df.insert(1, "base_model", result.base_model)
            sweep_df.insert(2, "family", result.family)
            sweep_df.insert(3, "stage", result.stage)
            sweep_df.insert(4, "calibration_method", result.calibration_method)
            threshold_rows.extend(sweep_df.to_dict(orient="records"))

    predictions_df = pd.DataFrame(prediction_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    threshold_optima_df = summarize_threshold_optima(thresholds_df)
    feature_importance_df = pd.concat(feature_frames, axis=0, ignore_index=True)

    # Build a merged table: ranking/calibration metrics (from metrics_df) +
    # F1-optimal threshold classification metrics (from threshold_optima_df).
    # This avoids comparing models at a single arbitrary 0.5 cutoff.
    _opt_f1 = threshold_optima_df.loc[
        threshold_optima_df["target_metric"].eq("f1"),
        ["model", "base_model", "family", "stage", "calibration_method",
         "best_threshold", "best_value", "precision", "recall",
         "balanced_accuracy", "mcc", "specificity", "accuracy"],
    ].rename(columns={
        "best_threshold": "optimal_threshold",
        "best_value": "f1_optimal",
        "precision": "precision_optimal",
        "recall": "recall_optimal",
        "balanced_accuracy": "balanced_accuracy_optimal",
        "mcc": "mcc_optimal",
        "specificity": "specificity_optimal",
        "accuracy": "accuracy_optimal",
    })
    main_metrics_optimal_df = metrics_df.merge(
        _opt_f1,
        on=["model", "base_model", "family", "stage", "calibration_method"],
        how="left",
    )

    main_model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "SVGP",
        "DSVGP-2L",
        "DSVGP-3L",
    ]
    main_metrics_df = metrics_df.loc[metrics_df["model"].isin(main_model_names)].reset_index(drop=True)
    main_metrics_optimal_main_df = main_metrics_optimal_df.loc[
        main_metrics_optimal_df["model"].isin(main_model_names)
    ].reset_index(drop=True)
    classical_calibration_df = metrics_df.loc[metrics_df["family"].eq("classical")].reset_index(drop=True)

    metrics_df.to_csv(os.path.join(TABLES_DIR, "all_metrics.csv"), index=False)
    main_metrics_df.to_csv(os.path.join(TABLES_DIR, "main_metrics.csv"), index=False)
    main_metrics_optimal_main_df.to_csv(os.path.join(TABLES_DIR, "main_metrics_optimal.csv"), index=False)
    classical_calibration_df.to_csv(
        os.path.join(TABLES_DIR, "classical_calibration_metrics.csv"), index=False
    )
    thresholds_df.to_csv(os.path.join(TABLES_DIR, "threshold_sweep.csv"), index=False)
    threshold_optima_df.to_csv(os.path.join(TABLES_DIR, "threshold_optima.csv"), index=False)
    predictions_df.to_csv(os.path.join(PREDICTIONS_DIR, "test_predictions.csv"), index=False)
    feature_importance_df.to_csv(os.path.join(TABLES_DIR, "feature_importance.csv"), index=False)

    runtime_df = metrics_df[
        ["model", "base_model", "family", "stage", "calibration_method", "train_time_sec"]
    ].copy()
    runtime_df.to_csv(os.path.join(TABLES_DIR, "runtime_summary.csv"), index=False)

    print(f"\n{'═' * 64}")
    print("  Main Raw Summary")
    print(f"{'═' * 64}")
    print(comparison_table(main_metrics_df.set_index("model").to_dict(orient="index")).to_string())

    print(f"\n{'═' * 64}")
    print("  Classical Calibration Summary")
    print(f"{'═' * 64}")
    print(
        classical_calibration_df[
            ["model", "stage", "calibration_method", "roc_auc", "avg_prec", "brier", "log_loss", "ece"]
        ].to_string(index=False)
    )

    print("\nSaved tables:")
    for name in [
        "split_summary.csv",
        "all_metrics.csv",
        "main_metrics.csv",
        "main_metrics_optimal.csv",
        "classical_calibration_metrics.csv",
        "threshold_sweep.csv",
        "threshold_optima.csv",
        "runtime_summary.csv",
        "feature_importance.csv",
    ]:
        print(f"  - {os.path.join(TABLES_DIR, name)}")
    print(f"Saved predictions: {os.path.join(PREDICTIONS_DIR, 'test_predictions.csv')}")


if __name__ == "__main__":
    main()