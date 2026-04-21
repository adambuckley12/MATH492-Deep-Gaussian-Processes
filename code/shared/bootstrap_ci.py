"""
bootstrap_ci.py — Bootstrap 95% confidence intervals for ROC-AUC and ECE.

Uses saved predictions.csv files (no retraining).  Outputs a tidy CSV of
[model, roc_auc_lo, roc_auc, roc_auc_hi, ece_lo, ece, ece_hi] for each
dataset/tier combination.

Usage
-----
    python bootstrap_ci.py           # prints results to stdout
    python bootstrap_ci.py --save    # also writes CI CSVs to asset dirs
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE   = os.path.dirname(os.path.abspath(__file__))
_UCI    = os.path.normpath(os.path.join(_HERE, "../credit-card-pd-uci/src/results"))
_FM     = os.path.normpath(os.path.join(_HERE, "../freddie-mac-pd/src/results"))
_ASSETS = os.path.normpath(os.path.join(_HERE, "../../sections/4-modeling/assets"))

DATASETS = {
    "credit-card-default": {
        "small":  (_UCI  + "/small/predictions/predictions.csv",  _ASSETS + "/credit-card-default/small"),
        "medium": (_UCI  + "/medium/predictions/predictions.csv", _ASSETS + "/credit-card-default/medium"),
        "large":  (_UCI  + "/large/predictions/predictions.csv",  _ASSETS + "/credit-card-default/large"),
    },
    "freddie-mac": {
        "small":  (_FM   + "/small/predictions/predictions.csv",  _ASSETS + "/freddie-mac/small"),
        "medium": (_FM   + "/medium/predictions/predictions.csv", _ASSETS + "/freddie-mac/medium"),
        "large":  (_FM   + "/large/predictions/predictions.csv",  _ASSETS + "/freddie-mac/large"),
    },
}

N_BOOT  = 2000
ALPHA   = 0.05
RNG     = np.random.default_rng(42)

# -- ECE helper ----------------------------------------------------------------

def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    # n_bins=15 to match classical/metrics.py expected_calibration_error default,
    # so bootstrap point estimates agree with the headline metrics tables.
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    total  = len(y_true)
    result = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac = mask.sum() / total
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        result += frac * abs(acc - conf)
    return result


# -- Bootstrap -----------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 n_boot: int = N_BOOT) -> dict:
    n = len(y_true)
    auc_boot = np.empty(n_boot)
    ece_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx        = RNG.integers(0, n, size=n)
        yt, yp     = y_true[idx], y_prob[idx]
        auc_boot[i] = roc_auc_score(yt, yp) if yt.sum() > 0 and yt.sum() < n else np.nan
        ece_boot[i] = ece(yt, yp)

    auc_boot = auc_boot[~np.isnan(auc_boot)]
    lo, hi   = np.percentile(auc_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    ece_lo, ece_hi = np.percentile(ece_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

    return dict(
        roc_auc_lo  = lo,
        roc_auc_hi  = hi,
        roc_auc_se  = auc_boot.std(),
        ece_lo      = ece_lo,
        ece_hi      = ece_hi,
        ece_se      = ece_boot.std(),
    )


# -- Main -----------------------------------------------------------------------

def run(save: bool = False):
    for ds_name, tiers in DATASETS.items():
        for tier, (pred_path, asset_dir) in tiers.items():
            if not os.path.exists(pred_path):
                print(f"[skip] {pred_path} not found")
                continue

            preds = pd.read_csv(pred_path)
            rows  = []
            for model, grp in preds.groupby("model"):
                y_true = grp["y_true"].to_numpy()
                y_prob = grp["y_prob"].to_numpy()

                point_auc = roc_auc_score(y_true, y_prob)
                point_ece = ece(y_true, y_prob)
                ci        = bootstrap_ci(y_true, y_prob)

                rows.append({
                    "model":       model,
                    "roc_auc":     round(point_auc, 4),
                    "roc_auc_lo":  round(ci["roc_auc_lo"], 4),
                    "roc_auc_hi":  round(ci["roc_auc_hi"], 4),
                    "roc_auc_se":  round(ci["roc_auc_se"], 4),
                    "ece":         round(point_ece, 4),
                    "ece_lo":      round(ci["ece_lo"], 4),
                    "ece_hi":      round(ci["ece_hi"], 4),
                    "ece_se":      round(ci["ece_se"], 4),
                })

            df = pd.DataFrame(rows)
            print(f"\n{'═'*70}")
            print(f"  {ds_name} — {tier}")
            print(f"{'═'*70}")
            print(df[["model","roc_auc","roc_auc_lo","roc_auc_hi",
                       "ece","ece_lo","ece_hi"]].to_string(index=False))

            if save:
                out = os.path.join(asset_dir, "bootstrap_ci.csv")
                df.to_csv(out, index=False)
                print(f"  → saved {out}")


if __name__ == "__main__":
    run(save="--save" in sys.argv)
