# Results

## Experimental Setup

These results are from **QUICK_MODE** (stratified 10% subsample), with approximately 2,100 training samples, 300 validation samples, and 600 test samples at a ~22.2% default base rate. 

This run uses:
- **Natural sampling** (`shuffle=True`) for GP/DGP mini-batches (no class-balanced sampler)
- **Meaningful-improvement deltas** for LR decay and early stopping (`EARLY_STOP_MIN_DELTA=1e-3`, `LR_PLATEAU_MIN_DELTA=1e-3`)
- **DGP-3 disabled in QUICK_MODE**, so only DGP-2 is trained among deep GP variants

---

## 5.1 Discriminative Performance

| Model | ROC-AUC | PR-AUC | F1 | Balanced Acc. |
|---|---:|---:|---:|---:|
| **DGP-2** | **0.7457** | **0.5136** | 0.4124 | 0.6279 |
| **SVGP** | 0.7439 | 0.5004 | 0.4264 | 0.6343 |
| Random Forest | 0.7394 | 0.4730 | 0.4434 | 0.6424 |
| XGBoost | 0.7264 | 0.4987 | **0.4818** | **0.6706** |
| Logistic Regression | 0.6964 | 0.4728 | 0.4734 | 0.6669 |

*Table 1: Discriminative metrics on the held-out test set.*

**Key observations:**
1. **DGP-2 is now best on ranking metrics** (ROC-AUC and PR-AUC), with SVGP a close second.
2. **XGBoost still leads at threshold metrics** (F1 and balanced accuracy at threshold 0.5), indicating a different operating-point trade-off.
3. **Logistic Regression remains the weakest ranker** by ROC-AUC, consistent with missing non-linear structure.

---

## 5.2 Calibration

| Model | Brier ↓ | Log-loss ↓ | ECE ↓ |
|---|---:|---:|---:|
| **DGP-2** | **0.1419** | **0.4503** | 0.0332 |
| **SVGP** | 0.1441 | 0.4543 | **0.0313** |
| Random Forest | 0.1480 | 0.4615 | 0.0613 |
| XGBoost | 0.1850 | 0.5558 | 0.1737 |
| Logistic Regression | 0.2095 | 0.6224 | 0.2214 |

*Table 2: Calibration metrics. Lower is better.*

**Key observations:**
1. **GP-family models now have the best calibration**: DGP-2 has best Brier/log-loss and SVGP has best ECE.
2. **Natural sampling improved GP calibration behavior** versus prior oversampling-based runs, while retaining strong discrimination.
3. **XGBoost and Logistic Regression are poorly calibrated** in this run and would need post-hoc calibration for PD-use cases.

---

## 5.3 Threshold Behaviour (0.5)

At threshold 0.5, DGP-2 and SVGP are more conservative positive callers than XGBoost:
- DGP-2: precision **0.6557**, recall **0.3008**
- SVGP: precision **0.6562**, recall **0.3158**
- XGBoost: precision **0.4294**, recall **0.5489**

So GP/DGP models are higher-precision, lower-recall at the default threshold; threshold tuning should be done against business cost.

---

## 5.4 Computational Cost (Quick Mode)

| Model | Train time (s) |
|---|---:|
| Logistic Regression | 0.0099 |
| XGBoost | 0.1151 |
| Random Forest | 0.4587 |
| SVGP | 28.4764 |
| DGP-2 | 94.4010 |

As expected, GP-family models are significantly slower to train than classical baselines, but they deliver stronger calibration and top ranking performance in this quick-mode run.

---

## 5.5 Summary

For this updated configuration, **DGP-2 and SVGP are the strongest probabilistic models overall**: they lead ranking quality and calibration quality, while XGBoost remains strong at a fixed 0.5 threshold. For credit-risk PD use, the GP-family outputs are currently the most reliable without extra calibration layers.