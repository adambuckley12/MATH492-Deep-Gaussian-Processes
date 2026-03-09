"""Post-hoc calibration utilities for probability predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)


@dataclass
class FittedCalibrator:
    method: str
    model: object

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        probs = _clip_probs(y_prob)
        if self.method == "platt":
            logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
            return self.model.predict_proba(logits)[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model.predict(probs), 0.0, 1.0)
        raise ValueError(f"Unsupported calibration method: {self.method}")


class _IdentityCalibrator:
    """No-op calibrator used when calibration data has only one class."""

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-X.ravel()))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.asarray(X)


def fit_probability_calibrator(
    method: str,
    y_prob: np.ndarray,
    y_true: np.ndarray,
) -> FittedCalibrator:
    """Fit a post-hoc calibrator from predicted probabilities."""
    probs = _clip_probs(y_prob)
    targets = np.asarray(y_true, dtype=int)

    # Skip calibration if only one class present
    if len(np.unique(targets)) < 2:
        return FittedCalibrator(method=method, model=_IdentityCalibrator())

    if method == "platt":
        logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
        model = LogisticRegression(solver="lbfgs", random_state=42)
        model.fit(logits, targets)
        return FittedCalibrator(method=method, model=model)

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probs, targets)
        return FittedCalibrator(method=method, model=model)

    raise ValueError(f"Unsupported calibration method: {method}")