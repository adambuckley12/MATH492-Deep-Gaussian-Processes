"""
shared/base.py
==============
Abstract base class for all models, and the shared pipeline configuration
dataclass used by both the classification and regression pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Abstract model interface

class BaseModel(ABC):
    """
    Common interface implemented by every model in this project.

    Subclasses must implement ``fit`` and ``predict``.
    ``save``, ``load``, and ``feature_importance`` are optional — the default
    implementations raise ``NotImplementedError`` so callers know they are
    absent rather than silently doing nothing.
    """

    #: Human-readable model name used in printed summaries and exported tables.
    name: str = "Model"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "BaseModel":
        """Train the model on ``X`` / ``y`` and return ``self``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return point predictions for ``X``."""

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{self.name} does not support save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{self.name} does not support load()")

    def feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        raise NotImplementedError(
            f"{self.name} does not provide feature_importance()"
        )


# Shared pipeline configuration

@dataclass
class PipelineConfig:
    """
    Hyperparameters and output paths shared by the classification and
    regression pipelines.

    Both pipelines accept this single config.  The regression pipeline
    ignores ``xgb_estimators`` (it uses its own XGBoost wrapper that
    handles early stopping internally).
    """

    seed: int = 42

    # Output directories — set by the dataset entry-point before calling the pipeline.
    tables_dir: str = ""
    predictions_dir: str = ""
    weights_dir: str = ""
    figures_dir: str = ""

    # SVGP
    svgp_inducing: int = 512
    svgp_epochs: int = 150
    svgp_batch_size: int = 512

    # DSVGP
    dgp_hidden_dim: int = 5
    dgp_inducing: int = 128
    dgp_epochs: int = 150
    dgp_batch_size: int = 512
    dgp_num_samples: int = 30

    # Classical
    rf_estimators: int = 500
    xgb_estimators: int = 1000  # used by the classification pipeline only
