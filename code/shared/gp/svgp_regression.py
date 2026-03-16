"""
gp/svgp_regression.py
=====================
Sparse Variational Gaussian Process (SVGP) for regression.

Mirrors svgp.py but uses a GaussianLikelihood instead of BernoulliLikelihood,
producing predictive means and variances suitable for RMSE / NLPD evaluation.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO


def _iter_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    n = X.shape[0]
    idx = torch.randperm(n)
    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield X[b], y[b]


class _SVGPRegressionModel(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        n_features = inducing_points.shape[-1]
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        var_strat = VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True,
        )
        super().__init__(var_strat)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
        )
        with torch.no_grad():
            self.covar_module.base_kernel.lengthscale = float(n_features) ** 0.5

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x),
        )


class SVGPRegressionModel:
    """Sparse Variational GP for regression."""

    name = "SVGP"

    def __init__(
        self,
        num_inducing: int = 512,
        n_epochs: int = 100,
        batch_size: int = 512,
        lr: float = 0.01,
        device: str | None = None,
        print_every: int = 1,
    ):
        self.num_inducing = num_inducing
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.print_every = print_every

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._model: _SVGPRegressionModel | None = None
        self._likelihood: GaussianLikelihood | None = None

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "SVGPRegressionModel":
        X = torch.tensor(np.asarray(X_train, dtype=np.float32))
        y = torch.tensor(np.asarray(y_train, dtype=np.float32))

        inducing_points = self._init_inducing(X)

        self._model = _SVGPRegressionModel(inducing_points).to(self.device)
        self._likelihood = GaussianLikelihood().to(self.device)

        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self.lr,
        )
        mll = VariationalELBO(self._likelihood, self._model, num_data=len(y))

        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in _iter_batches(X, y, self.batch_size):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                output = self._model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)

            if self.print_every and epoch % self.print_every == 0:
                avg = epoch_loss / len(y)
                print(f"  [{self.name}] Epoch {epoch:4d}/{self.n_epochs}  loss={avg:.4f}")

        return self

    @torch.no_grad()
    def predict(self, X: pd.DataFrame | np.ndarray,
                chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) predictions."""
        self._model.eval()
        self._likelihood.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        all_means, all_vars = [], []
        for start in range(0, len(X_arr), chunk_size):
            X_chunk = torch.tensor(X_arr[start:start + chunk_size]).to(self.device)
            f_dist = self._model(X_chunk)
            y_dist = self._likelihood(f_dist)
            all_means.append(y_dist.mean.cpu().numpy())
            all_vars.append(y_dist.variance.cpu().numpy())
        return np.concatenate(all_means), np.concatenate(all_vars)

    def _init_inducing(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        idx = torch.randperm(n)[: self.num_inducing]
        return X[idx].clone().to(self.device)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n_features = self._model.variational_strategy.inducing_points.shape[-1]
        torch.save({
            "config": {
                "num_inducing": self.num_inducing,
                "n_features": n_features,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "print_every": self.print_every,
            },
            "model_state": self._model.state_dict(),
            "likelihood_state": self._likelihood.state_dict(),
        }, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "SVGPRegressionModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        obj = cls(
            num_inducing=cfg["num_inducing"],
            n_epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            print_every=cfg["print_every"],
            device=device,
        )
        dummy = torch.zeros(cfg["num_inducing"], cfg["n_features"])
        obj._model = _SVGPRegressionModel(dummy).to(obj.device)
        obj._likelihood = GaussianLikelihood().to(obj.device)
        obj._model.load_state_dict(checkpoint["model_state"])
        obj._likelihood.load_state_dict(checkpoint["likelihood_state"])
        print(f"  [{obj.name}] weights loaded ← {path}")
        return obj
