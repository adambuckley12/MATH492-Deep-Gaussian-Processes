"""
dgp/dsvgp_regression.py
=======================
Deep Sparse Variational GP (DSVGP) for regression.

Mirrors dsvgp.py but uses GaussianLikelihood instead of BernoulliLikelihood.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel


def _iter_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    n = X.shape[0]
    idx = torch.randperm(n)
    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield X[b], y[b]


class _DeepGPLayer(DeepGPLayer):
    def __init__(
        self,
        input_dims: int,
        output_dims: int | None,
        num_inducing: int = 128,
        inducing_pts: torch.Tensor | None = None,
    ):
        if output_dims is None:
            if inducing_pts is None:
                inducing_pts = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            if inducing_pts is None:
                inducing_pts = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape,
        )
        var_strat = VariationalStrategy(
            self, inducing_pts, var_dist, learn_inducing_locations=True,
        )
        super().__init__(var_strat, input_dims, output_dims)

        MeanClass = ZeroMean if output_dims is None else ConstantMean
        self.mean_module = MeanClass(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )
        with torch.no_grad():
            self.covar_module.base_kernel.lengthscale = float(input_dims) ** 0.5

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _DSVGPRegressionModel(DeepGP):
    def __init__(
        self,
        input_dims: int,
        n_layers: int,
        hidden_dim: int,
        num_inducing: int,
        X_init: torch.Tensor | None = None,
    ):
        super().__init__()
        assert n_layers >= 2

        if X_init is not None:
            idx = torch.randperm(X_init.shape[0])[:num_inducing]
            pts = X_init[idx].clone()
            first_inducing = pts.unsqueeze(0).expand(hidden_dim, -1, -1).clone()
        else:
            first_inducing = None

        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(
            _DeepGPLayer(input_dims, hidden_dim, num_inducing, inducing_pts=first_inducing)
        )
        for _ in range(n_layers - 2):
            self.hidden_layers.append(
                _DeepGPLayer(hidden_dim, hidden_dim, num_inducing)
            )
        self.output_layer = _DeepGPLayer(hidden_dim, None, num_inducing)
        self.likelihood = GaussianLikelihood()

    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        return self.output_layer(h)

    def predict(self, x: torch.Tensor):
        with gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self(x))
        return preds


class DSVGPRegressionModel:
    """Deep Sparse Variational GP for regression."""

    def __init__(
        self,
        n_layers: int = 2,
        hidden_dim: int = 5,
        num_inducing: int = 128,
        n_epochs: int = 100,
        batch_size: int = 512,
        lr: float = 0.01,
        num_samples: int = 50,
        device: str | None = None,
        print_every: int = 10,
    ):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_inducing = num_inducing
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        self.print_every = print_every

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._model: _DSVGPRegressionModel | None = None

    @property
    def name(self) -> str:
        return f"DSVGP-{self.n_layers}L"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "DSVGPRegressionModel":
        X = torch.tensor(np.asarray(X_train, dtype=np.float32))
        y = torch.tensor(np.asarray(y_train, dtype=np.float32))

        input_dims = X.shape[1]
        self._model = _DSVGPRegressionModel(
            input_dims=input_dims,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            num_inducing=self.num_inducing,
            X_init=X,
        ).to(self.device)

        self._model.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mll = DeepApproximateMLL(
            VariationalELBO(self._model.likelihood, self._model, num_data=len(y))
        )

        train_samples = min(self.num_samples, 10)

        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in _iter_batches(X, y, self.batch_size):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                with gpytorch.settings.num_likelihood_samples(train_samples):
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
                chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) predictions."""
        self._model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        all_means, all_vars = [], []
        for start in range(0, len(X_arr), chunk_size):
            X_chunk = torch.tensor(X_arr[start:start + chunk_size]).to(self.device)
            with gpytorch.settings.num_likelihood_samples(self.num_samples):
                preds = self._model.likelihood(self._model(X_chunk))
                mean = preds.mean.mean(dim=0).cpu().numpy()
                var = preds.variance.mean(dim=0).cpu().numpy()
            all_means.append(mean)
            all_vars.append(var)
        return np.concatenate(all_means), np.concatenate(all_vars)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        input_dims = (
            self._model.hidden_layers[0]
            .variational_strategy.inducing_points.shape[-1]
        )
        torch.save({
            "config": {
                "n_layers": self.n_layers,
                "hidden_dim": self.hidden_dim,
                "num_inducing": self.num_inducing,
                "input_dims": input_dims,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "num_samples": self.num_samples,
                "print_every": self.print_every,
            },
            "model_state": self._model.state_dict(),
        }, path)
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DSVGPRegressionModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        obj = cls(
            n_layers=cfg["n_layers"],
            hidden_dim=cfg["hidden_dim"],
            num_inducing=cfg["num_inducing"],
            n_epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            num_samples=cfg["num_samples"],
            print_every=cfg["print_every"],
            device=device,
        )
        obj._model = _DSVGPRegressionModel(
            input_dims=cfg["input_dims"],
            n_layers=cfg["n_layers"],
            hidden_dim=cfg["hidden_dim"],
            num_inducing=cfg["num_inducing"],
            X_init=None,
        ).to(obj.device)
        state = checkpoint["model_state"]
        key = "hidden_layers.0.variational_strategy.inducing_points"
        if key in state and state[key].dim() == 2:
            pts = state[key]
            state[key] = pts.unsqueeze(0).expand(cfg["hidden_dim"], -1, -1).clone()
        obj._model.load_state_dict(state)
        print(f"  [{obj.name}] weights loaded ← {path}")
        return obj
