"""
gp/svgp.py
==========
Sparse Variational Gaussian Process (SVGP) for binary credit-default prediction.

Model specification
-------------------
* Likelihood   : Bernoulli (logit link) — natural choice for binary labels
* Kernel       : ScaleKernel( RBFKernel(ARD) ) — separate lengthscale per
                 input dimension; lets the model learn which features matter
* Variational  : CholeskyVariationalDistribution + VariationalStrategy with
                 ``learn_inducing_locations=True``
* Inducing pts : Initialised by k-means on a subsample of X_train, which
                 gives far better starting points than random
* ELBO         : VariationalELBO scaled by dataset size (``num_data``)

Training
--------
* Adam optimiser (separate learning rates for variational params vs kernel
  hyper-parameters is a common trick — kept uniform here for simplicity)
* Mini-batch SGD with configurable batch size (default 512)
* Convergence tracked via ELBO loss printed every ``print_every`` epochs

Inference
---------
* ``predict_proba``: samples the posterior ``num_samples`` times then averages
  the Bernoulli probabilities → Monte-Carlo estimate of p(y=1 | x, data)
* Hard predictions threshold at 0.5 by default

Usage
-----
    from gp.svgp import SVGPModel

    model = SVGPModel(num_inducing=512, n_epochs=100)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
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
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO

from classical.metrics import evaluate_classifier


def _iter_batches(
    X: torch.Tensor, y: torch.Tensor, batch_size: int
):
    """Yield (X_batch, y_batch) without torch DataLoader (avoids macOS deadlock)."""
    n   = X.shape[0]
    idx = torch.randperm(n)
    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield X[b], y[b]


# -- Internal GPyTorch model ---------------------------------------------------

class _SVGPInternalModel(ApproximateGP):
    """GPyTorch model (not used directly — wrapped by SVGPModel)."""

    def __init__(self, inducing_points: torch.Tensor):
        n_features = inducing_points.shape[-1]
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        var_strat = VariationalStrategy(
            self,
            inducing_points,
            var_dist,
            learn_inducing_locations=True,
        )
        super().__init__(var_strat)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
        )

        # Initialise lengthscale to sqrt(n_features) — prevents near-zero RBF
        # kernel values for standardised high-dimensional inputs where
        # ||Δx||² ~ n_features, making exp(-||Δx||²/2) approx = 0 with default l=1.
        with torch.no_grad():
            self.covar_module.base_kernel.lengthscale = float(n_features) ** 0.5

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


# -- Public wrapper ------------------------------------------------------------

class SVGPModel:
    """
    Sparse Variational GP for binary classification.

    Parameters
    ----------
    num_inducing : int, default 512
        Number of inducing points. More → better approximation, slower training.
        For 24 k samples with 25 features, 512 is a good balance.
    n_epochs     : int, default 100
    batch_size   : int, default 512
    lr           : float, default 0.01
    num_samples  : int, default 20
        Monte-Carlo samples used during prediction for the Bernoulli integral.
    device       : str | None
        ``'cuda'`` / ``'mps'`` / ``'cpu'``.  Auto-detected if None.
    print_every  : int, default 10
        Print training loss every N epochs (0 = silent).
    """

    name = "SVGP"

    def __init__(
        self,
        num_inducing: int = 512,
        n_epochs: int = 100,
        batch_size: int = 512,
        lr: float = 0.01,
        num_samples: int = 20,
        device: str | None = None,
        print_every: int = 1,
    ):
        self.num_inducing = num_inducing
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.num_samples  = num_samples
        self.print_every  = print_every

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # MPS (Apple Silicon) lacks float64 support required by
                # GPyTorch's Cholesky, so always fall back to CPU. MPS is float32 only.
                device = "cpu"
        self.device = torch.device(device)

        self._model: _SVGPInternalModel | None = None
        self._likelihood: BernoulliLikelihood | None = None

    # -- Fitting ---------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "SVGPModel":
        X = torch.tensor(np.asarray(X_train, dtype=np.float32))
        y = torch.tensor(np.asarray(y_train, dtype=np.float32))

        # Initialise inducing points via k-means subsample
        inducing_points = self._init_inducing(X)

        self._model      = _SVGPInternalModel(inducing_points).to(self.device)
        self._likelihood = BernoulliLikelihood().to(self.device)

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
                loss   = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)

            if self.print_every and epoch % self.print_every == 0:
                avg = epoch_loss / len(y)
                print(f"  [{self.name}] Epoch {epoch:4d}/{self.n_epochs}  "
                      f"loss={avg:.4f}")

        return self

    # -- Prediction ------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame | np.ndarray,
                      chunk_size: int = 512) -> np.ndarray:
        """Return P(y=1 | x) via the trained BernoulliLikelihood.

        The likelihood applies the same probit approximation used internally by
        VariationalELBO's expected_log_prob, so training and prediction are
        consistent.  This produces better-calibrated probabilities than the
        naive sigmoid(f_samples).mean() approach (which uses a different
        logistic link and relies on noisy MC sampling).

        Processes *X* in chunks to avoid building a single massive kernel
        matrix that can deadlock on macOS (OpenBLAS + OMP).
        """
        self._model.eval()
        self._likelihood.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        all_probs = []
        for start in range(0, len(X_arr), chunk_size):
            X_chunk = torch.tensor(X_arr[start:start + chunk_size]).to(self.device)
            # Posterior over latent function values  →  MultivariateNormal (N,)
            f_dist = self._model(X_chunk)
            # Pass through the Bernoulli likelihood: analytically integrates
            # out f uncertainty via probit approximation → Bernoulli (N,)
            probs = self._likelihood(f_dist).probs.cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs)

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        verbose: bool = True,
    ) -> dict:
        y_prob = self.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        return evaluate_classifier(
            self.name, np.asarray(y_test), y_pred, y_prob, verbose=verbose
        )

    # -- Helpers ---------------------------------------------------------------

    def _init_inducing(self, X: torch.Tensor) -> torch.Tensor:
        """
        Select inducing point locations via random subsample of training data.
        K-means would be ideal but is slow at this scale; random subsample is
        a well-accepted alternative for SVGP initialisation.
        """
        n = X.shape[0]
        idx = torch.randperm(n)[: self.num_inducing]
        return X[idx].clone().to(self.device)

    # -- Persistence ------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save model weights and architecture config to *path* (.pt).

        The checkpoint includes everything needed to reconstruct the model
        without retraining:
        ``config`` — hyperparameters needed to re-build the architecture
        ``model_state`` / ``likelihood_state`` — trained parameter tensors
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n_features = (
            self._model.variational_strategy.inducing_points.shape[-1]
        )
        torch.save(
            {
                "config": {
                    "num_inducing": self.num_inducing,
                    "n_features":  n_features,
                    "n_epochs":    self.n_epochs,
                    "batch_size":  self.batch_size,
                    "lr":          self.lr,
                    "num_samples": self.num_samples,
                    "print_every": self.print_every,
                },
                "model_state":      self._model.state_dict(),
                "likelihood_state": self._likelihood.state_dict(),
            },
            path,
        )
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "SVGPModel":
        """
        Reconstruct a trained SVGPModel from a checkpoint saved with
        :meth:`save`.  The architecture is rebuilt from the stored config
        before the state dict is loaded, so no training data is needed.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]

        obj = cls(
            num_inducing = cfg["num_inducing"],
            n_epochs     = cfg["n_epochs"],
            batch_size   = cfg["batch_size"],
            lr           = cfg["lr"],
            num_samples  = cfg["num_samples"],
            print_every  = cfg["print_every"],
            device       = device,
        )

        # Build architecture shell with the correct inducing-point shape
        dummy = torch.zeros(cfg["num_inducing"], cfg["n_features"])
        obj._model      = _SVGPInternalModel(dummy).to(obj.device)
        obj._likelihood = BernoulliLikelihood().to(obj.device)

        obj._model.load_state_dict(checkpoint["model_state"])
        obj._likelihood.load_state_dict(checkpoint["likelihood_state"])
        print(f"  [{obj.name}] weights loaded ← {path}")
        return obj
