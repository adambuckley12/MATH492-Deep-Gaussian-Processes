"""
dgp/dsvgp.py
============
Deep Sparse Variational Gaussian Process (DSVGP) for binary classification.

Architecture
------------
Each layer is a sparse GP (with inducing points) whose output feeds the next.
The final layer produces a scalar GP → Bernoulli likelihood.

  Input  --► GP Layer 1 (hidden_dim outputs)
             --► GP Layer 2 (hidden_dim outputs)   [3-layer only]
                 --► GP Layer N (scalar output)
                     --► BernoulliLikelihood --► P(default)

Intermediate layers have ``output_dims=hidden_dim`` (a batch of
``hidden_dim`` independent GPs), so the latent representation has
``hidden_dim`` dimensions. The final layer is a single GP (``output_dims=None``).

Design choices
--------------
* ARD RBF kernel per layer — each layer can learn its own length-scales
* ConstantMean in hidden layers; ZeroMean in output layer
* 128 inducing points per layer (balanced speed vs quality for 24 k samples)
* ``hidden_dim=5`` default: bottleneck forces the network to extract a compact
  representation; 5 is a standard choice for DGP on tabular data
* Training: Adam + mini-batch VariationalELBO via ``DeepApproximateMLL``
* Prediction: mean over ``num_samples`` Monte-Carlo forward passes

Usage
-----
    from dgp.dsvgp import DSVGPModel

    # 2-layer DGP
    model2 = DSVGPModel(n_layers=2, hidden_dim=5, num_inducing=128)
    model2.fit(X_train, y_train)

    # 3-layer DGP
    model3 = DSVGPModel(n_layers=3, hidden_dim=5, num_inducing=128)
    model3.fit(X_train, y_train)
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
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel

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


# --
# Single Deep GP layer
# --

class _DeepGPLayer(DeepGPLayer):
    """
    One layer of a Deep GP.

    Parameters
    ----------
    input_dims   : int   — dimensionality of this layer's *input*
    output_dims  : int | None
        None → scalar output (use for the final layer)
        int  → batch of ``output_dims`` independent GPs (hidden layers)
    num_inducing : int   — inducing points per GP in this layer
    inducing_pts : Tensor | None
        Pre-initialised inducing-point locations.  If None, falls back to
        ``torch.randn``.  Should be supplied for the first layer (real data
        subsample) to avoid the inducing points starting in the wrong region.
    """

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
                # Hidden layers — no real-space data to draw from, so use
                # random normal but at unit scale (hidden representations will
                # be re-scaled by the kernel anyway)
                inducing_pts = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
        )
        var_strat = VariationalStrategy(
            self,
            inducing_pts,
            var_dist,
            learn_inducing_locations=True,
        )
        super().__init__(var_strat, input_dims, output_dims)

        # Hidden layers use ConstantMean so the bias is learnable;
        # output layer uses ZeroMean (bias learned through variational params)
        MeanClass = ZeroMean if output_dims is None else ConstantMean
        self.mean_module  = MeanClass(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )

        # Initialise lengthscale to sqrt(input_dims).
        # Default l=1 gives RBF(x,x') approx 0 for standardised data in >~5 dims
        # because ||deltax||² ~ input_dims.  sqrt(input_dims) keeps the kernel
        # values in a healthy range from the very first gradient step.
        with torch.no_grad():
            self.covar_module.base_kernel.lengthscale = float(input_dims) ** 0.5

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


# --
# Internal Deep GP model (GPyTorch DeepGP subclass)
# --

class _DSVGPInternalModel(DeepGP):
    """
    Stacks ``n_layers`` GP layers (n_layers-1 hidden + 1 output).

    Architecture
    ~~~~~~~~~~~~
    n_layers=2:  input → hidden_layer  (hidden_dim GPs) → output_layer (1 GP)
    n_layers=3:  input → hidden_layer1 → hidden_layer2  → output_layer
    """

    def __init__(
        self,
        input_dims: int,
        n_layers: int,
        hidden_dim: int,
        num_inducing: int,
        X_init: torch.Tensor | None = None,
    ):
        super().__init__()
        assert n_layers >= 2, "Need at least 2 layers for a Deep GP."

        # -- First-layer inducing points from real training data ----------------
        # Sampling from actual data puts inducing points where the likelihood
        # signal is, preventing the first-layer kernel matrix from collapsing
        # to near-zero (which happens with torch.randn when data is standardised
        # and high-dimensional).
        if X_init is not None:
            idx = torch.randperm(X_init.shape[0])[:num_inducing]
            pts = X_init[idx].clone()  # (num_inducing, input_dims)
            # The first hidden layer has output_dims=hidden_dim, so its inducing
            # points need shape (hidden_dim, num_inducing, input_dims).
            # Expand by repeating across the GP-batch axis.
            first_inducing = pts.unsqueeze(0).expand(hidden_dim, -1, -1).clone()
        else:
            first_inducing = None  # falls back to randn inside _DeepGPLayer

        self.hidden_layers = torch.nn.ModuleList()

        # First hidden layer: input_dims → hidden_dim (data-initialised)
        self.hidden_layers.append(
            _DeepGPLayer(input_dims, hidden_dim, num_inducing,
                         inducing_pts=first_inducing)
        )
        # Additional hidden layers: hidden_dim → hidden_dim (random init fine)
        for _ in range(n_layers - 2):
            self.hidden_layers.append(
                _DeepGPLayer(hidden_dim, hidden_dim, num_inducing)
            )
        # Final output layer: hidden_dim → scalar
        self.output_layer = _DeepGPLayer(hidden_dim, None, num_inducing)

        self.likelihood = BernoulliLikelihood()

    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        return self.output_layer(h)

    def predict(self, x: torch.Tensor):
        with gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self(x))
        return preds


# --
# Public wrapper
# --

class DSVGPModel:
    """
    Deep Sparse Variational GP for binary classification.

    Parameters
    ----------
    n_layers     : int, default 2
        Total number of GP layers (must be ≥ 2).
        2 = one hidden layer + output layer
        3 = two hidden layers + output layer
    hidden_dim   : int, default 5
        Width of each hidden GP layer (number of independent GP outputs).
    num_inducing : int, default 128
        Inducing points per layer. More → better but slower.
    n_epochs     : int, default 100
    batch_size   : int, default 512
    lr           : float, default 0.01
    num_samples  : int, default 10
        MC samples per forward pass. Higher = better but slower.
    device       : str | None
    print_every  : int, default 10
    """

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
        self.n_layers     = n_layers
        self.hidden_dim   = hidden_dim
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
                # GPyTorch's Cholesky, so always fall back to CPU.
                device = "cpu"
        self.device = torch.device(device)

        self._model: _DSVGPInternalModel | None = None

    @property
    def name(self) -> str:
        return f"DSVGP-{self.n_layers}L"

    # -- Fitting

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "DSVGPModel":
        X = torch.tensor(np.asarray(X_train, dtype=np.float32))
        y = torch.tensor(np.asarray(y_train, dtype=np.float32))

        input_dims = X.shape[1]
        self._model = _DSVGPInternalModel(
            input_dims   = input_dims,
            n_layers     = self.n_layers,
            hidden_dim   = self.hidden_dim,
            num_inducing = self.num_inducing,
            X_init       = X,          # ← seed first-layer inducing pts from data
        ).to(self.device)

        self._model.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mll = DeepApproximateMLL(
            VariationalELBO(
                self._model.likelihood,
                self._model,
                num_data=len(y),
            )
        )

        # During training, fewer MC samples suffice (10 is the standard choice
        # in the DGP literature).  Using self.num_samples (50) here inflates
        # gradient variance and slows training without a quality benefit.
        train_samples = min(self.num_samples, 10)

        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in _iter_batches(X, y, self.batch_size):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with gpytorch.settings.num_likelihood_samples(train_samples):
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

    # Prediction

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame | np.ndarray,
                      chunk_size: int = 256) -> np.ndarray:
        """Return P(y=1 | x) by propagating MC paths through all GP layers.

        Uses the Bernoulli likelihood to convert latent function samples to
        probabilities, then averages over the ``num_samples`` MC paths.

        Processes *X* in chunks of ``chunk_size`` rows to avoid building a
        single massive kernel matrix that can deadlock on macOS (OpenBLAS + OMP).
        """
        self._model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        all_probs = []
        for start in range(0, len(X_arr), chunk_size):
            X_chunk = torch.tensor(X_arr[start:start + chunk_size]).to(self.device)
            # num_likelihood_samples propagates self.num_samples independent MC
            # paths through every hidden layer, giving a richer estimate of the
            # predictive integral P(y=1|x) = ∫ σ(f) p(f|data) df.
            with gpytorch.settings.num_likelihood_samples(self.num_samples):
                # Pass through all GP layers, then through the Bernoulli
                # likelihood.  preds.probs has shape (num_samples, N).
                preds = self._model.likelihood(self._model(X_chunk))
                probs = preds.probs.mean(dim=0).cpu().numpy()
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

    # Persistence

    def save(self, path: str) -> None:
        """
        Save model weights and architecture config to *path* (.pt).

        The checkpoint stores enough to reconstruct ``_DSVGPInternalModel``
        and load the trained parameters without retraining.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        input_dims = (
            self._model.hidden_layers[0]
            .variational_strategy.inducing_points.shape[-1]
        )
        torch.save(
            {
                "config": {
                    "n_layers":     self.n_layers,
                    "hidden_dim":   self.hidden_dim,
                    "num_inducing": self.num_inducing,
                    "input_dims":   input_dims,
                    "n_epochs":     self.n_epochs,
                    "batch_size":   self.batch_size,
                    "lr":           self.lr,
                    "num_samples":  self.num_samples,
                    "print_every":  self.print_every,
                },
                "model_state": self._model.state_dict(),
            },
            path,
        )
        print(f"  [{self.name}] weights saved → {path}")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DSVGPModel":
        """
        Reconstruct a trained DSVGPModel from a checkpoint saved with
        :meth:`save`.  The internal ``_DSVGPInternalModel`` is rebuilt from
        stored config and the state dict is loaded on top, so no training
        data or re-initialisation is needed.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]

        obj = cls(
            n_layers     = cfg["n_layers"],
            hidden_dim   = cfg["hidden_dim"],
            num_inducing = cfg["num_inducing"],
            n_epochs     = cfg["n_epochs"],
            batch_size   = cfg["batch_size"],
            lr           = cfg["lr"],
            num_samples  = cfg["num_samples"],
            print_every  = cfg["print_every"],
            device       = device,
        )

        # Reconstruct architecture shell; X_init=None uses randn placeholders
        # which will be immediately overwritten by load_state_dict.
        obj._model = _DSVGPInternalModel(
            input_dims   = cfg["input_dims"],
            n_layers     = cfg["n_layers"],
            hidden_dim   = cfg["hidden_dim"],
            num_inducing = cfg["num_inducing"],
            X_init       = None,
        ).to(obj.device)

        # -- Migrate checkpoints saved before the inducing-point shape fix ------
        # Old checkpoints stored hidden layer 0 inducing points with shape
        # [num_inducing, input_dims] instead of [hidden_dim, num_inducing, input_dims].
        # Detect and expand so load_state_dict succeeds on old weight files.
        state = checkpoint["model_state"]
        key = "hidden_layers.0.variational_strategy.inducing_points"
        if key in state and state[key].dim() == 2:
            pts = state[key]  # [num_inducing, input_dims]
            state[key] = pts.unsqueeze(0).expand(cfg["hidden_dim"], -1, -1).clone()

        obj._model.load_state_dict(state)
        print(f"  [{obj.name}] weights loaded ← {path}")
        return obj
