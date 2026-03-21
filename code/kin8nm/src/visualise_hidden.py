"""
visualise_hidden.py
===================
Visualise the input-space warping learned by DSVGP-2L on Kin8nm.

Produces a three-panel figure:
  Left   — test-set points in the original input space (PCA to 2D)
  Centre — same points in the learned hidden representation (5D → PCA 2D)
  Right  — the posterior predictive mean vs actual y

Saves to results/large/figures/hidden_representation.png
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import gpytorch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "..", "shared"))
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

from data.data_loader import load_kin8nm
from dgp.dsvgp_regression import DSVGPRegressionModel

SEED = 42
WEIGHTS = os.path.join(_HERE, "results", "large", "weights", "dsvgp_2l.pt")
OUT_DIR = os.path.join(_HERE, "results", "large", "figures")
OUT_PATH = os.path.join(OUT_DIR, "hidden_representation.png")

# ── load and split data exactly as the training pipeline does ──────────────
df = load_kin8nm(os.path.join(_HERE, "data"))
X_raw = df.drop(columns=["y"]).values.astype(np.float32)
y_raw = df["y"].values.astype(np.float32)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=SEED
)
X_train, _, y_train, _ = train_test_split(
    X_dev, y_dev, test_size=0.15, random_state=SEED
)

scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# ── load trained model ─────────────────────────────────────────────────────
print("Loading DSVGP-2L weights …")
model_wrapper = DSVGPRegressionModel.load(WEIGHTS)
dgp = model_wrapper._model
dgp.eval()

X_t = torch.tensor(X_test_scaled)

# ── extract intermediate (hidden) representation ───────────────────────────
# Pass through hidden_layers[0] variational strategy to get posterior mean
print("Extracting hidden representation …")
with torch.no_grad():
    layer0 = dgp.hidden_layers[0]
    # variational_strategy(x, prior=False) → posterior distribution
    # mean shape: [hidden_dim, N] = [5, N]
    posterior = layer0.variational_strategy(X_t, prior=False)
    hidden_mean = posterior.mean.T.cpu().numpy()  # [N, 5]

# ── predictive mean for scatter ────────────────────────────────────────────
print("Computing predictions …")
y_pred, _ = model_wrapper.predict(X_test_scaled)

# ── PCA projections ────────────────────────────────────────────────────────
pca_in = PCA(n_components=2, random_state=SEED).fit(X_test_scaled)
X_2d = pca_in.transform(X_test_scaled)
var_in = pca_in.explained_variance_ratio_

pca_h = PCA(n_components=2, random_state=SEED).fit(hidden_mean)
H_2d = pca_h.transform(hidden_mean)
var_h = pca_h.explained_variance_ratio_

# ── figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4))
gs = gridspec.GridSpec(1, 3, wspace=0.35)

cmap = "viridis"
s = 6
alpha = 0.55

# Panel 1 — input space
ax0 = fig.add_subplot(gs[0])
sc0 = ax0.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test, cmap=cmap, s=s, alpha=alpha)
ax0.set_title("Input space (PCA)", fontsize=11)
ax0.set_xlabel(f"PC1 ({var_in[0]*100:.1f}%)", fontsize=9)
ax0.set_ylabel(f"PC2 ({var_in[1]*100:.1f}%)", fontsize=9)
plt.colorbar(sc0, ax=ax0, label="y (target)")

# Panel 2 — hidden representation
ax1 = fig.add_subplot(gs[1])
sc1 = ax1.scatter(H_2d[:, 0], H_2d[:, 1], c=y_test, cmap=cmap, s=s, alpha=alpha)
ax1.set_title("Hidden representation (PCA)", fontsize=11)
ax1.set_xlabel(f"PC1 ({var_h[0]*100:.1f}%)", fontsize=9)
ax1.set_ylabel(f"PC2 ({var_h[1]*100:.1f}%)", fontsize=9)
plt.colorbar(sc1, ax=ax1, label="y (target)")

# Panel 3 — predicted vs actual
ax2 = fig.add_subplot(gs[2])
lim = (min(y_test.min(), y_pred.min()) - 0.05,
       max(y_test.max(), y_pred.max()) + 0.05)
ax2.scatter(y_test, y_pred, c=y_test, cmap=cmap, s=s, alpha=alpha)
ax2.plot(lim, lim, "k--", linewidth=0.9, label="y = ŷ")
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_title("Predicted vs actual", fontsize=11)
ax2.set_xlabel("Actual y", fontsize=9)
ax2.set_ylabel("Predicted ŷ", fontsize=9)
ax2.legend(fontsize=8)

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.close(fig)
