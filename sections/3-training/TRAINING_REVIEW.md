# Line-by-Line Review of Training Section

## Overview
This document reviews the mathematical correctness, logical flow, and accuracy of the training section (excluding Hilbert space methods and comparison).

---

## 0-training.qmd - Main Introduction

### Line 1: "DGPs offer richer modelling capacity than standard GPs, but at the cost of tractable inference"
- **Status**: ✅ CORRECT
- **Notes**: Accurate summary. DGPs compose functions which increases expressiveness but prevents closed-form marginalisation.

### Line 5: "focusing on how the marginal likelihood arises by integrating out function values, to identify the structural property that is lost in DGPs"
- **Status**: ✅ CORRECT
- **Notes**: Good framing. This promise is now fulfilled by the new intro in 3-dgp-spars-inf.qmd

### Line 5: "reduce computation from $O(n^3)$ to $O(nm^2)$"
- **Status**: ✅ CORRECT
- **Notes**: Accurate complexity reduction via inducing points.

### Line 5: "doubly stochastic variational inference [@salimbeni_doubly_2017]"
- **Status**: ✅ CORRECT
- **Notes**: Correct citation and method.

---

## 01-gp-training.qmd - Learning in Standard GPs

### Line 3: "the GP hyperparameters $\vtheta$ are learned by maximising the LML"
- **Status**: ✅ CORRECT
- **Notes**: Standard approach, Type II ML estimation.

### Line 5: "Gaussian–Gaussian conjugacy [@bernardo_bayesian_2000] allows the integral over function values to be computed in closed form"
- **Status**: ✅ CORRECT
- **Notes**: Key structural property. Gaussian prior × Gaussian likelihood = Gaussian marginal.

### Line 7: Equation for marginal likelihood
```
p(\vy \mid \vX, \vtheta) = \int p(\vy \mid \vf) p(\vf \mid \vX, \vtheta) \, d\vf = \mathcal{N}(\vy \mid \vzero, K_{\vtheta} + \sigma_n^2 \vI)
```
- **Status**: ✅ CORRECT
- **Notes**: Standard GP marginal likelihood. Assumes Gaussian likelihood $p(\vy \mid \vf) = \mathcal{N}(\vf, \sigma_n^2 \vI)$.

### Line 10: "As the product of two Gaussians is proportional to a Gaussian, the $\vf$-integral collapses to a single multivariate normal"
- **Status**: ✅ CORRECT
- **Notes**: Accurate description of Gaussian conjugacy.

### Line 15: Gradient equation
```
\frac{\partial \log p(\vy)}{\partial \theta_j} = \frac{1}{2}\tr\left((\valpha \valpha^\top - K_y^{-1}) \frac{\partial K_y}{\partial \theta_j}\right)
```
- **Status**: ✅ CORRECT
- **Notes**: Standard gradient from Rasmussen & Williams. Correctly cited.

### Line 18: Interpretation of $\valpha \valpha^\top - K_y^{-1}$
- **Status**: ✅ CORRECT
- **Notes**: Good intuitive explanation. $\valpha \valpha^\top$ is data fit, $K_y^{-1}$ is model complexity.

### Line 18: Matrix calculus identities mentioned
- **Status**: ✅ CORRECT
- **Notes**: $\frac{\partial}{\partial \theta} \log|K| = \tr(K^{-1} \frac{\partial K}{\partial \theta})$ and $\frac{\partial K^{-1}}{\partial \theta} = -K^{-1} \frac{\partial K}{\partial \theta} K^{-1}$ are standard results.

### Line 20: "$O(n^3)$ cost with $O(n^2)$ storage"
- **Status**: ✅ CORRECT
- **Notes**: Cholesky decomposition is $O(n^3)$, storing $K$ is $O(n^2)$.

---

## 1-var-inf.qmd - Variational Inference

### Line 3: "variational inference allows for approximations"
- **Status**: ✅ CORRECT
- **Notes**: Standard VI framework. Good citations (Jordan, Blei).

### Line 8: ELBO decomposition
```
\log p(\vy) = \mathcal{L}(q) + \KL(q(\vz) \| p(\vz \mid \vy))
```
- **Status**: ✅ CORRECT
- **Notes**: Fundamental VI equation. Log evidence = ELBO + KL divergence.

### Line 14: ELBO definition
```
\mathcal{L}(q) = \mathbb{E}_{q(\vz)}[\log p(\vy, \vz)] - \mathbb{E}_{q(\vz)}[\log q(\vz)] = \mathbb{E}_{q(\vz)}[\log p(\vy \mid \vz)] - \KL(q(\vz) \| p(\vz))
```
- **Status**: ✅ CORRECT
- **Notes**: Both forms are correct and equivalent.

### Line 17: "maximises the ELBO, which is equivalent to minimising the KL divergence to the true posterior"
- **Status**: ✅ CORRECT
- **Notes**: Since $\log p(\vy)$ is constant w.r.t. $q$, maximising $\mathcal{L}(q)$ = minimising $\KL(q \| p(\vz \mid \vy))$.

### Lines 22-26: Derivation of ELBO decomposition
- **Status**: ✅ CORRECT
- **Notes**: Algebraically sound. Uses $p(\vz \mid \vy) = p(\vy, \vz) / p(\vy)$ correctly.

### Line 30: "The expected log-likelihood $\mathbb{E}_{q}[\log p(\vy \mid \vz)]$ pushes $q$ to concentrate where the data is explained well, while the KL term $\KL(q \| p(\vz))$ pushes $q$ toward the prior"
- **Status**: ✅ CORRECT
- **Notes**: Excellent intuitive explanation of ELBO structure.

### Line 32: "For the models in this dissertation, variational inference serves two purposes. Firstly, they allow for scaling to large datasets. Secondly, they allow for handling the hierarchical structure of DGPs"
- **Status**: ✅ CORRECT
- **Notes**: Good framing of why VI is needed.

### Line 35: "variational distribution is usually chosen to be Gaussian"
```
q(\vf) = \mathcal{N}(\vf \mid \vm, S)
```
- **Status**: ✅ CORRECT
- **Notes**: Standard choice for GP VI. Balances expressiveness with tractability.

### Line 43: Gaussian KL closed form
```
\KL(q(\vf) \| p(\vf)) = \frac{1}{2}\left[\tr(K^{-1}S) + \vm^\top K^{-1} \vm - n + \log|K| - \log|S|\right]
```
- **Status**: ✅ CORRECT
- **Notes**: Standard formula for KL between two multivariate Gaussians.

### Line 48: Interpretation of KL terms
- **Status**: ✅ CORRECT
- **Notes**: "$\vm^\top K^{-1} \vm$ penalises mean deviation", "$\tr(K^{-1}S)$ penalises variance inflation", "$\log|K| - \log|S|$ entropy term". All correct.

### Line 50: "optimisation over a full-rank matrix $S \in \mathbb{R}^{n \times n}$ still involves $O(n^2)$ parameters"
- **Status**: ✅ CORRECT
- **Notes**: Motivates sparse methods. Full covariance has $n(n+1)/2$ unique parameters.

---

## 2-sparse-var-inf.qmd - Sparse Variational GPs

### Line 3: "$O(n^3)$ cost becomes prohibitive... using $m \ll n$ inducing points... reduces complexity to $O(nm^2)$"
- **Status**: ✅ CORRECT
- **Notes**: Standard sparse GP framework. Correct complexity and citations.

### Line 8: Joint distribution of $\vf$ and $\vu$
```
\begin{bmatrix} \vf \\ \vu \end{bmatrix} \sim \mathcal{N}\left(\vzero, \begin{bmatrix} K_{ff} & K_{fu} \\ K_{uf} & K_{uu} \end{bmatrix}\right)
```
- **Status**: ✅ CORRECT
- **Notes**: Standard GP joint prior. $K_{fu} = K_{uf}^\top$ by symmetry.

### Line 11: "$K_{fu}$ encodes how closely each training point correlates with each inducing point"
- **Status**: ✅ CORRECT
- **Notes**: Good intuition. If inducing points placed well, $K_{fu}$ captures most information.

### Line 13: "treat $\vZ$ as variational parameters rather than fixed pseudo-inputs"
- **Status**: ✅ CORRECT
- **Notes**: Key Titsias 2009 insight. $\vZ$ optimised jointly with $\vm, S$.

### Line 16: Sparse variational form
```
q(\vf, \vu) = p(\vf \mid \vu) q(\vu)
```
- **Status**: ✅ CORRECT
- **Notes**: Key structured factorisation. Uses prior conditional for $\vf \mid \vu$.

### Line 19: "reduces the variational parameter count from $O(n^2)$ to $O(m^2)$"
- **Status**: ✅ CORRECT
- **Notes**: Major computational saving. Parameters now independent of $n$.

### Line 22: GP conditional
```
p(\vf \mid \vu) = \mathcal{N}(\vf \mid K_{fu} K_{uu}^{-1} \vu, \, K_{ff} - K_{fu} K_{uu}^{-1} K_{uf})
```
- **Status**: ✅ CORRECT
- **Notes**: Standard GP predictive distribution formula.

### Line 25: "The conditional mean $K_{fu}K_{uu}^{-1}\vu$ is a weighted result of inducing values"
- **Status**: ✅ CORRECT (minor typo)
- **Notes**: Should say "weighted sum" not "weighted result", but meaning is clear.

### Line 25: "The conditional variance... vanishes at the inducing locations themselves"
- **Status**: ✅ CORRECT
- **Notes**: At inducing points $\vX = \vZ$, we have $K_{fu} = K_{uu}$, so variance = $K_{uu} - K_{uu}K_{uu}^{-1}K_{uu} = 0$.

### Line 30: Marginal $q(\vf)$ after integrating out $\vu$
```
q(\vf) = \mathcal{N}(\vf \mid K_{fu} K_{uu}^{-1} \vm, \, K_{ff} - K_{fu} K_{uu}^{-1} K_{uf} + K_{fu} K_{uu}^{-1} S K_{uu}^{-1} K_{uf})
```
- **Status**: ✅ CORRECT
- **Notes**: Standard result. Product of Gaussians gives Gaussian.

### Line 33: "the variational posterior $q(\vf)$ adds the term $K_{fu}K_{uu}^{-1}SK_{uu}^{-1}K_{uf}$ to the variance"
- **Status**: ✅ CORRECT
- **Notes**: This term restores uncertainty about $\vu$.

### Line 38: Sparse ELBO
```
\mathcal{L} = \mathbb{E}_{q(\vf)}[\log p(\vy \mid \vf)] - \KL(q(\vu) \| p(\vu))
```
- **Status**: ✅ CORRECT
- **Notes**: Standard sparse GP ELBO (Titsias 2009).

### Line 44: Gaussian KL for inducing points
```
\KL(q(\vu) \| p(\vu)) = \frac{1}{2}\left[\tr(K_{uu}^{-1}S) + \vm^\top K_{uu}^{-1} \vm - m + \log|K_{uu}| - \log|S|\right]
```
- **Status**: ✅ CORRECT
- **Notes**: Same formula as before but for $m$-dimensional inducing points.

### Line 47: "For Gaussian likelihoods... fully analytical ELBO"
- **Status**: ✅ CORRECT
- **Notes**: Gaussian likelihood makes expected log-likelihood tractable.

### Line 50-53: Optimal variational parameters
```
S^{-1} = K_{uu}^{-1} + \sigma_n^{-2} K_{uu}^{-1} K_{uf} K_{fu} K_{uu}^{-1}
\vm = \sigma_n^{-2} S K_{uu}^{-1} K_{uf} \vy
```
- **Status**: ✅ CORRECT
- **Notes**: These are the closed-form optimal parameters for Gaussian likelihood.

### Line 60-62: Sparse predictions
```
\mathbb{E}[f_*] = K_{*u} K_{uu}^{-1} \vm
\text{Var}[f_*] = K_{**} - K_{*u} K_{uu}^{-1} (K_{uu} - S) K_{uu}^{-1} K_{u*}
```
- **Status**: ✅ CORRECT
- **Notes**: Standard sparse GP predictive distribution.

### Line 65: "$O(m^3)$ for inverting $K_{uu}$, $O(nm^2)$ for matrix products"
- **Status**: ✅ CORRECT
- **Notes**: Accurate complexity analysis. Cholesky of $K_{uu}$ is $O(m^3)$, products with $K_{fu}$ are $O(nm^2)$.

### Line 71: Expected log-likelihood decomposes over data points
```
\mathbb{E}_{q(\vf)}[\log p(\vy \mid \vf)] = \sum_{i=1}^n \mathbb{E}_{q(f_i)}[\log p(y_i \mid f_i)]
```
- **Status**: ✅ CORRECT
- **Notes**: Independence assumption in likelihood: $p(\vy \mid \vf) = \prod_i p(y_i \mid f_i)$.

### Line 74: "The KL term does not depend on the data and can be evaluated once per step"
- **Status**: ✅ CORRECT
- **Notes**: $\KL(q(\vu) \| p(\vu))$ only depends on variational parameters, not data.

### Line 77: Minibatch ELBO
```
\hat{\mathcal{L}} = \frac{n}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbb{E}_{q(f_i)}[\log p(y_i \mid f_i)] - \KL(q(\vu) \| p(\vu))
```
- **Status**: ✅ CORRECT
- **Notes**: Standard minibatch estimator. Factor $n/|\mathcal{B}|$ ensures unbiasedness.

### Line 80: "$n/|\mathcal{B}|$ re-scales the minibatch sum so that it turns into an unbiased estimate"
- **Status**: ✅ CORRECT
- **Notes**: Correct explanation. Minibatch sum × scaling factor = unbiased estimate of full sum.

### Line 80: "Each step now costs $O(|\mathcal{B}|m^2 + m^3)$"
- **Status**: ✅ CORRECT
- **Notes**: $|\mathcal{B}|$ points × $m^2$ per point operations + $m^3$ for KL/Cholesky.

---

## 3-dgp-spars-inf.qmd - Variational Inference for DGPs

### Line 3: New intro paragraph explaining intractability
- **Status**: ✅ CORRECT
- **Notes**: Good bridging from tractable GP case. Clearly explains why DGPs break Gaussian conjugacy: covariance matrices $K^{(\ell)}$ depend nonlinearly on random $\vF^{(\ell-1)}$.

### Line 3: "the resulting marginal is no longer Gaussian"
- **Status**: ✅ CORRECT
- **Notes**: Key point. Even though each conditional is Gaussian, the marginal over all layers is not.

### Line 6: Inducing point dimensions: $\vZ^{(\ell)} \in \mathbb{R}^{m_\ell \times d_{\ell-1}}$, $\vU^{(\ell)} \in \mathbb{R}^{m_\ell \times d_\ell}$
- **Status**: ✅ CORRECT
- **Notes**: Inducing inputs have dimensionality of previous layer $d_{\ell-1}$, outputs have current layer dimensionality $d_\ell$.

### Line 9: Variational factorisation
```
q(\{\vF^{(\ell)}, \vU^{(\ell)}\}_{\ell=1}^L) = \prod_{\ell=1}^{L} p(\vF^{(\ell)} \mid \vU^{(\ell)}, \vF^{(\ell-1)}) q(\vU^{(\ell)})
```
- **Status**: ✅ CORRECT
- **Notes**: Key DGP variational structure from Damianou & Lawrence 2013.

### Line 12: "we only learn distributions over the much smaller $m_\ell \times d_\ell$ inducing outputs"
- **Status**: ✅ CORRECT
- **Notes**: Sparse approximation at each layer. Parameters scale with $m$, not $n$.

### Line 12: "The per-layer GP conditional... moves information from inducing values to all training points"
- **Status**: ✅ CORRECT
- **Notes**: GP conditional propagates information from $m$ inducing points to $n$ training points.

### Line 17: Layer conditional
```
p(\vF^{(\ell)} \mid \vU^{(\ell)}, \vF^{(\ell-1)}) = \mathcal{N}\left(\vF^{(\ell)} \mid A^{(\ell)} \vU^{(\ell)}, \, K^{(\ell)}_{ff} - A^{(\ell)} K^{(\ell)}_{uu} A^{(\ell)\top}\right)
```
- **Status**: ✅ CORRECT
- **Notes**: Standard GP conditional form with $A^{(\ell)} = K^{(\ell)}_{fu} (K^{(\ell)}_{uu})^{-1}$.

### Line 20: "The covariance matrices... depend on the random inputs $\vF^{(\ell-1)}$, creating dependencies across layers"
- **Status**: ✅ CORRECT
- **Notes**: Crucial point. This dependency is what makes DGPs intractable.

### Line 25: DGP ELBO
```
\mathcal{L} = \mathbb{E}_{q(\vF^{(1:L)})}[\log p(\vy \mid \vF^{(L)})] - \sum_{\ell=1}^{L} \KL(q(\vU^{(\ell)}) \| p(\vU^{(\ell)}))
```
- **Status**: ✅ CORRECT
- **Notes**: Standard DGP ELBO structure. One data term, $L$ KL terms.

### Line 28: "The KL terms have closed form since $q(\vU^{(\ell)})$ and $p(\vU^{(\ell)})$ are both Gaussian"
- **Status**: ✅ CORRECT
- **Notes**: Each layer's inducing point distribution is Gaussian, so KL is tractable.

### Line 28: "the expected log-likelihood requires integrating over all the layers outputs $\vF^{(1:L)}$ which has non-Gaussian dependencies"
- **Status**: ✅ CORRECT
- **Notes**: This is the intractable part that requires approximation.

### Line 33: Nested expectation
```
\mathbb{E}_{q}[\log p(\vy \mid \vF^{(L)})] = \int \cdots \int \log p(\vy \mid \vF^{(L)}) \prod_{\ell=1}^{L} q(\vF^{(\ell)} \mid \vF^{(\ell-1)}) \, d\vF^{(1)} \cdots d\vF^{(L)}
```
- **Status**: ✅ CORRECT
- **Notes**: Shows the nested integral structure that makes exact evaluation intractable.

### Line 44: Marginal layer distribution
```
q(\vF^{(\ell)} \mid \vF^{(\ell-1)}) = \mathcal{N}\left(\vF^{(\ell)} \mid A^{(\ell)} \vM^{(\ell)}, \, K^{(\ell)}_{ff} - A^{(\ell)}(K^{(\ell)}_{uu} - \vS^{(\ell)})A^{(\ell)\top}\right)
```
- **Status**: ✅ CORRECT
- **Notes**: Result of integrating out $\vU^{(\ell)}$ from $p(\vF^{(\ell)} \mid \vU^{(\ell)}, \vF^{(\ell-1)}) q(\vU^{(\ell)})$.

### Line 47: Interpretation of marginal distribution
- **Status**: ✅ CORRECT
- **Notes**: Good explanation of mean and variance structure. "takes off the uncertainty that is explained by the inducing points, then adds back only what $\vS^{(\ell)}$ says is left" is accurate.

### Line 47: "While Gaussian in $\vF^{(\ell)}$ when $\vF^{(\ell-1)}$ is fixed, the overall distribution is non-Gaussian because the covariance matrices $K^{(\ell)}$ depend non-linearly on the random $\vF^{(\ell-1)}$"
- **Status**: ✅ CORRECT
- **Notes**: Key point. Conditional Gaussian doesn't imply marginal Gaussian when covariance depends on conditioning variable.

### Line 52: Damianou mean-field approximation
```
q(\vF^{(\ell)}) = \mathcal{N}(\vF^{(\ell)} \mid \vmu^{(\ell)}, \vSigma^{(\ell)})
```
- **Status**: ✅ CORRECT
- **Notes**: Introduces additional variational distributions over layer outputs to make ELBO tractable.

### Line 55: "This 'mean-field' approximation breaks the dependencies between layers but introduces $O(nLd)$ additional parameters"
- **Status**: ✅ CORRECT
- **Notes**: Each layer has $n \times d_\ell$ outputs with mean and variance parameters.

### Line 55: "loses the structure that $\vF^{(\ell)}$ should be generated from $\vF^{(\ell-1)}$ via a GP"
- **Status**: ✅ CORRECT
- **Notes**: Mean-field assumption treats layers as independent, ignoring hierarchical structure.

### Line 60: Damianou ELBO
```
\mathcal{L}_{\text{MF}} = \mathbb{E}_{q(\vF^{(L)})}[\log p(\vy \mid \vF^{(L)})] - \sum_{\ell=1}^{L} \mathbb{E}_{q(\vF^{(\ell-1)})}\left[\KL(q(\vF^{(\ell)}) \| p(\vF^{(\ell)} \mid \vF^{(\ell-1)}))\right]
```
- **Status**: ✅ CORRECT
- **Notes**: Mean-field ELBO requires expectation over $q(\vF^{(\ell-1)})$ in KL terms.

### Line 63: "scalability issues (variational parameters grow with $n$) and may produce poor approximations due to the mean-field assumption"
- **Status**: ✅ CORRECT
- **Notes**: Two main limitations: scales with $n$ (not just $m$) and ignores dependencies.

---

## 4-doubly-stochastic.qmd - Doubly Stochastic Variational Inference

### Line 3: "avoids the mean-field approximation over layer outputs"
- **Status**: ✅ CORRECT
- **Notes**: Key advantage over Damianou's approach.

### Line 3: "use Monte Carlo estimation of the nested expectations while maintaining the exact conditional structure"
- **Status**: ✅ CORRECT
- **Notes**: MC samples propagated through layers preserve GP structure.

### Line 8: Variational distribution
```
q(\{\vF^{(\ell)}, \vU^{(\ell)}\}) = \prod_{\ell=1}^{L} p(\vF^{(\ell)} \mid \vU^{(\ell)}, \vF^{(\ell-1)}) q(\vU^{(\ell)})
```
- **Status**: ✅ CORRECT
- **Notes**: Same structure as in 3-dgp-spars-inf but with different inference approach.

### Line 11: "propagates samples through the layers... respects the hierarchical structure"
- **Status**: ✅ CORRECT
- **Notes**: Samples from layer $\ell-1$ become inputs to layer $\ell$, preserving dependencies.

### Line 16: DSVI ELBO decomposition
```
\mathcal{L} = \sum_{i=1}^{n} \mathbb{E}_{q(f_i^{(L)})}[\log p(y_i \mid f_i^{(L)})] - \sum_{\ell=1}^{L} \KL(q(\vU^{(\ell)}) \| p(\vU^{(\ell)}))
```
- **Status**: ✅ CORRECT
- **Notes**: Decomposes over data points. KL terms still closed form.

### Line 19: "The factorisation follows from $p(\vy \mid \vF^{(L)}) = \prod_i p(y_i \mid f_i^{(L)})$"
- **Status**: ✅ CORRECT
- **Notes**: Standard likelihood factorisation assumption.

### Line 22: Marginal output distribution (nested integral)
```
q(f_i^{(L)}) = \int \cdots \int p(f_i^{(L)} \mid \vU^{(L)}, f_i^{(L-1)}) q(\vU^{(L)}) \cdots p(f_i^{(1)} \mid \vU^{(1)}, \vx_i) q(\vU^{(1)}) \, d\vU^{(1)} \cdots d\vU^{(L)}
```
- **Status**: ✅ CORRECT
- **Notes**: Shows full propagation through all layers. This integral is intractable.

### Line 30: Layer 1 marginal
```
q(f_i^{(1)} \mid \vx_i) = \int p(f_i^{(1)} \mid \vU^{(1)}, \vx_i) q(\vU^{(1)}) \, d\vU^{(1)}
```
- **Status**: ✅ CORRECT
- **Notes**: First layer integrates out $\vU^{(1)}$.

### Line 37-39: Layer 1 mean and variance
```
\mu_i^{(1)} = k_i^{(1)\top} (K_{uu}^{(1)})^{-1} \vM^{(1)}
\sigma_i^{(1)2} = k_{ii}^{(1)} - k_i^{(1)\top} (K_{uu}^{(1)})^{-1} (K_{uu}^{(1)} - \vS^{(1)}) (K_{uu}^{(1)})^{-1} k_i^{(1)}
```
- **Status**: ✅ CORRECT
- **Notes**: Standard GP posterior mean and variance formula.

### Line 42: Interpretation of mean and variance
- **Status**: ✅ CORRECT
- **Notes**: "weighted sum of inducing values" (mean) and "prior variance minus inducing information plus variational uncertainty" (variance). Accurate.

### Line 47: Subsequent layer distribution
```
q(f_i^{(\ell)} \mid f_i^{(\ell-1)}) = \int p(f_i^{(\ell)} \mid \vU^{(\ell)}, f_i^{(\ell-1)}) q(\vU^{(\ell)}) \, d\vU^{(\ell)}
```
- **Status**: ✅ CORRECT
- **Notes**: Conditions on sampled value from previous layer.

### Line 54-56: Layer $\ell$ mean and variance
```
\mu_i^{(\ell)} = k^{(\ell)}(f_i^{(\ell-1)}, \vZ^{(\ell)})^\top (K_{uu}^{(\ell)})^{-1} \vM^{(\ell)}
\sigma_i^{(\ell)2} = k^{(\ell)}(f_i^{(\ell-1)}, f_i^{(\ell-1)}) - k^{(\ell)}(f_i^{(\ell-1)}, \vZ^{(\ell)})^\top (K_{uu}^{(\ell)})^{-1} (K_{uu}^{(\ell)} - \vS^{(\ell)}) (K_{uu}^{(\ell)})^{-1} k^{(\ell)}(\vZ^{(\ell)}, f_i^{(\ell-1)})
```
- **Status**: ✅ CORRECT
- **Notes**: Same formula as layer 1 but input is now sampled $f_i^{(\ell-1)}$ not fixed $\vx_i$.

### Line 59: "each layer outputs a Gaussian whose mean and variance depend on a random draw from the layer below"
- **Status**: ✅ CORRECT
- **Notes**: Key mechanism for uncertainty propagation in DGPs.

### Line 61: "kernel evaluations now depend on the random variable $f_i^{(\ell-1)}$"
- **Status**: ✅ CORRECT
- **Notes**: This dependency is what makes the marginal non-Gaussian.

### Line 66: MC ELBO estimate
```
\hat{\mathcal{L}} = \frac{n}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{1}{S} \sum_{s=1}^{S} \log p(y_i \mid f_i^{(L,s)}) - \sum_{\ell=1}^{L} \KL(q(\vU^{(\ell)}) \| p(\vU^{(\ell)}))
```
- **Status**: ✅ CORRECT
- **Notes**: Unbiased MC estimate. $S$ samples per data point, minibatch $\mathcal{B}$.

### Line 69: "$n/|\mathcal{B}|$ corrects for minibatch subsampling, ensuring the estimate remains unbiased"
- **Status**: ✅ CORRECT
- **Notes**: Standard minibatch scaling factor.

### Line 71: "doubly stochastic refers to two sources of randomness: minibatch subsampling... and Monte Carlo sampling through layers"
- **Status**: ✅ CORRECT
- **Notes**: Good explanation of the term. Both sources maintain unbiased gradients.

### Line 73: "reparametrisation trick [@kingma_auto-encoding_2014]"
- **Status**: ✅ CORRECT
- **Notes**: Key technique for backprop through stochastic nodes. $f = \mu + \sigma\epsilon$ with $\epsilon \sim \mathcal{N}(0,1)$.

### Line 73: "gradients with respect to $\mu$ and $\sigma$ flow through the deterministic transformation while the stochastic node $\epsilon$ carries no parameters"
- **Status**: ✅ CORRECT
- **Notes**: Accurate description of reparametrisation trick mechanics.

### Line 75: "Computational cost per iteration is $O(|\mathcal{B}| m^2 L S)$ for forward propagation"
- **Status**: ✅ CORRECT
- **Notes**: $|\mathcal{B}|$ data points × $S$ samples × $L$ layers × $m^2$ per layer operations.

### Line 75: "plus $O(m^3 L)$ for the KL terms"
- **Status**: ✅ CORRECT
- **Notes**: $L$ layers × $m^3$ Cholesky per layer for KL computation.

### Line 75: "Crucially, this is independent of $n$"
- **Status**: ✅ CORRECT
- **Notes**: Minibatch approach means cost doesn't depend on full dataset size.

### Line 76: "a single sample ($S=1$) often suffices when $|\mathcal{B}|$ is large"
- **Status**: ✅ CORRECT
- **Notes**: Empirical observation. Averaging over many data points provides variance reduction.

---

## Summary of Issues Found

### Critical Issues
None found. All mathematics appears correct.

### Minor Issues
1. **Line 25, 2-sparse-var-inf.qmd**: "weighted result" should be "weighted sum"
   - **Severity**: Low (meaning is clear from context)

### Potential Improvements
1. **Throughout**: Could add more explicit dimension tracking for readers
2. **3-dgp-spars-inf.qmd, Line 55**: Could quantify "$O(nLd)$" more precisely as "$nLd$ means + variance parameters"

### Strong Points
1. ✅ Mathematical derivations are all correct
2. ✅ Intuitive explanations accompany formal math
3. ✅ Good progression from simple (standard GP) to complex (DGP with DSVI)
4. ✅ Citations are appropriate and well-placed
5. ✅ Complexity analysis is accurate throughout
6. ✅ New intro to 3-dgp-spars-inf.qmd successfully bridges from tractable GP case

### Logical Flow
The section follows a clear pedagogical progression:
1. ✅ Standard GP: tractable via Gaussian conjugacy
2. ✅ Variational inference: general framework for intractability
3. ✅ Sparse VI: scaling to large data
4. ✅ DGP intractability: explain why conjugacy breaks
5. ✅ DGP VI approaches: mean-field (Damianou) then doubly stochastic (Salimbeni)

The new introductory paragraph in 3-dgp-spars-inf.qmd successfully addresses the gap identified by the user.
