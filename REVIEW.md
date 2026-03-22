# Dissertation Review

**Reviewed:** All 24 QMD section files, all shared Python modules, `kin8nm/src/run.py`, `kin8nm/src/data/data_preprocessor.py`, all 6 metrics CSVs, and `bootstrap_ci.py`.

**Overall assessment:** The dissertation is well-written, mathematically rigorous, and the experiments are reproducible and honestly reported. The theoretical progression (weight-space → GP → DGP → variational inference → experiments) is logical and clear. Key issues are: one notation error in the TDGP section, one figure caption error, several inconsistencies between the appendix tables and the main results, a cross-dataset mix in a summary statistic, a discrepancy in ECE values between tables due to two different ECE implementations, and a handful of grammar/style issues. No train/test leakage was found.

---

## Section-by-Section Review

---

## 0. Introduction — `sections/0-intro/intro.qmd`

### Paragraph 1 (lines 4–5): GP framework motivation

**Good:** Clear, accurate description of GPs. The sentence "The flexibility of GPs comes from their kernel functions, which encode prior assumptions about function structure" is precise.

**Errors:** None.

**Content Review:** The phrase "principled uncertainty quantification" does not appear here but is used in subsequent paragraphs. Consider introducing it here since it becomes the main GP-family selling point later.

**Methodology:** None.

**Link to rest of diss:** Flows well into the GP section.

**Grammar:** None.

---

### Paragraph 2 (lines 6): DGP motivation

**Good:** Correct framing of the expressiveness limitation.

**Errors:** Citation `@pmlr-v117-pravesh20a` is used to support "the expressiveness of a standard GP is limited by its kernel". Pravesh et al. (AISTATS 2020) is a niche paper on GP inducing features — it is an unusual/questionable choice here. The standard citations for this claim are [@rasmussen_gaussian_2008] or [@damianou_deep_2013] directly. **Verify this citation is correct and actually supports the claim.**

**Content Review:** None.

**Grammar:** "Further to this" is slightly formal British idiom — fine for a UK dissertation but could be "Additionally".

---

### Paragraph 3 (lines 8): DGP composition

**Good:** Accurate description of DGP composition and the tractability cost.

**Content Review:** "much like deep neural networks" — this analogy reappears at least 3 times across the dissertation (intro, 0-dgps.qmd, 1-dgp-def.qmd). It is appropriate once, but repeated use may be distracting. Consider keeping it only in the intro and using a different framing elsewhere.

**Grammar:** None.

---

### Paragraph 4 (lines 10–12): Dissertation structure

**Good:** Clear roadmap.

**Errors:** None.

**Content Review:** "We discuss advantages, limitations and tradeoffs between DGP structures" — the comparison is primarily between DGPs and classical ML as well. This sentence undersells the scope. Could say "...between GP-family models and classical machine learning baselines".

**Grammar:** "between" should be "among" when referring to 3+ items (DGP structures plural). Minor.

---

### Citations in this section

- `@pmlr-v117-pravesh20a` — **suspect**. Verify the Pravesh 2020 paper supports "GP expressiveness is limited by its kernel". Likely should be `@rasmussen_gaussian_2008` or `@damianou_deep_2013`.
- `@damianou_deep_2013` — correctly cited as the DGP origin paper. ✓

---

## 1.0 GP Introduction — `sections/1-gps/0-gp-intro.qmd`

### Paragraph 1 (lines 3–7): Problem setup and parametric models

**Good:** Clear motivation, correct framing of supervised ML.

**Errors:** None.

**Content Review:** The text distinguishes "frequentist (MLE)" from "Bayesian (GP)" but the real axis should be **parametric vs. non-parametric**, not frequentist vs. Bayesian. Bayesian linear regression is parametric AND Bayesian. The GP's defining property is being non-parametric, not being Bayesian per se. The framing at line 9 is slightly misleading: *"In contrast to frequentist approaches that specify a parametric model… GPs place a prior directly on an infinite-dimensional function space"* — one can have a Bayesian parametric model that is still limited by its parameterisation. The contrast should be: **parametric** (fixed-dimensional function space) vs. **non-parametric** (function space grows with data). The Bayesian aspect is separate.

**Link to rest of diss:** Fine — leads into weight-space → GP development.

**Grammar:** None.

---

### Citations in this section

- `@bishop_pattern_2006`, `@murphy_machine_2013` — correctly cited for parametric Bayesian methods. ✓
- `@rasmussen_gaussian_2008`, `@mackay_introduction_1998`, `@williams_gaussian_1995` — correctly cited for GPs. ✓

---

## 1.1 Weight Space View — `sections/1-gps/1-weight-space.qmd`

### Basis function expansion (lines 5–16)

**Good:** Clear and correct presentation of the basis function model. The distinction "linear in parameters, non-linear in inputs" is properly made.

**Errors:** None.

**Content Review:** None.

**Grammar:** "which we call $\hat{\vw}$" at line 17 — slightly informal. "denoted $\hat{\vw}$" would be more standard.

---

### Bayesian linear regression / prior and likelihood (lines 21–45)

**Good:** Correct Gaussian prior and likelihood setup. Well-explained. The reference to the appendix for design matrix construction is good practice.

**Errors:** None.

**Content Review:** `@gelman_bayesian_2025` — is this a 2025 edition? Verify the year is correct (usually cited as Gelman et al., 3rd ed. 2013 or BDA3). If this is a 2025 print, fine; otherwise this may be a citation error.

**Grammar:** "so-called *Bayesian linear regression* [@bishop_pattern_2006; @gelman_bayesian_2025]" — "so-called" is mildly dismissive. "known as *Bayesian linear regression*" is more neutral.

---

### Posterior over weights (lines 47–70)

**Good:** Correct posterior mean and covariance. Math verified ✓. Reference to appendix for derivation is appropriate.

**Errors:** None.

**Link to rest of diss:** The posterior mean $\bar{\vw} = \sigma_n^{-2}\Sigma_w\Phi^\top\vy$ is used in the predictive distribution derivation. ✓

---

### Predictive distribution (lines 72–87)

**Good:** Correct closed-form predictive mean and variance. ✓

**Errors:** None.

---

### From weight space to function space (lines 88–141)

**Good:** The matrix inversion lemma application is correct — verified algebraically. ✓ The step-by-step derivation showing $\vphi$ vanishes is one of the clearest sections in the dissertation.

**Errors:** The derivation at line 107 states: $\E[y_*] = \sigma_n^{-2}\vphi(\vx_*)^\top\Sigma_p\Phi^\top\vy - \sigma_n^{-2}\vk_*^\top(K+\sigma_n^2\vI)^{-1}K\vy$. This is correct: $\sigma_n^{-2}\vphi(\vx_*)^\top\Sigma_p\Phi^\top = \sigma_n^{-2}\vk_*^\top$. The subsequent simplification to $\vk_*^\top(K+\sigma_n^2\vI)^{-1}\vy$ is correct. ✓

**Content Review:** The citation `@williams_computing_1996` for the "infinite hidden units → GP" result is correct. ✓

**Link to rest of diss:** The predictive equation in @eq-predictive-kernel-full is explicitly matched to the GP predictive in @eq-gp-predictive later, which is a good connection. ✓

---

### Citations in this section

- `@rasmussen_gaussian_2008`, `@mackay_introduction_1998` — correctly cited throughout. ✓
- `@scholkopf_learning_2002` for the kernel function definition — correct. ✓
- `@williams_computing_1996` for neural network → GP limit — correct. ✓
- `@mackay_practical_1992` for i.i.d. Gaussian weight prior — correct. ✓

---

## 1.2 Gaussian Processes — `sections/1-gps/2-gaussian-processes.qmd`

### GP definition and prior (lines 1–18)

**Good:** Formally correct GP definition. ✓

**Errors:** None.

**Content Review:** "any finite number of which have a joint Gaussian distribution" — this is the standard definition from Rasmussen & Williams. ✓ The note "non-zero mean functions can be useful for encoding strong prior beliefs" is a good practical point.

---

### Kernels and covariance functions (lines 19–103)

**Good:** SE and periodic kernels correctly defined. ✓ The code figure is illustrative.

**Errors:**
1. **Figure caption error** (`fig-gp-regression`, lines 163–164): Caption says "conditioning on **five observations** (red dots)" but the code array `X_train_np = np.array([-3.5, -1.5, -1.6, -1.4, 0.0, 2.0, 4.0])` contains **7 points**. The caption should say "seven observations". This is a factual error.

**Content Review:** The Matérn and rational quadratic kernels are correctly described at line 101 with an appropriate citation. ✓

**Grammar:** Line 119: Long run-on sentence: "Additionally, since kernel hyperparameters are typically constrained to be positive, as a negative length-scale or noise variance would not make sense, it is standard practice to optimise them in log-space to ensure positivity and improve numerical stability." Split at the comma: "...would not make sense. It is therefore standard practice...".

---

### Kernel hyperparameters (lines 105–119)

**Good:** LML expression and interpretation of three terms is excellent — pedagogically clear.

**Errors:** None.

**Content Review:** Citation `@kingma_adam_2015` for Adam optimiser — correctly cited. ✓ Mention of L-BFGS is appropriate.

**Methodology:** LML non-convexity and local optima correctly acknowledged. ✓

---

### GP Regression (lines 121–157)

**Good:** Joint distribution and posterior derivations are correct. ✓ The connection back to @eq-predictive-kernel-full is explicit and good.

**Errors:** None. Note: @eq-gp-posterior-moments correctly shows the LATENT posterior (without $+\sigma_n^2$), while @eq-gp-predictive correctly includes $+\sigma_n^2$ for the noisy observation. This distinction is properly maintained throughout. ✓

**Content Review:** The intuitive paragraph at lines 154–157 explaining what the posterior mean and variance represent is excellent.

---

### Training a GP (lines 271–372) and Prediction (lines 374–388)

**Good:** LML gradient derivation is correct (verified against Rasmussen & Williams Eq. 5.9). ✓ The epistemic/aleatoric variance decomposition is correct and well-cited. ✓

**Errors:** None.

**Content Review:** The `@fig-gp-training` code uses `torch.manual_seed(42)` but the caption description of a "poorly initialised" GP is deliberately constructed. This is fine as an illustration.

**Grammar:** Line 287: "a GP with a poorly initialised length-scale" — "poorly initialised" could be "intentionally misconfigured" to be more accurate.

---

### Citations in this section

- `@rasmussen_gaussian_2008` — cited correctly throughout. ✓
- `@snelson_sparse_2006`, `@titsias_variational_2009` — correctly cited for sparse/approximate methods. ✓
- `@kendall_what_2017` — cited for epistemic/aleatoric decomposition. ✓

---

## 2.0 DGP Motivation — `sections/2-dgps/0-dgps.qmd`

**Good:** Succinct, correct framing. The link to @sec-learning is clean.

**Content Review:** "much like deep neural networks" — as noted in the intro, this analogy is used again. Fine here as it's the natural DGP introduction.

---

## 2.1 DGP Definition — `sections/2-dgps/1-dgp-def.qmd`

### Definition and generative model (lines 1–47)

**Good:** Formal definition of L-layer DGP is correct. The generative model equations are well-structured. ✓ The Markov property is correctly identified and explained. ✓

**Errors:** None.

**Content Review:** Line 61: "This leads to $\sum_{\ell=1}^L n \cdot d_\ell = O(nLd)$ total latent variables." This is correct if $d_\ell \approx d$ for all layers. ✓

---

### Multi-output / composed function (lines 49–63)

**Good:** The composed function notation and multi-output convention are standard.

**Errors:** None.

---

### Non-stationarity and expressiveness (lines 123–129)

**Good:** The composed covariance derivation at @eq-composed_covariance is correct. ✓ The "stretch/compress" intuition is excellent — one of the best expository passages in the dissertation.

**Errors:** None.

**Content Review:** Citations `@damianou_deep_2013` and `@dunlop_how_2018` are correctly used here. ✓

---

### Citations in this section

- `@damianou_deep_2013` — origin paper for DGPs. ✓
- `@dunlop_how_2018` — correctly cited for DGP non-stationarity analysis. ✓

---

## 2.2 Intractability — `sections/2-dgps/2-intractbility.qmd`

**Good:** Clear and correct proof of intractability. The "infinite mixture of Gaussians with varying covariance" argument is mathematically sound. ✓ The two-point example (lines 27–53) effectively illustrates the non-Gaussianity of the marginal. ✓

**Errors:** None.

**Content Review:** Line 11: "unless the kernels are degenerate" — more precisely, this should say "unless the kernel is linear (degenerate)" or "unless the intermediate mapping is linear", since a linear kernel would make the composition a single GP. The word "degenerate" without qualification is vague.

**Grammar:** "As well as the intractability of the marginal likelihood, the resulting distribution over outputs is non-Gaussian." — this is two separate claims. It would be cleaner as: "The marginal likelihood is intractable (as shown above). Furthermore, the resulting distribution over outputs is non-Gaussian, as the following example shows."

---

### Citations in this section

- `@damianou_deep_2013`, `@salimbeni_doubly_2017` — correctly cited as motivating approximate inference. ✓

---

## 3.0 Learning Overview — `sections/3-training/0-training.qmd`

**Good:** Concise roadmap paragraph. The arc (GP hyperparameter learning → variational inference → doubly stochastic) matches the actual section structure. ✓

---

## 3.01 GP Training — `sections/3-training/01-gp-training.qmd`

**Good:** The marginalisation derivation (line 7) is correct. ✓ The gradient formula at @eq-gradient is the standard Rasmussen & Williams form, equivalent to the expression in 2-gaussian-processes.qmd — both are verified correct. ✓

**Errors:** None.

**Content Review:** The interpretation of $\valpha\valpha^\top - K_y^{-1}$ (line 18) is excellent — data-fit vs. model complexity balance is a key insight.

**Grammar:** Line 20: "For smaller datasets this is typically feasible, however as datasets grow, $O(n^3)$ scaling becomes prohibitive for large datasets." — comma splice. Should be: "For smaller datasets this is typically feasible; however, as datasets grow...".

---

## 3.1 Variational Inference — `sections/3-training/1-var-inf.qmd`

### Opening / ELBO decomposition (lines 1–28)

**Good:** ELBO decomposition is correct, verified from first principles. ✓ The proof of @eq-vi_decomposition is correct. ✓

**Errors:** None.

**Grammar:** Line 3: "The idea is to approximate the true posterior $p(\vz \mid \vy)$ with a simpler, tractable distribution $q(\vz)$, which is chosen in order to minimise the Kullback–Leibler (KL) divergence between $q$ and $p$:" — The sentence ending in a colon is a fragment (no main clause after the colon, only a blank line and then a new paragraph). This looks like missing text — either a direct equation should follow or the colon should be removed.

---

### Gaussian variational distribution (lines 33–50)

**Good:** Gaussian KL formula at @eq-kl_gaussian is correct. ✓ The interpretation of each KL term is well done.

**Errors:** None.

**Content Review:** Good motivation for choosing a Gaussian variational family.

---

### Citations in this section

- `@jordan_introduction_1999`, `@blei_variational_2017` — correctly cited for variational inference. ✓
- `@bishop_pattern_2006` — correct for ELBO interpretation. ✓

---

## 3.2 Sparse Variational GPs — `sections/3-training/2-sparse-var-inf.qmd`

### Inducing point setup (lines 1–33)

**Good:** Sparse variational GP framework is correctly presented per Titsias (2009). The factorisation $q(\vf, \vu) = p(\vf \mid \vu) q(\vu)$ is the key insight, clearly explained. ✓

**Errors:** None.

**Content Review:** Line 5: "If we consider inducing inputs…" — "If" makes this seem conditional when it is not. "Consider inducing inputs…" or "Let $\vZ = \{\vz_j\}_{j=1}^m$…" is cleaner.

---

### ELBO and optimal parameters (lines 35–65)

**Good:** The ELBO @eq-sparse_elbo is correct. ✓ Optimal variational parameters at @eq-optimal_params verified. ✓ Sparse predictions at @eq-sparse_predictions verified. ✓

**Errors:** None.

**Content Review:** Note for the reader: the Titsias ELBO, when the Gaussian likelihood expectation is fully expanded, includes a trace term $-\frac{1}{2\sigma_n^2}\text{tr}(K_{ff} - Q_{ff})$ (where $Q_{ff} = K_{fu}K_{uu}^{-1}K_{uf}$) that penalises approximation error. This trace term is implicitly captured in $\E_{q(\vf)}[\log p(\vy \mid \vf)]$ but is not explicitly shown. For completeness, a brief note would help readers consulting Titsias (2009) directly and seeing the explicit form.

---

### Stochastic optimisation (lines 67–82)

**Good:** Minibatch ELBO derivation at @eq-sparse_minibatch_elbo is correct. ✓ The $n/|\mathcal{B}|$ rescaling explanation is clear. ✓

**Errors:** None.

**Methodology:** The statement "The KL term does not depend on the data and can be evaluated once per step" — this is correct in that the KL is a function of variational parameters and hyperparameters only, not of individual data points. ✓

---

### Citations

- `@quinonero-candela_unifying_2005`, `@titsias_variational_2009` — correctly cited. ✓
- `@hensman_gaussian_2013` — correctly cited for minibatch VI. ✓

---

## 3.3 DGP Variational Inference — `sections/3-training/3-dgp-spars-inf.qmd`

### Opening (lines 1–3)

**Good:** Correctly recaps why GP conjugacy breaks in DGPs.

**Content Review:** "This breaks in DGPs because…" — slightly informal. "This property does not hold in DGPs" is cleaner.

---

### Variational factorisation and layer conditional (lines 5–28)

**Good:** The variational factorisation @eq-dgp_variational_factorization is the Salimbeni & Deisenroth (2017) approach. ✓ The layer conditional is correct. ✓

**Errors:** None.

**Content Review:** The phrase "Following from work by @damianou_deep_2013 and @titsias_variational_2009" correctly attributes the inducing point idea to Titsias and the DGP extension to Damianou. ✓

---

### DGP ELBO (lines 22–64)

**Good:** The DGP ELBO @eq-dgp_elbo is standard. ✓ The Gaussian marginal after integrating out $\vU^{(\ell)}$ at @eq-marginal_q_layer_explicit is correct. ✓

**Errors:** None.

**Content Review:** The paragraph explaining the Damianou (2013) mean-field approximation is fair and accurate. The limitations (grows with $n$, breaks hierarchical structure) are correctly identified. ✓

**Methodology:** The text notes the mean-field ELBO @eq-damianou_elbo introduces variational parameters $\{\vmu^{(\ell)}, \vSigma^{(\ell)}\}$ that "grow with $n$". This is a correct and important point about scalability. ✓

---

### Citations

- `@damianou_deep_2013`, `@titsias_variational_2009` — correctly cited. ✓
- `@salimbeni_doubly_2017` — correctly positioned as the improvement. ✓

---

## 3.4 Doubly Stochastic VI — `sections/3-training/4-doubly-stochastic.qmd`

### DSVI overview and ELBO (lines 1–17)

**Good:** Correct description of Salimbeni & Deisenroth (2017). ✓ The per-data-point ELBO @eq-dsvi_elbo is correct. ✓

**Errors:** None.

---

### Layer-wise moments (lines 25–51)

**Good:** Layer 1 mean and variance at @eq-layer1_moments are correct. ✓ The recursive extension to subsequent layers at @eq-layer_ell_moments is correct. ✓

**Errors:** None.

**Content Review:** The uncertainty propagation explanation (line 51: "uncertainty compounds as samples pass through each layer") is accurate and well-worded.

---

### MC ELBO and doubly stochastic sources (lines 53–65)

**Good:** @eq-dsvi_mc_elbo is correct. ✓ The "doubly stochastic" naming (minibatch + layer samples) is correctly explained. ✓

**Errors:** None.

**Content Review:** The reparameterisation trick is correctly cited to `@kingma_auto-encoding_2014`. ✓

---

### Citations

- `@salimbeni_doubly_2017` — correctly cited as the doubly stochastic VI paper. ✓
- `@kingma_auto-encoding_2014` — correctly cited for reparameterisation trick. ✓

---

## 3.5 Method Comparison — `sections/3-training/6-comparison.qmd`

### Complexity table (lines 8–16)

**Good:** Complexity table is well-structured.

**Errors:** The mean-field DGP row shows $O(n^3 L)$ training. This is correct for the Damianou (2013) approach where variational parameters $\vmu^{(\ell)}, \vSigma^{(\ell)}$ are $n \times d_\ell$ and require $O(n^3)$ operations. ✓

**Content Review:** Note: the training entry for Doubly Stochastic DGP should technically include the $+O(m^3 L)$ for KL terms (mentioned later in the text), though $O(|\mathcal{B}| m^2 L)$ dominates for typical settings.

---

### Approximation quality (lines 18–22)

**Good:** The mode-seeking ($\KL(q \| p)$) vs. mass-covering ($\KL(p \| q)$) trade-off is correctly identified. ✓

---

### Practical considerations (lines 24–33)

**Good:** Good practical advice. The inducing point placement, natural gradient, and initialisation points are all correctly described. ✓

**Content Review:** Line 30: "For Gaussian variational posteriors the natural gradient update is equivalent to a single Newton step in the natural parameter space" — this is the result from Salimbeni et al. (2018). Correct. ✓

**Methodology:** The "Rao-Blackwellisation" mention at line 28 is appropriate but could confuse readers unfamiliar with this technique. A brief note that it reduces gradient variance by analytically marginalising some variables would help.

---

### Empirical performance (lines 34–40)

**Content Review:** The last paragraph references `@bui_deep_2016` for increasing inducing points / sample count. This is a correct citation context. ✓

---

### Citations

- `@titsias_variational_2009` — ✓
- `@salimbeni_doubly_2017` — ✓
- `@salimbeni_natural_2018` — ✓
- `@amari_natural_1998` — ✓
- `@ranganath_black_2014` — correctly cited for variance reduction. ✓
- `@duvenaud_avoiding_2014` — correctly cited for pathologies. ✓
- `@bui_deep_2016` — ✓

---

## 4.0 Advanced Architectures Introduction — `sections/4-advanced/0-advanced-intro.qmd`

**Good:** Clean, accurate overview. Both architectures are correctly described at a high level. The caveat that they are not experimentally evaluated is appropriate.

**Errors:** None.

**Content Review:** `@duvenaud_avoiding_2014` correctly cited for DGP pathologies. ✓

---

## 4.1 Thin and Deep GPs — `sections/4-advanced/1-thin-deep.qmd`

### Opening and motivation (lines 1–5)

**Good:** Correct description of both Duvenaud (2014) pathologies and Heinonen (2016) approach. ✓

---

### TDGP layer definition (lines 7–27)

**Good:** The concept is correctly described.

**Errors:**
1. **Notation error** at @eq-tdgp-layer (line 16): $\vH^{(\ell)} = \vX\,\text{diag}(g^{(\ell)}(\vX))$ is incorrect notation. `diag()` applied to a matrix $\mathbb{R}^{n \times d}$ returns a vector (the diagonal), not a matrix. The intended operation is **element-wise (Hadamard) product**: $\vH^{(\ell)} = \vX \odot g^{(\ell)}(\vX)$, or equivalently, $H^{(\ell)}_{ij} = X_{ij} \cdot g^{(\ell)}_j(\vx_i)$. The text description ("each input dimension is rescaled by a smooth, spatially varying factor") confirms this is element-wise multiplication, not a diagonal-matrix product. Fix: replace $\vX\,\text{diag}(g^{(\ell)}(\vX))$ with $\vX \odot g^{(\ell)}(\vX)$ and clarify that $g^{(\ell)}(\vX) \in \mathbb{R}^{n \times d}$ applies one GP per input dimension evaluated at each data point.

**Content Review:** None.

---

### Pathology avoidance (lines 28–38)

**Good:** Correct explanation of why TDGP avoids collapse.

**Errors:** Line 32: "The Jacobian of the composition $\phi^{(L)} \circ \cdots \circ \phi^{(1)}$ is a product of **diagonal-plus-rank-one** matrices". For the mapping $\phi^{(\ell)}(\vx) = \vx \odot g^{(\ell)}(\vx)$, the Jacobian with respect to $\vx$ is $\text{Diag}(g^{(\ell)}(\vx)) + \text{rank-}d$ correction (one rank-1 term per output dimension, summing to rank $d$). Calling it "rank-one" is imprecise — it should be "rank-$d$" (or "full-rank perturbation") for the general case. However, if each dimension of $g^{(\ell)}$ is an independent scalar GP (so only the $j$-th output responds to inputs), the Jacobian entry $\partial H^{(\ell)}_{ij}/\partial x_{ik} = g^{(\ell)}_j(\vx_i)\delta_{jk} + x_{ij}\partial g^{(\ell)}_j/\partial x_{ik}$, which for a single output $j$ is indeed rank-1. The "diagonal-plus-rank-one" description may come from Souza et al. (2023); if so, cite this claim directly to that paper rather than stating it as a derived fact.

---

### Dimensionality reduction and inference (lines 34–47)

**Good:** ARD as automatic dimension selection is correctly described. The TDGP ELBO at @eq-tdgp-elbo matches the standard DSVI ELBO. ✓

---

### Citations

- `@souza_thin_2023` — correctly cited throughout. ✓
- `@duvenaud_avoiding_2014` — correctly cited for pathologies. ✓
- `@heinonen_nonstationary_2016` — correctly cited for GP-parameterised lengthscale. ✓
- `@salimbeni_doubly_2017` — correctly cited for inference. ✓

---

## 4.2 Deeply Non-Stationary GPs — `sections/4-advanced/2-deeply-non-stationary.qmd`

### Non-stationarity through composition (lines 1–19)

**Good:** The effective lengthscale derivation via first-order Taylor expansion is mathematically correct. ✓ @eq-effective-lengthscale is well-derived. ✓

**Errors:** None.

---

### Architecture design choices (lines 21–33)

**Good:** Three design choices (input-connected layers, lengthscale parameterisation, linear mean functions) are correctly described.

**Errors:** None.

**Content Review:** Line 33: Describing linear mean functions as providing a "skip connection" is an apt analogy to ResNets, though the analogy should not be overextended.

---

### Gibbs kernel and connection (lines 101–119)

**Good:** Gibbs kernel at @eq-gibbs-kernel is correctly defined. ✓ The DGP-Gibbs comparison is accurate. ✓

**Errors:** None.

---

### Citations

- `@paciorek_nonstationary_2004` — correctly cited for non-stationary kernels. ✓
- `@salimbeni_deep_2019` — correctly cited throughout. ✓
- `@duvenaud_avoiding_2014` — correctly cited for collapse pathology. ✓

---

## 4.3 Model Evaluation Overview — `sections/4-modeling/0-overview.qmd`

### Opening (lines 1–10)

**Good:** Clear framing of the two experiments. The model set (3 classical + 3 GP-family) is well-defined.

---

### Hyperparameter setup (lines 11+)

**Errors:**
1. **Inducing point asymmetry not discussed**: SVGP uses 512 inducing points vs 128 per layer for DSVGP. This gives SVGP a substantially larger variational parameter budget (512 inducing points × 1 layer = 512 vs 128 × 2 = 256 for DSVGP-2L). While the experimental design is practical and defensible, this asymmetry could disadvantage DGP models. It should be explicitly acknowledged — even one sentence noting the inducing point counts were chosen for computational tractability rather than fairness.

**Content Review:** "Hyperparameters for all models were fixed based roughly on established values from prior work" — "based roughly on" is poor phrasing. Better: "Hyperparameters were set based on established defaults from prior work, without systematic search."

**Methodology:** The lack of hyperparameter tuning is acknowledged, which is honest. For a dissertation, grid search or Bayesian optimisation would improve rigour. This is noted correctly as a limitation. The claim "based off available compute resources" is fine but "based on" is more standard than "based off".

---

## 4.4 Kin8nm Results — `sections/4-modeling/1-kin8nm.qmd`

### Dataset description (lines 1–5)

**Good:** Correct description of Kin8nm. Properly referenced.

---

### Experimental design (lines 7–15)

**Good:** Clear split protocol.

**Errors:** The text says "$N_{\text{dev}} = 500$: 425 training, 75 validation" (85/15 split). Checking: 500 × 0.85 = 425, 500 × 0.15 = 75. ✓ Similarly for medium: 3000 × 0.85 = 2550, 3000 × 0.15 = 450. ✓ Large: 6553 × 0.85 ≈ 5570, 6553 × 0.15 ≈ 983. ✓

---

### Results table — full scale (lines 17–28)

**Good:** All values verified against `kin8nm/large/metrics.csv`. Every number matches to the precision shown. ✓

| Metric | CSV | Table | Match? |
|--------|-----|-------|--------|
| Ridge RMSE | 0.20233 | 0.2023 | ✓ |
| RF RMSE | 0.14820 | 0.1482 | ✓ |
| XGBoost RMSE | 0.11899 | 0.1190 | ✓ |
| SVGP RMSE | 0.07832 | 0.0783 | ✓ |
| DSVGP-2L RMSE | 0.06986 | 0.0699 | ✓ |
| DSVGP-3L RMSE | 0.07590 | 0.0759 | ✓ |
| SVGP NLPD | -1.1163 | -1.116 | ✓ |
| DSVGP-2L NLPD | -1.2406 | -1.241 | ✓ |
| DSVGP-3L NLPD | -1.1575 | -1.158 | ✓ |

**Errors:** None — numbers are correctly reported.

---

### Analysis and hidden representation (lines 32–59)

**Good:** The non-stationarity explanation for DGP advantage on Kin8nm is theoretically sound. ✓ The claim "DSVGP-3L R² = 0.002 at small scale" is verified (CSV: 0.0016 → rounds to 0.002). ✓

**Errors:** The specific percentages at line 38 ("first two PCs capture only 27.2% of the variance", "first PC of hidden space alone captures 47.8%", "input-space PC1's 13.6%") cannot be verified from the CSV files alone — they come from the PCA visualisation code in `visualise_hidden.py`. These claims are plausible given the experimental setup, but they should be reproducible from the committed code. **Ensure these numbers are computed deterministically and match the figure.**

---

### Scaling table (lines 44–59)

**Good:** All scaling values verified against the three CSV files. Every number correct. ✓

**Errors:** None.

**Content Review:** The runtime appendix table says "approximately 4× the cost of DSVGP-2L" for Kin8nm (152.4/36.2 = 4.21). But the main conclusion section says "adds approximately 3.6× training time" — the 3.6× figure comes from the **credit card** dataset (706.5/197.0 = 3.59). The conclusion conflates the two datasets. At a minimum, the conclusion should either say "≈3.6–4×" or distinguish per dataset. See also the detailed note under the Conclusion section.

---

### Citations in this section

- `@salimbeni_doubly_2017`, `@damianou_deep_2013` — correctly cited for Kin8nm DGP literature. ✓
- `@duvenaud_avoiding_2014` — correctly cited for pathological small-data behaviour. ✓

---

## 4.5 Financial Risk Introduction — `sections/4-modeling/1.5-finance-intro.qmd`

**Good:** Concisely motivates GP/DGP for credit risk. Citations to Basel III and EBA guidelines are appropriate for a UK academic context. ✓

**Content Review:** "GP-based methods have been applied to credit-risk and credit-rating problems [@mahajan_gaussian_2022; @huang_using_2011]" — verify these citations actually concern credit-risk GP models specifically, not merely GP methods in adjacent contexts.

---

## 4.6 UCI Credit Card Results — `sections/4-modeling/1-uci-credit-card-default.qmd`

### Dataset description and preprocessing (lines 1–16)

**Good:** Preprocessing steps (log-transform, one-hot, standardisation) are clear and the anti-leakage note is explicit. ✓

**Errors:** None.

**Content Review:** Line 9: "all continuous predictors were standardised to zero mean and unit variance fit on the training split only" — missing comma. Should be "...unit variance, fit on the training split only". Minor grammar.

**Methodology:** Line 16: "All models are then retrained on the combined train+validation set, following the convention for small-data GP experiments in @hensman_gaussian_2013." — The Hensman (2013) citation is for **stochastic VI for GPs** (minibatch training), not specifically for the retrain-on-full-data convention. This citation is **misleading**. The retrain-on-full-data protocol is a standard practice, but Hensman (2013) is not the appropriate citation. Suggest removing the citation or replacing with a more general model selection reference.

---

### Large-scale results (lines 22–41)

**Good:** All values verified against `credit-card-pd-uci/large/metrics.csv`. Every number correct. ✓

| Metric | CSV | Table | Match? |
|--------|-----|-------|--------|
| LR ROC-AUC | 0.7388 | 0.739 | ✓ |
| RF ROC-AUC | 0.7689 | 0.769 | ✓ |
| XGBoost ROC-AUC | 0.7792 | 0.779 | ✓ |
| SVGP ROC-AUC | 0.7777 | 0.778 | ✓ |
| SVGP ECE | 0.01050 | 0.010 | ✓ |
| DSVGP-2L ECE | 0.01352 | 0.014 | ✓ |
| DSVGP-3L ECE | 0.02061 | 0.021 | ✓ |
| LR ECE | 0.22528 | 0.225 | ✓ |
| RF ECE | 0.08688 | 0.087 | ✓ |
| XGBoost ECE | 0.19615 | 0.196 | ✓ |

**Errors:** None.

**Content Review:** Line 39: "GP models' lower optimal thresholds (0.30 vs 0.35–0.50 for classical models) reflect narrower posterior score distributions — a consequence of principled uncertainty quantification rather than a deficiency." This is a sound interpretation: well-calibrated models with a 22.1% base rate will score near 0.22 on average, requiring a lower threshold to capture defaults. ✓

Line 41: The SR 11-7 citation is appropriate. ✓ However, "adds validation burden under SR 11-7" implies SR 11-7 places the model risk management burden on recalibration — this is a reasonable interpretation for an illustrative discussion, but SR 11-7 is a US Federal Reserve guidance document, not a UK/EU regulatory requirement. This should be caveated or replaced with the UK PRA equivalent (SS3/18) for a UK dissertation.

---

### Scaling behaviour (lines 43–71)

**Good:** All scaling values verified against the three CSV files. Every number correct. ✓

**Errors:** None.

**Content Review:** Line 69: "the anomalously low ECE = 0.005 for DSVGP-3L at small scale does not indicate good model performance" — this is correctly explained. The model predicts near the base rate (22.1%) for all inputs, so it appears calibrated by ECE while being discriminatively random. This is a subtle but important point. ✓

**Methodology:** The ECE for DSVGP-3L at small scale (ECE = 0.005, ROC-AUC = 0.523) highlights a limitation of ECE as a standalone metric: a trivially constant model can achieve excellent ECE. The text acknowledges this but could more explicitly warn that ECE should always be interpreted alongside discrimination metrics.

---

### Computational profile (lines 73–88)

**Good:** All runtime values verified against CSV. ✓ The 3.6× claim for DSVGP-3L vs DSVGP-2L at large scale: 706.5/197.0 = 3.59× ≈ 3.6×. ✓

---

### Summary (lines 90–92)

**Good:** Clean, accurate summary. The "~820× vs XGBoost" claim: 197/0.24 = 820.8× ≈ 820×. ✓

---

### Citations in this section

- `@yeh_comparisons_2009` — correctly cited for the UCI dataset. ✓
- `@eba_gl_2017_16`, `@iasb_ifrs9_2014`, `@basel_committee_basel_2011` — correctly cited for regulatory context. ✓
- `@niculescu_mizil_predicting_2005` — correctly cited for calibration issues with classical models. ✓
- `@platt_probabilistic_1999` — correctly cited for Platt scaling. ✓
- `@federal_reserve_sr11_7` — **see note above** — SR 11-7 is US-specific; in a UK dissertation consider adding PRA SS3/18.
- `@hensman_gaussian_2013` — **misleading citation** for the retrain-on-full-data protocol. ✓

---

## 4.7 Empirical Summary — `sections/4-modeling/3-conclusion.qmd`

**Good:** Concise, accurate. Correctly defers detailed analysis to @sec-conclusion. ✓

---

## 5. Conclusion — `sections/5-conclusion/conclusion.qmd`

### Opening (lines 1–3)

**Good:** Well-framed central question.

**Grammar:** Line 3: "This dissertation set out to answer **if** Deep Gaussian Processes improve..." — in academic writing, **"whether"** is preferred over "if" for indirect questions. Change to "...to answer **whether** Deep Gaussian Processes improve...".

---

### Key Findings (lines 5–12)

**Good:** Findings are accurate and well-supported by the data.

**Errors:**
1. **3.6× cost figure conflates datasets**: Line 12 states "the extra layer adds approximately **3.6×** training time". 3.6× is from the credit card dataset (706.5/197.0 = 3.59×). For Kin8nm, the ratio is 152.4/36.2 = **4.21×**, and the appendix explicitly states "approximately 4× the cost". The conclusion should say either "≈3.6–4×" or qualify "on the credit card task" and "≈4× on Kin8nm". Using "3.6×" without qualification is inconsistent with the appendix.

2. Line 7 contains "DSVGP-2L reduced RMSE by 10.7% over SVGP (0.0699 vs 0.0783)" — verify: (0.0783 - 0.0699)/0.0783 = 0.0084/0.0783 = 10.7%. ✓

**Content Review:** The GP calibration finding ("4–22× improvement") is correct: min improvement = 0.087/0.021 = 4.14×, max = 0.225/0.010 = 22.5×. ✓

---

### Limitations (lines 13–15)

**Good:** Honest and accurate. CPU-only computation, two datasets, fixed architecture are all real limitations.

**Content Review:** "variable-width or residual DGP architectures [@salimbeni_doubly_2017] were not explored" — Salimbeni (2017) does discuss variable-width in their experimental section; this citation is appropriate. ✓

---

### Future Directions (lines 17–21)

**Good:** All suggestions are grounded and correctly cited.

**Grammar:** Line 3 (opening): "...to answer if..." — **"whether"** (see above).

---

### Citations

- `@damianou_deep_2013` — correctly cited for DGP non-stationarity prediction. ✓
- `@duvenaud_avoiding_2014` — correctly cited for small-data collapse. ✓

---

## 6. Appendix — `sections/99-appendix/0-gps-appendix.qmd`

### Design Matrix Construction (lines 3–10)

**Good:** Correct and clear. ✓

---

### Completing the Square (lines 12–52)

**Good:** Derivation is correct. All steps verified. ✓

**Errors:** None.

---

### Predictive Mean and Variance (lines 54–79)

**Good:** Correct application of linear transformation of Gaussian. ✓

**Errors:** None.

---

### Covariance Blocks (lines 81–107)

**Good:** Correct derivation of the three covariance blocks.

**Errors:** None.

**Content Review:** This appendix section derives the covariance blocks in the **weight-space** view (using $\Phi$ and $\Sigma_p$). It is referenced from @sec-gp-regression in the **function-space** view. The two are equivalent because $K = \Phi\Sigma_p\Phi^\top$ and $\vk_* = \Phi\Sigma_p\vphi(\vx_*)$, but this bridging equation is not explicitly stated. A one-sentence note ("noting that $K = \Phi\Sigma_p\Phi^\top$ and $\vk_* = \Phi\Sigma_p\vphi(\vx_*)$") would clarify why weight-space and function-space formulas match.

---

### Code Snippets (lines 108–401)

**Good:** The code snippets in the appendix are well-selected and correctly tied to the claims in the text.

**Errors:** None.

**Methodology — multi-tier split protocol (lines 112–135):**
Minor note: The `run_tier` function in kin8nm creates a new preprocessor call per tier. Since the same `preprocessor` object is reused across tiers in `run.py`, the scaler is **refitted** for each tier (each tier's `X_train` is different). This means the test set (`X_test`) is transformed with a slightly different scaler at each tier. In practice the differences are tiny (all tiers are subsets of the same standardised pool), but technically the test set is not identically preprocessed across tiers. The text states "all models across sizes are evaluated on the same test set" — this is true for the **raw** test labels, but the feature vectors seen by models differ marginally across tiers. This is standard ML benchmarking practice and not a data leakage issue, but the statement could be more precise.

**Methodology — DGP variance prediction (lines 342–365):**
The DSVGP regression prediction code computes:
```python
var = preds.variance.mean(dim=0).cpu().numpy()
```
This averages the conditional variance $\E[\text{Var}[y \mid f^{(L,s)}]]$ over MC samples — only the **expected conditional variance**, not the total predictive variance. By the **law of total variance**:
$$\text{Var}[y] = \E[\text{Var}[y \mid f^{(L)}]] + \text{Var}[\E[y \mid f^{(L)}]]$$
The second term (variance of the conditional means across MC samples) is omitted. This leads to **underestimated** predictive variance. The correct computation would add `preds.mean.var(dim=0)`. However, with 30 MC samples (`num_samples=30` for prediction), the variance-of-means term may be dominated by the noise variance for typical datasets, and the practical impact on NLPD may be small. This is worth **investigating and either correcting or justifying**.

**Methodology — threshold re-computation (lines 382–400):**
The threshold search scans $[0.01, 0.99]$ at $\Delta t = 0.001$. This is appropriately fine-grained. ✓

---

### Two Tables in Appendix — Discrepancy Alert

#### F1-Optimal Threshold Table (`tbl-app-uci-threshold-optima`, lines 458–467)

**Errors — SIGNIFICANT:** The F1, Precision, and Recall values in this table **do not match the CSV data** for several models:

| Model | CSV f1 | Table f1 | CSV prec | Table prec | CSV recall | Table recall |
|-------|--------|----------|----------|------------|------------|--------------|
| SVGP | 0.5465 | 0.546 ✓ | 0.5505 | **0.541** ✗ | 0.5426 | **0.550** ✗ |
| DSVGP-2L | 0.5403 | **0.541** ✗ | 0.5330 | 0.534 ✓ | 0.5479 | **0.549** ✓ |
| DSVGP-3L | 0.5444 | **0.541** ✗ | 0.5244 | **0.515** ✗ | 0.5659 | **0.569** ✓ |

These discrepancies (up to 0.010 in F1, 0.009 in precision) indicate this table was generated from a **different analysis run** than the current CSV results. The precision and recall values for SVGP are reversed (0.541 vs 0.550) compared to the CSV. This table should be regenerated from the current results.

#### Bootstrap CI Table (`tbl-app-uci-ci`, lines 471–480)

**Errors — MODERATE:**

| Model | Main table ECE | CI table ECE | Discrepancy |
|-------|---------------|-------------|-------------|
| SVGP | 0.010 | 0.015 | 0.005 |
| DSVGP-2L | 0.014 | 0.017 | 0.003 |
| DSVGP-3L | 0.021 | 0.016 | **0.005 (reversed!)** |

**Root cause:** The `bootstrap_ci.py` script computes ECE using its own `ece()` function (10 bins, half-open intervals `[lo, hi)`), while the main pipeline's `evaluate_predictions()` likely uses a different ECE implementation (possibly different bin count, different boundary handling, or using calibration error computed at the F1-optimal threshold rather than at the full probability range). The `bootstrap_ci.py` ECE is the **point estimate from `bootstrap_ci.py`'s own formula**, not the same formula used in the main results table. **Both tables are internally self-consistent**, but the reader has no way of knowing the values are computed with different ECE implementations. This must be explicitly noted in the text — ideally both tables should use the same ECE implementation, or the discrepancy should be documented.

The DSVGP-3L reversal (0.021 main vs 0.016 CI) is particularly confusing since the CI table value is lower than the main table value while the CI includes [0.011, 0.025] — which does contain 0.021. The discrepancy is explained by different ECE functions.

---

### Runtime Appendix Table (`tbl-app-kin8nm-runtime`)

**Good:** All values verified against CSV. ✓

**Errors:** Caption says "approximately 4× the cost of DSVGP-2L at all scales" which is consistent with actual values (4.00–4.21×). ✓ However, the conclusion section says "3.6×" — see note above.

---

### NLPD Table (`tbl-app-kin8nm-nlpd`)

**Good:** All values verified against CSV. ✓

---

## Code Review

---

## `code/shared/pipeline.py`

### Two-stage training protocol

**Correctness:** The two-stage protocol (Stage 1: train on X_train, find threshold; Stage 2: retrain on X_full, evaluate on X_test) is implemented correctly. ✓

**Data leakage:** No leakage — preprocessing happens before this pipeline receives the data, and the pipeline receives already-preprocessed splits. ✓

**Data leakage for XGBoost:** XGBoost early-stopping rounds are determined from Stage 1 (X_train + X_val), then the model is retrained on X_full for that many rounds **without** early stopping. This is correct and avoids leakage of test labels. ✓

**Data leakage — threshold selection:** Thresholds are selected on X_val using Stage 1 models (trained on X_train only). The final test metrics use these val-set thresholds. No test set leakage. ✓

**Consistency with dissertation:** The pipeline implements exactly what the text describes in @sec-credit-card and @sec-uci-models. ✓

---

## `code/shared/regression_pipeline.py`

### XGBoost training inconsistency

**Correctness:** For classical models, the code trains Ridge and RF on `X_full` (lines 113), but XGBoost with `m.fit(X_train, y_train, X_val, y_val)` (line 110-111) — training on `X_train` only with early stopping on `X_val`, **not retrained on X_full**. This is **inconsistent** with the classification pipeline where XGBoost IS retrained on X_full.

This means the regression XGBoost is evaluated at a disadvantage vs Ridge/RF (which train on 15% more data). The text does not disclose this inconsistency. The practical impact is small (XGBoost still achieves best classical RMSE) but it should be noted.

### NLPD implementation

**Correctness:** The NLPD function is mathematically correct: $-\log \mathcal{N}(y; \mu, \sigma^2) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$. ✓ Numerical floor `max(y_var, 1e-6)` is appropriate. ✓

### DSVGP variance prediction

**Potential issue:** In the appendix code, the DSVGP regression prediction averages `preds.variance.mean(dim=0)` over MC samples — this is the **expected conditional variance**, missing the **variance-of-means** term (law of total variance). This could lead to underestimated predictive uncertainty, which would improve NLPD artificially (or hurt it, depending on the direction of the gap). **Investigate whether this affects NLPD values meaningfully.**

---

## `code/shared/bootstrap_ci.py`

### ECE implementation

**Issue:** This module implements its own `ece()` function with 10 bins and half-open `[lo, hi)` intervals. This differs from the ECE used in the main pipeline's `evaluate_predictions()` function, explaining the discrepancies between the main results table and the bootstrap CI table. The two implementations should either be harmonised or the discrepancy explicitly documented in the dissertation.

**Correctness:** The ECE formula used here is correct (weighted average of |accuracy - confidence| per bin, weighted by bin fraction). ✓ The bootstrap resampling with replacement is standard. ✓ The 95% CI uses percentile method. ✓

---

## `code/kin8nm/src/run.py`

### Preprocessing scope

**Data leakage:** Scaler is fit on `X_train` only and applied via `transform()` to `X_val` and `X_test`. ✓ No leakage.

**Minor issue:** The `DataPreprocessor` object is shared across tiers — each tier refits the scaler on its tier's `X_train`. This means `X_test` is preprocessed with a slightly different scaler at each tier (the tier's own $\mu_\text{train}, \sigma_\text{train}$). In practice the differences are negligible (all tiers sample from the same standardised pool), and this is standard ML practice. Technically, the test set features differ slightly across tiers. The claim "all models across sizes are evaluated on the same test set" should more precisely say "the same raw test labels". Minor.

---

## Cross-cutting Issues

### Issue 1: "3.6×" vs "4×" runtime ratio

- `tbl-app-kin8nm-runtime` caption says "approximately 4× the cost of DSVGP-2L" for Kin8nm (actual: 4.21×). ✓
- Main text at @sec-uci-cost says "approximately 3.6× the cost" for credit card (actual: 3.59×). ✓
- **Conclusion** (@sec-conclusion line 12) says "adds approximately 3.6× training time" without specifying which dataset.
- **Fix:** State "approximately 3.6× on the credit card task and 4× on Kin8nm" in the conclusion.

### Issue 2: Two ECE implementations

- Pipeline uses one ECE formula → main results tables
- `bootstrap_ci.py` uses a different ECE formula → CI table
- **Fix:** Harmonise both to the same implementation, or add a footnote noting the discrepancy.

### Issue 3: Appendix threshold table values stale

- `tbl-app-uci-threshold-optima` F1, precision, recall values don't match current CSV results
- **Fix:** Regenerate this table from the current predictions.csv.

### Issue 4: DSVGP variance prediction formula

- `preds.variance.mean(dim=0)` omits the variance-of-means term
- **Investigate:** Does adding `preds.mean.var(dim=0)` significantly change NLPD values?

### Issue 5: Figure caption error

- `fig-gp-regression`: caption says "five observations" but code has 7 data points
- **Fix:** Change "five observations" to "seven observations" in the caption.

### Issue 6: TDGP notation error

- @eq-tdgp-layer: `X diag(g^{(ℓ)}(X))` is incorrect notation
- **Fix:** Replace with `X ⊙ g^{(ℓ)}(X)` (element-wise product)

---

## Summary of All Issues by Priority

### Must Fix (errors/factual mistakes)
1. **Fig-gp-regression caption**: "five observations" → "seven observations" (`2-gaussian-processes.qmd`, fig-gp-regression caption)
2. **TDGP notation**: `X diag(g^{(ℓ)}(X))` → `X ⊙ g^{(ℓ)}(X)` (`1-thin-deep.qmd`, eq-tdgp-layer, line 16)
3. **Appendix threshold table** (`tbl-app-uci-threshold-optima`): F1/Precision/Recall values don't match current CSV — regenerate from `predictions.csv`
4. **Conclusion 3.6× claim**: Specify dataset or use "3.6–4×" (`conclusion.qmd`, line 12)

### Should Fix (methodological/citation issues)
5. **ECE discrepancy** (main table 0.010 vs CI table 0.015 for SVGP): Add a note explaining two different ECE implementations are used, or harmonise them
6. **Citation @hensman_gaussian_2013** for retrain-on-full-data protocol: misleading, remove or replace (`1-uci-credit-card-default.qmd`, line 16)
7. **Citation @pmlr-v117-pravesh20a** for "GP expressiveness limited by kernel": verify correctness or replace with `@rasmussen_gaussian_2008` / `@damianou_deep_2013` (`intro.qmd`, line 6)
8. **SR 11-7 citation**: US Federal Reserve regulation cited in a UK dissertation; add UK PRA SS3/18 equivalent (`1-uci-credit-card-default.qmd`, line 41)
9. **DSVGP variance underestimation**: Investigate whether `preds.variance.mean(dim=0)` omits the variance-of-means term and whether this materially affects NLPD (`regression_pipeline.py` / appendix code)
10. **Inducing point asymmetry** (512 for SVGP vs 128 for DSVGP): Acknowledge in `0-overview.qmd`

### Nice to Have (style/minor grammar)
11. `intro.qmd` line 3: "provides an exploration of" → "explores"
12. `0-gp-intro.qmd` line 9: Parametric vs. non-parametric framing is more accurate than frequentist vs. Bayesian
13. `1-var-inf.qmd` line 3: Incomplete sentence ending in colon — add following clause or restructure
14. `01-gp-training.qmd` line 20: Comma splice before "however" — use semicolon
15. `1-weight-space.qmd` line 25: "so-called" → "known as"
16. `2-sparse-var-inf.qmd` line 5: "If we consider inducing inputs" → "Consider inducing inputs"
17. `1-uci-credit-card-default.qmd` line 9: Missing comma: "unit variance fit on" → "unit variance, fit on"
18. `conclusion.qmd` line 3: "to answer if" → "to answer whether"
19. `0-overview.qmd` line 11: "based roughly on established values" → "based on established defaults"
20. `2-dgps/2-intractbility.qmd` line 11: "unless the kernels are degenerate" is vague — specify "linear kernel" or "degenerate linear case"
21. `1-thin-deep.qmd` line 32: "diagonal-plus-rank-one matrices" — more accurately "diagonal-plus-rank-$d$ matrices" for $d$-dimensional inputs
22. `regression_pipeline.py`: XGBoost trains on `X_train` only for regression (no retrain on `X_full`), inconsistent with classification pipeline — document or fix
