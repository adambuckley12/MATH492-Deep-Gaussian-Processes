# Dissertation Full Review: Deep Gaussian Processes

## Context

This is a complete line-by-line review of Adam Buckley's dissertation "Deep Gaussian Processes" submitted at Lancaster University. The review covers: what is good, what is bad/weak, what is missing, what is irrelevant, flaws, appendix analysis, and a full reference audit.

---

## FILES REVIEWED

| File | Content |
|------|---------|
| `index.qmd` | Title, abstract, document structure |
| `sections/0-intro/intro.qmd` | Introduction |
| `sections/1-gps/0-gp-intro.qmd` | Towards GPs (overview) |
| `sections/1-gps/1-weight-space.qmd` | Weight space view + Bayesian LR |
| `sections/1-gps/2-gaussian-processes.qmd` | GP definition, kernels, regression, training, prediction |
| `sections/2-dgps/0-dgps.qmd` | DGP motivation |
| `sections/2-dgps/1-dgp-def.qmd` | DGP definition and formulation |
| `sections/2-dgps/2-intractbility.qmd` | Intractability of exact inference |
| `sections/3-training/0-training.qmd` | Learning overview |
| `sections/3-training/01-gp-training.qmd` | GP hyperparameter learning |
| `sections/3-training/1-var-inf.qmd` | Variational inference |
| `sections/3-training/2-sparse-var-inf.qmd` | Sparse variational GPs |
| `sections/3-training/3-dgp-spars-inf.qmd` | Variational inference for DGPs |
| `sections/3-training/4-doubly-stochastic.qmd` | Doubly stochastic VI |
| `sections/4-modeling/0-overview.qmd` | Experimental overview |
| `sections/4-modeling/1-kin8nm.qmd` | Kin8nm regression |
| `sections/4-modeling/1-uci-credit-card-default.qmd` | Credit card classification |
| `sections/4-modeling/3-conclusion.qmd` | Empirical summary |
| `sections/5-conclusion/conclusion.qmd` | Conclusion |
| `sections/99-appendix/0-gps-appendix.qmd` | Appendix |
| `references.bib` | Full bibliography (873 lines, ~87 entries) |

---

## SECTION-BY-SECTION REVIEW

---

### ABSTRACT (`index.qmd`, lines 12–17)

**Good:**
- Clear and well-written.
- Correctly summarises the 10.7% RMSE improvement and 4–22× calibration advantage.
- Honest about limitations (3-layer failure at small scale).

**Issues:**
- Says "ROC-AUC within ±0.003" of XGBoost — but the main results table shows the gap is 0.001–0.003, which is within ±0.003, so this is fine.
- "calibrated uncertainty estimates" is slightly vague — could clarify as "probability calibration (ECE)".

---

### INTRODUCTION (`sections/0-intro/intro.qmd`)

**Good:**
- Sets up the GP → DGP motivation clearly in just 15 lines.
- Correctly identifies the tractability problem.
- Finance context is established.

**Flaws/Bad:**
1. **Repetition:** Line 10 says "We begin by deriving standard GP regression from first principles." Line 15 then says "We begin by deriving the theory of standard GPs." — this is the third iteration of the same statement (lines 10, 12–13, and 15 all describe the dissertation structure). The last sentence is redundant and should be deleted.
2. **Missing research question:** The conclusion frames the dissertation around the question "Do Deep Gaussian Processes improve on single-layer GPs for real-world machine learning tasks, and if so, when and at what cost?" — but this question is **never stated in the introduction**. This is a significant structural omission for a dissertation.
3. **Misleading dataset framing:** The introduction says "we apply DGPs and GPs to real-world datasets within finance, focusing on credit-card default prediction" — but the dissertation also uses Kin8nm (robot kinematics), which is not mentioned in the intro. A reader would not expect a robot kinematics experiment.
4. **Unusual citation for key claim:** Line 6 cites `[@pmlr-v117-pravesh20a]` (Kothari & Livni 2020, about kernel expressive power via association schemes) to support "the expressiveness of a standard GP is limited by its kernel." This is a very indirect citation. The natural references here are Rasmussen & Williams (2008) directly or Duvenaud et al. (2014). Kothari & Livni is an algorithmic learning theory paper.

---

### SECTION 1: TOWARDS GAUSSIAN PROCESSES (`sections/1-gps/0-gp-intro.qmd`)

**Good:**
- Clear motivation from supervised learning to non-parametric inference.
- Good citations.

**Issues:**
- Line 9: "In contrast to frequentist approaches that specify a parametric model and estimate its parameters via maximum likelihood estimation (MLE)... GPs place a prior directly on an infinite-dimensional function space." — This is slightly misleading: GPs themselves use type-II MLE (empirical Bayes) for hyperparameter estimation, so the contrast with MLE is not sharp. Better framing: "In contrast to parametric models that fix the function family in advance, GPs..."

---

### SECTION 1.1: THE WEIGHT SPACE VIEW (`sections/1-gps/1-weight-space.qmd`)

**Good:**
- Derivation is clear, well-structured, and pedagogically effective.
- The kernel trick derivation via matrix inversion lemma is nicely handled.
- Function-space perspective transition is intuitive.
- Good appendix references.

**FLAW — Mathematical inconsistency (serious):**
- **eq-predictive-mean-var (line 84):** States the predictive variance for `f_* = f(x_*)` (the latent function) includes `+σ_n²`:
  ```
  Var[f_* | x_*, X, y] = φ(x_*)ᵀ Σ_w φ(x_*) + σ_n²
  ```
  This is **incorrect for the latent function** f_*. The `σ_n²` term only applies when predicting the noisy observation `y_*`. The Appendix derivation at `sec-predictive-variance` correctly gives: `Var[f_*] = φ(x_*)ᵀ Σ_w φ(x_*)` (no σ_n²). This is a contradiction within the dissertation. Similarly, `eq-predictive-kernel` (line 130) carries this error forward. The GP regression section correctly distinguishes by labelling eq-gp-predictive as for `y_*`, but this earlier equation mislabels f_*.

  **Impact:** The final result in `eq-predictive-kernel-full` includes σ_n² for f_* which is then said to match the GP regression result for `y_*` — these are different things. The weight-space section should either predict `y_*` (and keep σ_n²) or predict `f_*` (and drop σ_n²), but should not conflate them.

---

### SECTION 1.2: GAUSSIAN PROCESSES (`sections/1-gps/2-gaussian-processes.qmd`)

**Good:**
- GP prior, kernels, hyperparameter estimation, regression, training, and prediction are all covered rigorously.
- Figure descriptions are clear.
- The variance decomposition into epistemic/aleatoric is well-handled.
- `@fig-gp-regression` and `@fig-gp-training` are both useful visualizations.

**Issues:**
- Line 374–376: Trailing blank lines (minor formatting issue).
- Line 379: "making predictions at test time efficient, and the main computational challenge is purely the training phase" — grammatically awkward; should be "making predictions at test time **is** efficient."
- The gradient formula in the training section (line 281 of 2-gaussian-processes.qmd) is presented in the standard form and then re-derived in the training section (01-gp-training.qmd). There is some mild redundancy between the treatment of GP training in section 2 and in section 4.

---

### SECTION 2: DEEP GAUSSIAN PROCESSES

#### `0-dgps.qmd`

**Good:** Short but effective motivation. Good transition from GP limitations to DGPs.

#### `1-dgp-def.qmd`

**Good:**
- Rigorous definition of L-layer DGP.
- Generative model notation is clear.
- Multi-output layer formulation included.
- `@fig-dgp-composition` is a strong visualization.
- Non-stationarity derivation via composed covariance (eq-composed_covariance) is elegant.

**Issues:**
- `@damianou_deep_2013` is cited for the non-stationarity result but also `@dunlop_how_2018` is cited — both are appropriate here.
- Line 41: "The directed graph structure implies that each layer depends only on its immediate predecessor (which is a Markov property)" — parenthetical awkward.

#### `2-intractbility.qmd`

**Good:**
- Clear two-layer worked example showing the non-Gaussian mixture.
- Explicit posterior formula showing numerator/denominator intractability.

**Minor issue:**
- **Filename typo:** `2-intractbility.qmd` should be `2-intractability.qmd` (missing 'a'). Doesn't affect the compiled output but is unprofessional.

---

### SECTION 3: LEARNING IN GP AND DGP MODELS

#### `0-training.qmd`

Fine as an overview. No significant issues.

#### `01-gp-training.qmd`

**Good:**
- Clean re-derivation of LML from Gaussian conjugacy.
- Gradient interpretation is excellent and insightful.

**Minor issue:**
- Line 20: "as datasets grow, the O(n³) scaling often can become a challenge depending on available compute resources" — awkward phrasing. Should be "O(n³) scaling becomes prohibitive for large datasets."

#### `1-var-inf.qmd`

**Good:**
- ELBO derivation from KL decomposition is well-presented.
- KL terms interpreted clearly.

**Issues:**
- The derivation at lines 19–28 restates the ELBO derivation that's implicitly covered earlier — slightly redundant for an expert reader, appropriate for a teaching document.

#### `2-sparse-var-inf.qmd`

**Good:**
- Clean presentation of Titsias (2009) sparse GP.
- Optimal parameter equations included.
- Minibatch ELBO well-explained with scaling factor n/|B|.

**Issues:**
- Line 25 has a formatting gap: "The conditional mean K_fu K_uu^{-1} u is a weighted sum of inducing values The conditional..." — missing period after "inducing values".

#### `3-dgp-spars-inf.qmd`

**Good:**
- Clearly presents the Damianou vs. Salimbeni approaches.
- Mean-field ELBO correctly identified as the Damianou formulation.

**Issues:**
- Citation for variational factorization (eq-dgp_variational_factorization) only cites `@damianou_deep_2013` but this form of the variational distribution also relies on `@titsias_variational_2009`. Both should be cited.
- The paragraph about `q(F^(ℓ) | F^(ℓ-1))` at lines 42–48 being Gaussian when F^(ℓ-1) is fixed but non-Gaussian overall is correct and well-written.

#### `4-doubly-stochastic.qmd`

**Good:**
- Clear presentation of doubly stochastic VI.
- Reparametrization trick well-explained.
- Computational complexity clearly stated: O(|B|m²LS + m³L).
- "Doubly stochastic" term correctly attributed and explained.

**No significant flaws in this section.**

---

### SECTION 4: MODEL EVALUATION

#### `0-overview.qmd`

**Good:**
- Clear experimental motivation.
- Hyperparameter choices explained and justified.
- Honest about not doing hyperparameter search.

#### `1-kin8nm.qmd`

**Good:**
- Excellent hidden representation analysis (`@fig-kin8nm-hidden`) — this is the standout piece of evidence in the dissertation.
- PCA variance explained (13.6% → 47.8%) is compelling.
- Small-scale failure explained with reference to Duvenaud et al.

**FLAW — FACTUAL ERROR (critical):**
- **Line 59 in scaling section:** States "At full scale, DSVGP-2L overtakes SVGP (RMSE 0.0670 vs 0.0749, a 10.7% reduction)." But the large-scale results table (`tbl-kin8nm-large`) clearly shows **DSVGP-2L RMSE = 0.0699 and SVGP RMSE = 0.0783**. The numbers in the text (0.0670 and 0.0749) are different from the table — these appear to be stale numbers from a previous run. The 10.7% reduction is correctly computed from the table values (0.0699 vs 0.0783), but the raw numbers in the text are wrong.

**Other issues:**
- Line 42: "the third layer... only leads to a small improvement in R² (0.917 vs 0.911)" — this compares DSVGP-3L to SVGP, but the context is discussing why DSVGP-3L underperforms DSVGP-2L. The comparison should be DSVGP-3L (0.917) vs. DSVGP-2L (0.929), not vs. SVGP. As written it is misleading.

#### `1-uci-credit-card-default.qmd`

**Good:**
- Regulatory context (Basel IRB, IFRS 9) is well-integrated.
- ECE comparison is the standout finding.
- Three-tier evaluation design is methodologically sound.

**FLAW — Table inconsistencies (serious):**

1. **ROC-AUC inconsistency between main table and bootstrap CI table:**
   - `tbl-uci-large`: SVGP = 0.778, DSVGP-2L = 0.777, DSVGP-3L = 0.776
   - `tbl-app-uci-ci`: SVGP = 0.776, DSVGP-2L = 0.778, DSVGP-3L = 0.777
   - These are **different values** for the same experiment. The tables appear to be from different runs. SVGP and DSVGP-2L values are effectively swapped.

2. **F1/threshold inconsistency between main table and threshold table:**
   - `tbl-uci-large`: Logistic Regression F1 = 0.500, t* = 0.50; XGBoost F1 = 0.536, t* = 0.50; RF F1 = 0.532, t* = 0.35
   - `tbl-app-uci-threshold-optima`: Logistic Regression t* = 0.55, F1 = 0.513; XGBoost t* = 0.55, F1 = 0.543; RF t* = 0.40, F1 = 0.538
   - These are systematically different — same metrics, different values across the two tables.

**Other issues:**
- Line 69: DSVGP-3L at small scale has ECE = 0.005 (best of all models) while also having ROC-AUC = 0.523 (near-random). This is counterintuitive and not explained. A near-random classifier outputting probabilities near the base rate (22.1%) could appear well-calibrated by ECE even while being useless for discrimination. This should be noted.
- The claim "GP-based methods have been applied to credit-risk problems... but their native calibration advantage... has not been directly quantified at scale" — this is plausible but not explicitly verified. The cited papers (Mahajan 2022, Huang 2011) don't specifically show calibration comparisons at scale.

#### `3-conclusion.qmd` (Empirical Summary)

Fine — brief and accurate. No issues.

---

### SECTION 5: CONCLUSION (`sections/5-conclusion/conclusion.qmd`)

**Good:**
- Addresses the central question directly and explicitly.
- Key findings are well-summarized.
- Limitations are honest and specific.
- Future directions are meaningful (stochastic-depth DGPs, amortized VI, regulatory backtesting, deep kernel learning).

**Issues:**
- "DSVGP-2L reduced RMSE by 10.7% over SVGP (0.0699 vs 0.0783)" — correct numbers used here (unlike the scaling section which uses 0.0670/0.0749).
- "three-layer DGPs fail to outperform two-layer variants and collapse at small data scales, suggesting that **on these tasks**, additional depth beyond two layers offers diminishing returns" — "on these tasks" qualifier is appropriately cautious.
- Future directions cite `@duvenaud_avoiding_2014` for "stochastic-depth DGPs" — but that paper is about avoiding pathologies in deep networks, not specifically stochastic-depth DGPs. The connection is implied but not exact.

---

## APPENDIX REVIEW

### Mathematical Derivations (sec-design-matrix, sec-completing-the-square, sec-predictive-variance, sec-covariance-blocks)

**GOOD — All four derivations are correct and appropriately detailed.**
- The completing-the-square derivation correctly does NOT include σ_n² in the predictive variance (consistent with correct Bayesian LR).
- The covariance blocks derivation is clean.

**Note:** The appendix predictive variance (no σ_n²) contradicts the weight-space main text (which includes σ_n²) — reinforcing the flaw identified above.

### Code Snippets

#### Should stay:
- Multi-tier split protocol — directly supports reproducibility claims.
- Pre-processing pipeline (`DataCleaner`, `DataPreprocessor`) — necessary to verify no data leakage.
- Regression extensions (NLPD, SVGP predict, DSVGP predict) — specific enough to be useful.
- Optimal threshold re-computation — important for the F1 metric.

#### Questionable (could be removed or shortened):
- **SVGP training loop** — this is essentially standard GPyTorch boilerplate (Adam optimizer, VariationalELBO, minibatch loop). Adds little beyond what's described in the text.
- **DSVGP training loop** — more specific (shows `num_likelihood_samples` context), worth keeping briefly.
- **Post-hoc calibration utility** — explicitly stated as "not used in the default pipeline." This code serves no purpose in the dissertation other than "completeness." It should be removed or relocated to a footnote. Including code you didn't run is methodologically confusing.

### Supplementary Figures

#### UCI Credit Card (sec-appendix-uci-scaling):
- Small tier: ROC/PR, reliability diagrams, confusion matrices, threshold curves — all appropriate.
- Medium tier: Same — all appropriate.
- Large tier runtime vs. calibration plot — useful.

**Note:** Confusion matrices at small scale for all 6 models may be slightly excessive detail, but they're reasonable.

#### Kin8nm (sec-appendix-kin8nm-scaling):
- Predicted vs. actual, residuals, metrics bar chart, runtime — all appropriate for small/medium tiers.

**No irrelevant figures identified in the appendix.** All figures directly supplement main text results.

### Supplementary Tables

- F1 threshold summary — appropriate, but has inconsistencies with main text (noted above).
- Bootstrap CI table — appropriate, but has inconsistencies with main table (noted above).
- NLPD across tiers — appropriate, positive NLPD at small scale explained.
- Runtime across tiers — appropriate.

---

## REFERENCE AUDIT

### Duplicate entries:
1. `salimbeni_doubly_2017` and `salimbeni_doubly_2017-1` — **EXACT DUPLICATE** (same DOI: 10.48550/arXiv.1705.08933, same paper). Remove `-1` variant.
2. `noauthor_standard_nodate` and `xu_standard_2025` — Same paper "Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization." The `noauthor_standard_nodate` is an incomplete HTML snapshot of the same arXiv paper. Remove `noauthor_standard_nodate`.

### Incorrect metadata:
1. **`gelman_bayesian_2025`** — year = {2025} is the date of a PDF typo correction. The 3rd edition of BDA was published in **2013**. Year should be 2013.
2. **`dunlop_how_2018`** — Missing journal, volume, and pages. This paper was published in Journal of Machine Learning Research, vol. 19, 2018. Entry needs: `journal = {Journal of Machine Learning Research}`, `volume = {19}`, `pages = {1--46}`.
3. **`paciorek_nonstationary_nodate`** — Missing year, journal, and institution. This appears to be a PhD thesis (CMU, 2003). Needs full thesis metadata.
4. **`ober_promises_nodate`** — Missing year and venue. This was published at UAI 2021. Needs: `year = {2021}`, `booktitle = {Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence}`.
5. **`noauthor_gaussian_nodate`** — Web page with no author. Should use: `author = {{PyMC Development Team}}` or equivalent.
6. **`murphy_machine_2013`** — Key says 2013 but year field is 2012. The book was first published August 2012. Key is misleading; year field 2012 is correct.

### References that appear in bibliography but are NEVER cited in any file:

These entries exist only in `references.bib` and are not referenced in any `.qmd` file:

| Citation key | Description | Verdict |
|---|---|---|
| `jidling_linearly_2017` | Linearly constrained GPs | Irrelevant to this dissertation topic |
| `van_der_vaart_rates_2008` | Rates of contraction of posterior distributions | Too theoretical, not cited |
| `anderson_introduction_2003` | Introduction to multivariate statistical analysis | Not cited |
| `eaton_multivariate_2007` | Multivariate statistics: a vector space approach | Not cited |
| `emzir_hilbert-space_2019` | Hilbert-space reduced-rank methods for DGPs | Not cited — clearly from early research phase |
| `noauthor_gaussian_nodate` | PyMC HSGP webpage | Not cited — irrelevant |
| `souza_thin_2023` | Thin and Deep GPs (TDGP) | Not cited, despite being a strong related work candidate |
| `calandra_manifold_2016` | Manifold GPs | Not cited |
| `wilson_deep_2015` | Deep Kernel Learning | Mentioned in future work but NOT cited in text |
| `ober_promises_nodate` | Promises and Pitfalls of DKL | Not cited |
| `xu_standard_2025` | Standard GP for high-dim BO | Not cited |
| `noauthor_standard_nodate` | Duplicate HTML of xu_standard_2025 | Remove |
| `mattos_deep_2017` | Deep recurrent GPs | Not cited |
| `jin_inverse_2017` | IRL via DGP | Not cited |
| `mattos_latent_2016` | Latent autoregressive GPs | Not cited |
| `dai_variational_2016` | Variational auto-encoded DGPs | Not cited |
| `mattos_recurrent_2016` | Recurrent GPs | Not cited |
| `salimbeni_doubly_2017-1` | DUPLICATE of salimbeni_doubly_2017 | Remove |
| `amari_natural_1998` | Natural gradient learning | Not cited |
| `ranganath_black_2014` | Black box variational inference | Not cited |
| `rahimi_random_2007` | Random features for kernel machines | Not cited |
| `salimbeni_natural_2018` | Natural gradients in practice for GPs | Not cited |
| `lehdili_performance-enhancing_2025` | GP for market risk calculation | Not cited |
| `yang_gaussian_2015` | GP-based algorithmic trading | Not cited |
| `munoz-gonzalez_laplace_2014` | Laplace approx with GPs for volatility | Not cited |
| `rudin_stop_2019` | Stop explaining black box ML | Not cited |
| `goodman_european_2017` | EU regulations on algorithmic decision-making | Not cited |
| `box_bayesian_1992` | Bayesian inference in statistical analysis | Not cited |
| `zadrozny_transforming_2002` | Transforming classifier scores to probabilities | Not cited (appendix mentions calibration but doesn't cite this) |
| `paciorek_nonstationary_nodate` | Nonstationary GPs for regression | Not cited |
| `kumar_deep_2018` | DGPs with convolutional kernels | Not cited |
| `cutajar_deep_2019` | DGPs for multi-fidelity modeling | Not cited |
| `bui_deep_2016` | DGPs using approximate EP | Not cited |

**Total unused references: ~31 entries out of ~87** (~36% of the bibliography is unused).

**Verdict:** The bibliography was clearly accumulated during research/reading but many entries were never incorporated into the text. This is not inherently wrong (researchers read more than they cite), but several of these represent important related work (TDGP, manifold GPs, DKL, DGPs via EP) that would belong in a related work section if one existed.

---

## WHAT IS MISSING

1. **Research question in introduction** — The central question is only revealed in the conclusion. It should be stated explicitly in the introduction.

2. **Related work section** — A dissertation normally situates its contribution relative to prior work. The dissertation is almost entirely tutorial/expository (reproducing GP/DGP theory) and experimental. No structured discussion of alternative DGP approaches: TDGP (Souza et al. 2023), manifold GPs (Calandra 2016), DGPs via EP (Bui et al. 2016), recurrent GPs (Mattos et al. 2016). These are cited in the bibliography but never discussed.

3. **Explicit mention of Kin8nm in introduction** — The introduction says the work focuses on "credit-card default prediction" but doesn't mention robot kinematics.

4. **Explanation for counterintuitive ECE result** — DSVGP-3L having ECE = 0.005 (best) while having near-random discrimination (ROC-AUC = 0.523) at small scale is unexplained and potentially misleading.

5. **Confidence intervals for Kin8nm regression** — Bootstrap CIs are provided for the UCI experiment but not for Kin8nm RMSE values.

6. **Statement of limitations for the Kin8nm hidden representation claim** — The claim that the hidden layer "learns a smooth, non-linear reparametrisation" is visually compelling but not quantified beyond PCA variance explained.

---

## WHAT IS IRRELEVANT

1. **Post-hoc calibration code in appendix** — Code that is "not used in the default pipeline" adds bulk without contributing to the dissertation's findings.

2. **SVGP training loop code** — Standard GPyTorch boilerplate that adds no dissertation-specific content.

3. **~31 unreferenced bibliography entries** — Should be cleaned up.

4. **`salimbeni_doubly_2017-1` duplicate** — Should be removed.

5. **`noauthor_standard_nodate`** — Should be removed (duplicate/web version of xu_standard_2025).

---

## PRIORITIZED ISSUE LIST

### CRITICAL (factual errors / mathematical flaws):
1. **RMSE numbers wrong in Kin8nm scaling section** (`1-kin8nm.qmd` line 59): Text says "0.0670 vs 0.0749" but table shows "0.0699 vs 0.0783."
2. **σ_n² included for f_* in weight-space predictive** (`1-weight-space.qmd` eq-predictive-mean-var, eq-predictive-kernel): Contradicts appendix derivation; should either drop σ_n² or change to predicting y_*.
3. **Inconsistency between tbl-uci-large and tbl-app-uci-ci** (ROC-AUC values differ for same models).
4. **Inconsistency between tbl-uci-large and tbl-app-uci-threshold-optima** (F1 and t* values differ).

### SIGNIFICANT:
5. **Missing research question in introduction.**
6. **Introduction doesn't mention Kin8nm dataset.**
7. **Duplicate bibliography entry** (`salimbeni_doubly_2017-1`).
8. **`gelman_bayesian_2025` year should be 2013.**
9. **`dunlop_how_2018` missing journal/volume/pages.**
10. **Kin8nm R² comparison misleading** (compares DSVGP-3L to SVGP not to DSVGP-2L).

### MINOR:
11. **Trailing blank lines** in `2-gaussian-processes.qmd`.
12. **Filename typo:** `2-intractbility.qmd`.
13. **Missing period** in `2-sparse-var-inf.qmd` line 25.
14. **Awkward grammar** in `2-gaussian-processes.qmd` line 379.
15. **Unnecessary `pmlr-v117-pravesh20a` citation** for "GP limited by kernel" claim.
16. **~31 unused bibliography entries** that should be cleaned or incorporated.

---

## IMPLEMENTATION PLAN

The task is to produce a comprehensive review markdown file, not to edit the dissertation files. The review will be written to a new output file at:

`/Users/adam/dev/uni/4/diss_writeup/DISSERTATION_REVIEW.md`

The file will be structured as:
1. Executive summary (ratings by section)
2. Section-by-section analysis (issues, strengths, flaws)
3. Appendix review
4. Reference audit (full table of unused/incorrect references)
5. Prioritized fix list

### Files to create:
- `/Users/adam/dev/uni/4/diss_writeup/DISSERTATION_REVIEW.md`

### Files NOT to modify:
- Any `.qmd` source files (review only, no edits unless user asks)
- `references.bib` (flag issues in review)
