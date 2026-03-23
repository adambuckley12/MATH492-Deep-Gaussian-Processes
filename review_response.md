# Review Response: Claim-by-Claim Analysis

---

## Claim 1: Missing Identity Mean Functions — "Fatal Flaw"

### What the reviewer says
The code uses `ConstantMean` for hidden layers instead of a `LinearMean` (identity/skip connection). Salimbeni & Deisenroth (2017) "mandated" linear mean functions. This omission "guarantees" rank collapse and is the sole cause of the R²=0.002 result.

### What is actually true
**Partially valid, significantly overstated.**

The code does use `ConstantMean` for hidden layers (`MeanClass = ZeroMean if output_dims is None else ConstantMean`). A `ConstantMean` learns a single scalar offset per GP — it does not pass inputs forward, and is not an identity mapping.

The reviewer is correct that linear/identity mean functions in hidden DGP layers are theoretically motivated: they act as residual skip connections and help gradients flow through depth. This is documented in Duvenaud et al. (2014) and discussed in the dissertation itself (see `sections/4-advanced/2-deeply-non-stationary.qmd`, which notes that linear mean functions "prevent pathological collapse").

**However, the reviewer dramatically overclaims:**

1. **GPyTorch's own DGP tutorial uses `ConstantMean` for hidden layers** — this is the library's default approach, not an aberration. Many published DGP experiments use it without catastrophic failure.

2. **The causal claim is unproven.** The reviewer asserts that `ConstantMean` is the *sole* cause of R²=0.002. In reality, small data + deep non-parametric models is a well-known difficulty (Duvenaud 2014 — which the dissertation cites correctly). Both factors plausibly contribute.

3. **"Mandated" is false.** Salimbeni & Deisenroth (2017) introduced DSVI; their specific mean function choices in experiments are an empirical detail, not a mandate. The GP community has used various mean functions.

4. **The claim that failure is "mathematically guaranteed"** is rhetoric, not proof. Many ConstantMean DGPs train fine at larger scales — the dissertation's own 3L results at medium and large Kin8nm scale are reasonable (RMSE 0.0887 at medium).

### Worth addressing?
**Yes, partially.** A one-sentence caveat noting that `LinearMean` hidden layers would have been a theoretically stronger choice, and may have reduced the 3L degradation, is fair to add. But the "fatal flaw" framing is unfair and the conclusion (that all 3L results are invalid) is not warranted.

---

## Claim 2: Hidden Dimension = 5 as Information Bottleneck

### What the reviewer says
`hidden_dim=5` with 23 input features creates a destructive information bottleneck for the UCI dataset. This "entirely explains" why DGPs failed to improve over SVGP on credit risk.

### What is actually true
**Debatable concern, massively overstated causal claim.**

It is true that hidden_dim=5 is a significant compression from 23 features. And it is true that aggressive bottlenecking can destroy discriminative signal.

**However:**

1. **hidden_dim=5 is standard in the DGP literature.** Salimbeni & Deisenroth and subsequent papers use small hidden dimensions (often 1–5) even on multi-dimensional inputs. It is not an obvious error.

2. **The claim that dim=5 "entirely explains" the DGP's lack of improvement is unfounded.** The SVGP also achieves essentially the same ROC-AUC as the DGPs (0.778 vs 0.776–0.777), and SVGP operates in the full input space. If the bottleneck was the issue, SVGP should dominate — but DGPs at large scale match it within noise. The more parsimonious explanation (which the dissertation gives) is that the credit default surface is smooth enough that hierarchical composition adds nothing.

3. **The claim that "GP kernel evaluates them as highly similar" after compression** — this only matters if the compression destroys class-discriminative information AND the compressed space happens to map different-class points to the same location. With learned inducing points and a learnable kernel, the DGP has some capacity to adapt.

4. **The reviewer's proposed fix (hidden_dim ≥ input_dim)** is not established in the DGP literature as a rule.

### Worth addressing?
**Briefly.** A note that hidden_dim is a tunable hyperparameter and that larger values were not explored is honest. But the claim that this "entirely explains" failure is not worth dignifying.

---

## Claim 3: Benchmark Comparison with Salimbeni & Deisenroth (2017)

### What the reviewer says
S&D's 3L achieves the same RMSE as their 2L (0.06), while the dissertation's 3L (0.0759) is worse than 2L (0.0699). This shows the dissertation's 3L failure is a bug, not a fundamental result.

### What is actually true
**The qualitative observation is fair; the comparison methodology is flawed.**

The numbers are not directly comparable. S&D (2017) used:
- Different inducing point counts (likely 100, not 128)
- Different hidden dimensions (not necessarily 5)
- Possibly different data splits (the paper may use a standard 90/10 split rather than 15/20% tiered dev/test)
- Different training epochs and optimizer settings
- Potentially different mean functions (the exact implementation details are not fully reproduced in the paper)

The reviewer presents the table as if it is an apples-to-apples comparison. It is not.

**That said:** the qualitative observation — that properly implemented deep models should not degrade at depth 3 on this benchmark — is valid and worth engaging with. The S&D result showing stable RMSE at 2–5 layers supports the claim that the 3L implementation has suboptimal training dynamics.

### Worth addressing?
**Yes.** Acknowledge that the S&D results suggest a well-tuned 3L should not degrade below 2L, and frame the dissertation's 3L results as a consequence of architectural choices (mean functions, fixed hyperparameters) rather than a fundamental property of depth.

---

## Claim 4: XGBoost Calibration — "Strawman Comparison"

### What the reviewer says
Comparing raw XGBoost ECE (0.196) to native GP ECE (0.010) is methodologically unfair because XGBoost is always calibrated in practice before ECE is evaluated.

### What is actually true
**Valid criticism, but overstated, and the dissertation partially anticipated it.**

The reviewer is correct that:
- Tree-based models structurally produce compressed, non-probabilistic score distributions
- Platt scaling / isotonic regression is standard industry practice before computing ECE
- The 4–22× advantage figure compares optimally-calibrated GPs to uncalibrated XGBoost

**However:**
- The dissertation explicitly acknowledges this: *"XGBoost's ECE of 0.196 suggests miscalibrated estimates that would likely require post-hoc recalibration (such as Platt scaling or isotonic regression)"* — this is not presented as a final state
- The reviewer calls it a "strawman" but the dissertation cites Niculescu-Mizil & Caruana (2005) and references calibration correction methods
- The relevant question for the dissertation's thesis is not just "what ECE after calibration?" but "does the model require additional calibration steps?" — GPs not requiring this step is a practical advantage

**What would strengthen this:** running Platt-scaled XGBoost and reporting its ECE would reduce the reviewer's attack surface significantly.

### Worth addressing?
**Yes — add calibrated XGBoost ECE or qualify the 4–22× claim as "over uncalibrated XGBoost."** The current framing slightly overclaims the magnitude of the advantage.

---

## Claim 5: Basel III / Regulatory Mischaracterization

### What the reviewer says
Claiming that native GP test-set calibration satisfies Basel III / EBA IRB requirements is false. Basel mandates Through-the-Cycle calibration to long-run average default rates (LRA DR), which a model trained on 2005 data cannot provide. Post-hoc calibration to LRA DR is required for all models regardless.

### What is actually true
**Substantially valid.**

The EBA/GL/2017/16 guidelines do require calibration to long-run average default rates covering multiple economic cycles, not point-in-time (PIT) test-set accuracy. A model trained on one year of data from 2005 (the UCI dataset) cannot, by definition, be "calibrated" in the Basel sense — regardless of how well its predictions track the test-set base rate.

The dissertation's claim that *"GP models' native calibration is far more consistent with these standards"* overstates what ECE on a single holdout set demonstrates. Point-in-time test ECE ≠ regulatory TTC calibration.

**What the dissertation actually says** is more measured than the reviewer characterises — it says GPs are "more consistent with" the standards, not that they comply. But the framing is still misleading because it implies the ECE advantage translates to regulatory compliance, when in practice:
1. All models (GP or otherwise) need to be backtested against long-run default rates
2. All models need stress-testing across economic cycles
3. The choice of 2005 UCI data, covering a single economic snapshot, makes regulatory calibration claims unverifiable

### Worth addressing?
**Yes — this is the strongest valid critique.** Soften the regulatory language: note that test-set ECE measures PIT calibration accuracy, which is a necessary but not sufficient condition for Basel IRB compliance. The native calibration advantage is still meaningful (it reduces one downstream calibration task) but should not be presented as satisfying TTC regulatory requirements.

---

## Claim 6: Threshold Optimization / Probabilistic Disconnect

### What the reviewer says
Optimizing a threshold to maximise F1 and then evaluating ECE at that threshold introduces a disconnect between discriminative tuning and probabilistic evaluation.

### What is actually true
**Mostly false — reviewer appears confused about what ECE measures.**

ECE is computed on the *predicted probabilities*, not on binary class predictions. ECE measures how well `P(ŷ = 1 | x)` aligns with observed empirical frequency — it is entirely independent of any classification threshold. The decision threshold affects F1, Accuracy, Precision, Recall — it does not affect ECE or Brier score.

The methodology (validate threshold on validation set, retrain on train+val, evaluate on test) is standard and correct.

### Worth addressing?
**No.** The reviewer is wrong here. ECE is not affected by threshold choice.

---

## Claim 7: Lack of Cross-Validation at Small Scale

### What the reviewer says
K-fold cross-validation is "mandatory" for evaluating models at small data scales. Single-split results at N=425 are unreliable.

### What is actually true
**Partially valid, but "mandatory" is too strong and "mandatory" applies to the reviewer's standard, not the field's.**

A single fixed split at N=425 does have higher variance than k-fold CV, and the R²=0.002 result could have different value under different splits. This is a real limitation.

**However:**
- The test set is large (1,639 / 6,000 observations) and fixed — evaluation variance comes from training instability, not evaluation noise
- K-fold CV is standard for hyperparameter tuning, not always for final evaluation
- "Mandatory" is not accurate — many benchmarked papers use single splits on small tiers

### Worth addressing?
**Briefly.** A sentence noting that small-scale results are subject to sampling variance and that k-fold would provide tighter confidence intervals is honest. The R²=0.002 result itself is consistent across both datasets (UCI 3L also collapses at small scale), which makes it less likely to be a single-split anomaly.

---

## Summary Table

| Claim | Valid? | Severity | Worth addressing? |
|-------|--------|----------|-------------------|
| Missing identity mean functions → fatal flaw | Partially valid, wildly overstated | Medium | Yes — add caveat, don't accept "fatal flaw" framing |
| hidden_dim=5 → entirely explains DGP failure | Weak correlation, unproven causal claim | Low | Briefly note it as a limitation |
| S&D benchmark comparison proves bug | Non-comparable experiments; qualitative point has merit | Medium | Acknowledge S&D's 3L stability |
| XGBoost ECE comparison is strawman | Valid but dissertation partly anticipates it | Medium | Qualify claim or add calibrated XGBoost ECE |
| Basel III compliance claimed from PIT ECE | Valid — regulatory overreach | High | Must soften regulatory language |
| Threshold optimisation corrupts ECE | False — reviewer misunderstands ECE | None | Do not address |
| No cross-validation | Mild legitimate limitation | Low | One-sentence caveat |

---

## Overall Assessment

The review is written in an adversarial register that overstates every limitation into a "fatal flaw" or "catastrophic error." The rhetorical strategy is to pick genuine (if modest) limitations — mean function choice, calibration comparison framing, regulatory language — and inflate them into complete invalidation of the dissertation's conclusions.

**The genuinely valid points are:**
1. Linear mean functions for hidden layers are better motivated and would likely have helped 3L training — worth acknowledging
2. The 4–22× calibration advantage should be qualified as being over *uncalibrated* XGBoost
3. The regulatory language around Basel III is stronger than the evidence (PIT ECE) justifies and should be softened

**The invalid or overstated points are:**
- "Fatal flaw" / "catastrophically false" framing of the mean function choice
- The claim that hidden_dim=5 "entirely explains" DGP underperformance on UCI
- The claim that the S&D benchmark comparison proves a bug (non-comparable setups)
- The threshold/ECE disconnect (reviewer error)
- "Mandatory" cross-validation (too strong)
- The implication that any of this invalidates the core finding that DGPs outperform SVGPs on non-stationary targets (Kin8nm) while not improving on smooth targets (UCI)

The dissertation's central empirical claims — that 2L DGPs outperform SVGPs on Kin8nm (10.7% RMSE reduction), that 3L DGPs fail to improve on 2L at all scales tested, and that GP models show native calibration advantages over uncalibrated classical methods — are not overturned by this review.
