# Causal ML in the email experiment

Reference guide for the causal framework and the **Causal Machine Learning** techniques applied in this repository. It complements [`GUIA_CONCEPTUAL_TECNICA.md`](GUIA_CONCEPTUAL_TECNICA.md) with more methodological depth.

---

## 1. Causal question vs predictive question

| Approach | Question | Example in this project |
|----------|----------|-------------------------|
| **Predictive** | Who will click? | A `ctor` classifier with no counterfactual |
| **Causal** | Did the nudge *cause* more clicks than control? | ATE in an RCT |
| **Causal ML** | For *which profile* does the nudge cause more clicks? | CATE + personalization |

A predictive model may associate "younger age" with more clicks because younger customers already clicked more **without** the nudge. The CATE tries to estimate the **increment attributable to the treatment** conditional on the profile \(X\).

---

## 2. Identification in an RCT

For an effect estimator to have a causal interpretation, assumptions are needed. In a well-executed randomized experiment:

### 2.1 Random assignment (ignorability)

\[
Y(0), Y(1) \perp T \quad \Rightarrow \quad \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0] = \mathbb{E}[Y(1) - Y(0)]
\]

Assignment to `ctrl`, `trat1` or `trat2` is independent of the potential outcomes. That is why the **group difference in means** is a valid ATE estimator without adjusting for covariates.

### 2.2 SUTVA (Stable Unit Treatment Value Assumption)

One customer's outcome does not depend on the treatment assigned to another customer. For mass emails this is reasonable: each customer receives one variant; there is no "spillover" between units.

### 2.3 Positivity

Every customer has a positive probability of being in each arm. With balanced randomization (~33% per group), this holds.

### 2.4 Potential outcomes with three arms

Here there are **three** treatments, not one. For each customer \(i\):

\[
Y_i(\text{ctrl}),\; Y_i(\text{trat1}),\; Y_i(\text{trat2})
\]

We only observe one. The comparisons are **pairwise**:

- `trat1` vs `ctrl` → effect of nudge A
- `trat2` vs `ctrl` → effect of nudge B
- `trat2` vs `trat1` → incremental effect of B over A

The meta-learners in the code binarize: \(T=0\) if `ctrl`, \(T=1\) if the chosen treatment arm.

---

## 3. From ATE to CATE to personalization

```
RCT (random group)
        │
        ▼
   ATE = E[Y(trat) − Y(ctrl)]     ← "does it work on average?"
        │
        ▼
   CATE(x) = E[Y(trat) − Y(ctrl) | X=x]   ← "does it work for this profile?"
        │
        ▼
   Policy: send trat2 if CATE(x) > threshold
```

**ATE** feeds global decisions (deploy `trat2` to everyone).

**CATE** feeds **prioritization** and **personalization** (who receives the stronger nudge first).

**Important:** in this dataset the mean CATE **does not match** the ATE (~0.20 vs ~0.40 for `trat2` vs `ctrl` / `ctor`). Use CATE to **rank segments** (ranking), and ATE for **business impact magnitudes**.

---

## 4. Meta-learners (EconML)

They all estimate \(\hat\tau(x)\) from supervised outcome models. Implemented in `src/causal.py` with `RandomForestClassifier` as the base model (binary outcome).

### 4.1 S-Learner (Single model)

A single model predicts \(Y\) using \(X\) and \(T\):

\[
\hat\tau(x) = \hat\mu(x, T=1) - \hat\mu(x, T=0)
\]

- **Advantage:** simple, a single model.
- **Risk:** if the treatment effect is small, the model may "ignore" \(T\) and underestimate \(\tau(x)\).

In this project: mean S-Learner CATE ≈ **0.21** (trat2 vs ctrl, `ctor`).

### 4.2 T-Learner (Two models)

Separate models per arm:

\[
\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)
\]

- **Advantage:** per-arm flexibility; good with strong heterogeneity.
- **Risk:** error accumulates if an arm has few observations in some segment.

Mean T-Learner CATE ≈ **0.21**.

### 4.3 X-Learner

Combines the T-Learner with a **propensity model** \(\hat e(x) = P(T=1 \mid X)\) and cross-imputation of individual effects. It usually works better when one arm is smaller or heterogeneity is marked.

In an RCT, \(\hat e(x) \approx 0.5\) constant; the X-Learner can still help in the effect-regression stage.

Mean X-Learner CATE ≈ **0.20** (used by default for segmentation in notebook 03).

### 4.4 When to use each

| Situation | Recommended learner |
|-----------|---------------------|
| Balanced RCT, initial exploration | T-Learner or S-Learner |
| Small or imbalanced treatment arm | X-Learner |
| Many covariates, suspected residual confounding | DML (see §5) |
| Segment ranking only | Any; validate with `validate_cate_vs_ate` |

---

## 5. Double Machine Learning (LinearDML)

Meta-learners estimate effects **directly** from outcome models. **DML** (Chernozhukov et al.) separates:

1. **Nuisance functions:** \(\hat\mu(x)\) (outcome) and \(\hat e(x)\) (propensity), with **cross-fitting** to avoid overfitting.
2. **Final stage:** regression of the "residual outcome" on the "residual treatment" → a **Neyman-orthogonal** estimator (more robust to nuisance errors).

In EconML:

```python
from econml.dml import LinearDML

dml = LinearDML(
    model_y=RandomForestClassifier(...),
    model_t=RandomForestClassifier(...),
    discrete_treatment=True,
    discrete_outcome=True,
    cv=3,
)
dml.fit(Y, T, X=x)
cate = dml.effect(x)
```

**Result in this project (trat2 vs ctrl, `ctor`):** mean LinearDML ≈ **0.16** — closer to the meta-learners than to the ATE, but with different variance across segments.

**DML's advantage:** solid theory under confounding (observational); in an RCT it mainly contributes **cross-fitting** and an alternative **calibration**.

---

## 6. DR-Learner: why we don't use it here

EconML's `DRLearner` (doubly robust) is powerful on observational data. In tests with this dataset and `discrete_treatment=True` without fine tuning, the CATE means came out around ~107–120 (absurd vs an ATE of 0.40).

**Typical causes:**

- Binary **outcome and treatment** require coherent models and link functions.
- DR combines propensity and outcome; with an RF that is not calibrated in the tails, the pseudo-outcomes can blow up.
- In an **RCT**, the ATE is already identified without DR; the marginal benefit does not justify the misspecification risk.

**Conclusion:** we document DR conceptually; for production in this repo we prefer T/X-Learner + validation, or LinearDML with cross-fitting.

---

## 7. Validation: CATE vs ATE calibration

Under correct identification and a well-specified model:

\[
\frac{1}{n}\sum_i \hat\tau(x_i) \approx \widehat{ATE}
\]

The `validate_cate_vs_ate` function in `src/causal.py` compares:

| Metric | Typical value (trat2, ctor) |
|--------|-----------------------------|
| ATE (diff in means) | **0.40** |
| Mean S-Learner | ~0.21 |
| Mean T-Learner | ~0.21 |
| Mean X-Learner | ~0.20 |
| Mean LinearDML | ~0.16 |

**Honest interpretation (a sign of causal maturity):**

1. Do **not** scale "+0.20 pp per customer" to 500k if the ATE says +40 pp.
2. **Do** use CATE for: age 18–35 CATE ≈ 0.67 vs 51+ ≈ 0 — a reliable relative ordering.
3. **Implemented improvements:** `CausalForestDML`, `calibrate_cate_to_ate` (shift/scale), funnel mediation in `src/mediation.py`, and out-of-sample uplift validation in `src/uplift.py` (see §11).

---

## 8. Multiple outcomes and funnel mediation

The funnel imposes structure:

\[
\text{ctor} = \text{or} \times \text{click\_if\_opened}
\quad\Rightarrow\quad
\mathbb{E}[\text{ctor}\mid T] = P(\text{or}=1\mid T)\times P(\text{ctor}=1\mid \text{or}=1, T)
\]

A nudge can raise **opening** (`or`) or **clicks conditional on opening** (CTO). Kitagawa–Blinder–Oaxaca decomposition (treatment weights on the conversion path), implemented in `src/mediation.py`:

\[
\Delta\text{ctor} = \underbrace{\text{CTO}_{\text{ctrl}}\cdot\Delta\text{or}}_{\text{via opening}}
+ \underbrace{\text{OR}_{\text{trat}}\cdot\Delta\text{CTO}}_{\text{via conversion}}
\]

| Comparison | ATE ctor | Via opening | Via conversion | Conversion share |
|------------|----------|-------------|----------------|------------------|
| trat1 vs ctrl | +26.5 pp | +9.7 pp | +16.9 pp | **64%** |
| trat2 vs ctrl | +40.2 pp | +9.7 pp | +30.5 pp | **76%** |
| trat2 vs trat1 | +13.6 pp | ~0 | +13.6 pp | **~100%** |

**Causal reading:** both nudges open the funnel (~+32 pp in `or`). Trat2's advantage over trat1 is **almost entirely post-open conversion** (CTO 58% → 80%). No complex mediation model is needed: the `ctor ⊂ or` nesting allows this exact decomposition.

```python
from src.mediation import funnel_mediation, all_funnel_mediations

funnel_mediation(df, "trat2", "ctrl")
all_funnel_mediations(df)
```

---

## 9. CausalForestDML and post-hoc calibration

### CausalForestDML

A causal forest with DML residualization + honest trees for \(\tau(x)\). In this dataset (trat2 vs ctrl, `ctor`): mean ≈ **0.19** — same order of magnitude as the meta-learners; it does not close the gap vs the ATE of 0.40 on its own.

Enabled by default in `fit_cate(..., include_causal_forest=True)`.

### Calibration to the ATE

To report magnitudes aligned with the RCT without losing the segment **ranking**:

```python
from src.causal import calibrate_cate_to_ate

# shift: cate - mean(cate) + ATE  (preserves relative differences)
cate_cal = calibrate_cate_to_ate(est.cate_x, ate=0.4015, method="shift")
```

Use the raw CATE for **who to prioritize**; calibrated CATE only if you need to communicate per-segment magnitudes aligned with the global ATE.

---

## 10. Code flow in the repository

```python
from src.data import load_data
from src.causal import (
    prep_binary_comparison, fit_cate, validate_cate_vs_ate,
    segment_cate_summary, calibrate_cate_to_ate,
)
from src.mediation import funnel_mediation

df = load_data()
print(funnel_mediation(df, "trat2", "ctrl"))

X, T, Y = prep_binary_comparison(df, treatment_arm="trat2", outcome="ctor")
est = fit_cate(X, T, Y, label="trat2 vs ctrl (ctor)")

print(est.summary())
print(validate_cate_vs_ate(df, "trat2", "ctor", est))

cate_df = est.to_frame()
cate_df["edad"] = df.loc[df["grupo"].isin(["ctrl", "trat2"]), "edad"].values
segment_cate_summary(
    cate_df, "edad",
    bins=[18, 36, 51, 100],
    labels=["18-35", "36-50", "51+"],
)
```

Tests: `pytest` — ATE, funnel mediation, calibration, and CATE validation.

Notebook: [`notebooks/03_causal_ml_heterogeneity.ipynb`](../notebooks/03_causal_ml_heterogeneity.ipynb).

---

## 11. Out-of-sample validation (uplift) and uncertainty

Estimating the CATE is not enough: we must show that the model **ranks** customers by
response and bound the **uncertainty**. This lives in [`src/uplift.py`](../src/uplift.py) and in
the notebook [`05_uplift_validation.ipynb`](../notebooks/05_uplift_validation.ipynb).

**1. Out-of-sample ranking (Qini / AUUC).** We split control + treatment into train/test
(stratified by treatment), fit the CATE on *train*, and score the *test* set the model never
saw. The `causalml.metrics` functions compare the model curve against random targeting:

- **Qini**: area between the model's Qini curve and random (normalized, ~0 = random).
- **AUUC**: area under the cumulative gain curve.

```python
from src.uplift import evaluate_uplift

scores, scored = evaluate_uplift(df, "trat2", "ctor", learners=("t", "x", "cf"))
print(scores)  # qini/auuc per learner, ranked best-first
```

A clearly positive out-of-sample Qini is the evidence that the *targeting* from notebook 03
actually works (it is not overfitting). `CausalForestDML` typically leads.

**2. CATE confidence intervals.** `CausalForestDML` provides a per-customer CI; we flag as
*significant* those whose interval excludes 0, avoiding over-interpretation of segment
differences that could be noise. For the CI of a **group average** (per segment), we use the
forest's own inference rather than averaging individual bounds:

```python
from src.uplift import cate_with_confidence, summarize_cate_ci, segment_cate_ci

ci = cate_with_confidence(df, "trat2", "ctor")   # cate, ci_lower, ci_upper, significant
print(summarize_cate_ci(ci))                      # mean_cate, mean_ci_width, frac_significant

# formal group-average CI via effect_inference().population_summary()
seg = segment_cate_ci(df, "trat2", "edad", bins=[17, 35, 50, 100],
                      labels=["18-35", "36-50", "51+"])
print(seg)  # per segment: mean_cate, ci_lower, ci_upper, significant
```

Typical result: the 18–35 and 36–50 segments show a significant effect, while the 51+ group
interval includes 0 — the effect is not distinguishable from noise there.

---

## 12. References

- Imbens & Rubin (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*.
- Chernozhukov et al. (2018). Double/debiased machine learning for treatment and structural parameters.
- Athey, Tibshirani & Wager (2019). Generalized random forests.
- Kitagawa (1955) / Blinder–Oaxaca — mean-difference decompositions.
- Radcliffe (2007). Using control groups to target on predicted lift: building and assessing uplift models.
- [EconML documentation](https://econml.azurewebsites.net/) — meta-learners and DML.
- Künzel et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning.
