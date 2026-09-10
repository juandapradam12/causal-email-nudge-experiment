# Conceptual and technical guide — Email experiment with nudges

Reference document for the `causal-email-nudge-experiment` project. It summarizes the business problem, the causal framework, the code implementation, and how to interpret each numerical result.

---

## 1. The business problem

A bank wants to **increase engagement** with transactional or marketing emails. Instead of testing only copy or visual design, the experiment evaluates **behavioral-science nudges** embedded in the email.

### Questions the project answers

| Level | Question | Method |
|-------|----------|--------|
| Descriptive | How are customers and outcomes distributed? | EDA (`01_load_and_eda.ipynb`) |
| Global inference | Do the nudges work on average? | ATE, tests, regression (`02_basic_experiment_analysis.ipynb`) |
| Local inference | For **whom** does each nudge work best? | CATE with meta-learners (`03_causal_ml_heterogeneity.ipynb`) |
| Validation | Does the targeting hold out-of-sample? | Qini/AUUC + CATE CIs (`05_uplift_validation.ipynb`) |
| Decision | What to deploy and at what scale? | Storytelling + impact (`04_data_storytelling.ipynb`) |

### Experimental design: randomized controlled trial (RCT)

- **Target population:** 500,000 bank customers.
- **Analyzed sample:** 5,000 customers (~1%), randomly assigned.
- **Arms:**
  - `ctrl` — control email (no nudge).
  - `trat1` — behavioral nudge 1.
  - `trat2` — behavioral nudge 2.

Randomization is the key piece: in a well-executed randomized controlled trial (RCT), **we do not need to control for covariates to estimate the average treatment effect (ATE)**. Covariates come into play to (a) check balance, (b) gain precision in regression, and (c) estimate heterogeneous effects (CATE).

### What is `ctrl` and what is `trat`? (conceptual)

The names in the `grupo` column are **experiment arms**, not arbitrary labels:

| Code | Usual name | What it is in practice |
|------|------------|------------------------|
| `ctrl` | **Control** | Email **without** the behavioral-science nudge. It is the **causal reference**: "what happens with the standard email?" |
| `trat1` | **Treatment 1** | Same base email + **nudge A** (e.g. anchoring, urgency, different framing). |
| `trat2` | **Treatment 2** | Same base email + **nudge B** (another behavioral intervention). |

**`ctrl` does not mean "receive no email".** Every customer receives an email; control receives the version **without** the experimental nudge. The causal question is: *does the nudge improve engagement relative to the email we were already sending?*

**`trat` (trat1 / trat2)** are the **interventions** we want to evaluate. Each defines a distinct potential outcome:

$$
Y_i(\text{ctrl}),\quad Y_i(\text{trat1}),\quad Y_i(\text{trat2})
$$

For customer $i$ we only observe **one** — the one for the randomly assigned arm:

$$
Y_i^{\text{obs}} = Y_i(T_i), \quad T_i \in \{\text{ctrl}, \text{trat1}, \text{trat2}\}
$$

The other two are **counterfactuals** (unobserved). Randomization lets us replace counterfactual expectations with the corresponding group means:

$$
ATE_{\text{trat2 vs ctrl}} = \mathbb{E}[Y(\text{trat2}) - Y(\text{ctrl})] \approx \bar{Y}_{\text{trat2}} - \bar{Y}_{\text{ctrl}}
$$

**Analogy:** in a clinical trial, "placebo" is not "no medicine"; it is the reference treatment. Here `ctrl` is the reference email; `trat1` and `trat2` are the nudge variants.

For depth on CATE, meta-learners and DML, see [`CAUSAL_ML.md`](CAUSAL_ML.md).

---

## 2. Variables and their causal meaning

| Variable | Type | Causal role | Interpretation |
|----------|------|-------------|----------------|
| `iid` | ID | — | Unique customer identifier |
| `grupo` | Treatment $T$ | **Intervention** | Email variant received |
| `or` | Binary | **Intermediate outcome** | Did they open the email? (open rate) |
| `ctor` | Binary | **Final outcome** | Did they click the button? |
| `sexo`, `edad`, `inve`, `uso_app`, `tarjeta_debito`, `tipo_tarjeta`, `formacion` | Covariates $X$ | **Pre-treatment** | Customer profile before the email |

### Relationship between `or` and `ctor`

In the data, **`ctor` is nested within `or`**: if `or = 0`, then `ctor = 0` always. Therefore:

$$
\text{ctor} = \mathbb{1}[\text{opened}] \times \mathbb{1}[\text{clicked}]
$$

- **Open rate:** $\bar{or} = P(\text{open})$
- **Click rate (`ctor`):** $P(\text{open} \cap \text{click})$ — overall click-conversion rate.
- **Click-to-open (conditional CTOR):** $P(\text{click} \mid \text{open}) = \bar{ctor} / \bar{or}$ when $or > 0$.

The nudge can act at **two funnel stages**:

1. **Opening** — the subject line / preview convinces them to open.
2. **Post-open conversion** — the email content convinces them to click.

That is why we analyze **both outcomes** separately.

---

## 3. Causal framework: notation and estimands

### Potential outcomes model (Rubin)

For each customer $i$ there exist potential outcomes $Y_i(0), Y_i(1), Y_i(2)$ depending on the assigned arm. We only observe one:

$$
Y_i^{\text{obs}} = Y_i(T_i), \quad T_i \in \{\text{ctrl}, \text{trat1}, \text{trat2}\}
$$

### ATE (Average Treatment Effect)

To compare `trat2` vs `ctrl` on click rate:

$$
ATE = \mathbb{E}[Y(\text{trat2}) - Y(\text{ctrl})]
$$

In an RCT with a binary outcome, the natural estimator is the **difference in proportions**:

$$
\widehat{ATE} = \bar{Y}_{\text{trat2}} - \bar{Y}_{\text{ctrl}}
$$

**Results in this dataset:**

| Comparison | Outcome | ATE (pp) | p-value | 95% CI |
|------------|---------|----------|---------|--------|
| trat1 vs ctrl | or | +31.9 pp | ≈ 0 | [28.7, 35.1] |
| trat2 vs ctrl | or | +32.0 pp | ≈ 0 | [28.8, 35.1] |
| trat2 vs trat1 | or | +0.0 pp | 1.00 | [-3.3, +3.4] |
| trat1 vs ctrl | ctor | +26.5 pp | ≈ 0 | [23.9, 29.2] |
| trat2 vs ctrl | ctor | +40.2 pp | ≈ 0 | [37.4, 42.9] |
| trat2 vs trat1 | ctor | +13.6 pp | ≈ 0 | [10.3, 17.0] |

**Reading:**

- Both nudges **roughly double** the open rate (~29% → ~61%).
- On opening, **trat1 ≈ trat2** (no statistically significant difference).
- On clicks, **trat2 dominates**: +40 pp vs control, and +14 pp vs trat1.
- Nudge 2 improves over nudge 1 mainly in **click conversion**, not opening.

### CATE (Conditional Average Treatment Effect)

$$
CATE(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
$$

It answers: *how much extra benefit does a customer with profile $x$ get from receiving the treatment?*

This enables **personalization**: send `trat2` first to segments with a high CATE.

---

## 4. Analysis pipeline by phase

```
datos_prueba_tecnica.csv
        │
        ▼
  src/data.py ─── load_data(), types, GROUP_LABELS
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  01 EDA                                    02 Classic ATE
  • group balance                           • diff in proportions + CI
  • or/ctor rates                           • chi-square
  • covariate balance                       • adjusted logistic regression
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
                    03 Causal ML (CATE)
                    • T-Learner / X-Learner
                    • segmentation by age, uso_app
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      05 Uplift validation        04 Storytelling
      • Qini / AUUC (OOS)         • executive narrative
      • CATE confidence intervals • impact at 500k customers
                                  • recommendations
```

### Phase 1–2: Infrastructure (`src/data.py`)

```python
from src.data import load_data, GROUP_LABELS
df = load_data()
```

- Types `grupo` as an ordered categorical (`ctrl < trat1 < trat2`).
- Converts binaries to `int`.
- Centralizes the CSV path in `DATA_PATH`.

### Phase 3: EDA (`01_load_and_eda.ipynb`)

**Goal:** Validate data quality and design plausibility.

Implemented checks:

1. 5,000 rows, 5,000 unique `iid`, no nulls.
2. Group-size balance (~1,650–1,700 per arm).
3. Covariate balance tests (ANOVA / chi-square).

**Observed rates per group:**

| Group | Open rate | Click rate |
|-------|-----------|------------|
| ctrl  | 28.8%     | 8.7%       |
| trat1 | 60.8%     | 35.3%      |
| trat2 | 60.8%     | 48.9%      |

**Note on balance:** Some covariates (`edad`, `inve`, `sexo`) show low p-values in univariate tests. This is **expected with 5,000 observations** — balance tests detect tiny differences. What matters is that the magnitudes are small and that the ATE does not depend on adjustments (verified in phase 4).

### Phase 4: Classic analysis (`02_basic_experiment_analysis.ipynb` + `src/analysis.py`)

#### Diff-in-means estimator

For binary proportions, the standard error is:

$$
SE = \sqrt{\frac{p_T(1-p_T)}{n_T} + \frac{p_C(1-p_C)}{n_C}}
$$

95% CI: $\widehat{ATE} \pm 1.96 \cdot SE$

Reusable implementation:

```python
from src.analysis import all_ate_comparisons, scale_impact

ate_df = all_ate_comparisons(df)
impact = scale_impact(ate_pp=0.4015, population_size=500_000, outcome_label="clicks")
# → ~200,773 additional clicks vs control with trat2
```

#### Adjusted logistic regression

Model:

$$
\log\frac{P(Y=1)}{1-P(Y=1)} = \beta_0 + \beta_1 \cdot \mathbb{1}[trat1] + \beta_2 \cdot \mathbb{1}[trat2] + \gamma^T X
$$

| Outcome | Treatment | OR | Interpretation |
|---------|-----------|-----|----------------|
| or | trat1 | 2.62 | 2.6× odds of opening vs control |
| or | trat2 | 1.66* | See note below |
| ctor | trat1 | 2.49 | 2.5× odds of clicking vs control |
| ctor | trat2 | 2.00 | 2× odds of clicking vs control |

\*The `trat2` ORs on open rate are lower than `trat1` **after adjusting for covariates**, while the raw ATE is almost identical. This indicates **residual covariate confounding** (slight imbalance) — another reason to trust the RCT's nonparametric estimator as the primary source.

### Phase 5: Causal ML (`03_causal_ml_heterogeneity.ipynb` + `src/causal.py`)

#### Why meta-learners?

In an RCT the ATE is easy to estimate. But the business wants **actionable segments**. Meta-learners decompose the problem into supervised outcome models:

| Learner | Idea | Effect formula |
|---------|------|----------------|
| **S-Learner** | One model with $T$ as a feature | $\hat\tau(x) = \hat\mu(x,1) - \hat\mu(x,0)$ |
| **T-Learner** | Separate model per arm | $\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)$ |
| **X-Learner** | Uses propensity + cross-imputation | Better when an arm is smaller or heterogeneity is strong |
| **LinearDML** | Cross-fitting + orthogonal regression | Robust to poorly estimated nuisances; see `CAUSAL_ML.md` |

Implementation:

```python
from src.causal import prep_binary_comparison, fit_cate, validate_cate_vs_ate

X, T, Y = prep_binary_comparison(df, treatment_arm="trat2", outcome="ctor")
est = fit_cate(X, T, Y, label="trat2 vs ctrl (ctor)")
validation = validate_cate_vs_ate(df, "trat2", "ctor", est)
```

#### CATE results (X-Learner, outcome `ctor`)

| Comparison | Manual ATE | Mean CATE | CATE Std |
|------------|------------|-----------|----------|
| trat1 vs ctrl | 0.265 | 0.171 | 0.41 |
| trat2 vs ctrl | 0.402 | 0.204 | 0.46 |

#### Detected heterogeneity (trat2 vs ctrl)

| Segment | Mean CATE | Interpretation |
|---------|-----------|----------------|
| Age 18–35 | **0.67** | Younger: very high response to nudge 2 |
| Age 36–50 | 0.21 | Moderate response |
| Age 51+ | **−0.01** | No net benefit (possible fatigue or mismatch) |
| No app | 0.17 | |
| With app | **0.24** | Digital users respond more |

#### ⚠️ Important validation: CATE calibration

The **mean CATE should approximate the ATE** (both estimate the same estimand under causal identification). In this project there is a **systematic gap** (~10–20 pp):

- Manual ATE trat2: **0.40**
- Mean X-Learner CATE: **0.20**

**Probable causes:**

1. **Binary outcome + Random Forest:** the meta-learners use regression/classification models that can be poorly calibrated in the tails.
2. **High individual variance:** std(CATE) ≈ 0.46; many negative CATEs offset the extreme positive ones.
3. **Poorly converged propensity:** in an RCT the propensity is ~0.5, but the logistic model may not converge well with many dummies (warning in the notebook).

**Practical implication:**

- Use CATE for **relative segment ranking** (who responds more vs less), not for absolute impact magnitudes.
- For absolute magnitudes, trust the **ATE from notebook 02**.
- Optional: `calibrate_cate_to_ate(cate, ate, method="shift")` aligns the mean to the ATE while preserving the ranking.
- Additional estimators: `LinearDML` and `CausalForestDML` (same order of magnitude ~0.16–0.20).
- The ranking is confirmed **out-of-sample** in phase 5b (Qini/AUUC), with formal confidence intervals per segment.

#### Funnel mediation (`src/mediation.py`)

Because `ctor` is nested within `or`:

| Comparison | Via opening | Via conversion | Insight |
|------------|-------------|----------------|---------|
| trat1 vs ctrl | 36% | **64%** | Also improves CTO, not just opening |
| trat2 vs ctrl | 24% | **76%** | Post-open conversion is the driver |
| trat2 vs trat1 | ~0% | **~100%** | Same opening; trat2 wins only on clicks |

### Phase 5b: Uplift validation (`05_uplift_validation.ipynb` + `src/uplift.py`)

Estimating the CATE is not enough — we must show the model **ranks** customers well and quantify **uncertainty**:

- **Out-of-sample ranking:** train/test split (stratified by treatment), fit on train, score the held-out test set, and measure **Qini** and **AUUC** (`causalml.metrics`). A clearly positive Qini beats random targeting and shows the targeting is not overfitting; the Causal Forest typically leads.
- **Confidence intervals:** per-customer CIs from `CausalForestDML`, and a formal **group-average CI per segment** via `effect_inference(...).population_summary()`. The 18–35 and 36–50 segments are significant; the 51+ interval includes 0.

```python
from src.uplift import evaluate_uplift, cate_with_confidence, segment_cate_ci

scores, scored = evaluate_uplift(df, "trat2", "ctor", learners=("t", "x", "cf"))
seg = segment_cate_ci(df, "trat2", "edad", bins=[17, 35, 50, 100],
                      labels=["18-35", "36-50", "51+"])
```

---

## 5. Causal funnel diagram

```mermaid
flowchart LR
    A[Random assignment<br/>grupo] --> B{Opens email?<br/>or}
    B -->|Yes| C{Click?<br/>ctor}
    B -->|No| D[ctor = 0]
    C -->|Yes| E[Conversion]
    C -->|No| F[No conversion]

    style A fill:#e3f2fd
    style E fill:#c8e6c9
```

The nudges move the funnel at **two points**:

- **trat1 and trat2** → large jump in `or` (opening).
- **trat2 additionally** → extra jump in `ctor` given that they already opened.

---

## 6. Business decisions (storytelling)

### Main recommendation

**Deploy `trat2` as the main variant** for the base of 500,000 customers.

### Estimated impact

| Metric | Control | Trat2 | Delta |
|--------|---------|-------|-------|
| Click rate | 8.7% | 48.9% | **+40.2 pp** |
| Clicks in 500k | 43,650 | 244,400 | **+200,773** |

### Suggested personalization

1. **Prioritize `trat2` for customers 18–35 and app users** (high CATE).
2. **Evaluate an alternative for 51+** — the near-zero CATE suggests nudge 2 does not help or may be counterproductive.
3. **Keep a continuous A/B** post-launch to detect nudge fatigue.

---

## 7. Repository file map

```
causal-email-nudge-experiment/
├── data/datos_prueba_tecnica.csv    # 5,000 experiment rows
├── docs/
│   ├── EXPERIMENT_DESIGN.md         # Experiment design brief (Markdown)
│   ├── VARIABLES.md                 # Variable dictionary (Markdown)
│   ├── GUIA_CONCEPTUAL_TECNICA.md   # ← this document
│   ├── CAUSAL_ML.md                 # Causal ML framework (meta-learners, DML)
│   ├── 04_DATA_STORYTELLING.md      # Executive narrative (Markdown)
│   └── reports/                     # Rendered, executed notebook reports (HTML)
├── notebooks/
│   ├── 01_load_and_eda.ipynb
│   ├── 02_basic_experiment_analysis.ipynb
│   ├── 03_causal_ml_heterogeneity.ipynb
│   ├── 04_data_storytelling.ipynb
│   └── 05_uplift_validation.ipynb
├── src/
│   ├── data.py       # Loading and typing
│   ├── analysis.py   # ATE, regression, impact
│   ├── causal.py     # CATE (meta-learners + DML + CausalForest)
│   ├── mediation.py  # Funnel decomposition or → ctor
│   └── uplift.py     # Out-of-sample uplift (Qini/AUUC) + CATE confidence intervals
├── tests/test_core.py
├── tests/test_uplift.py
├── scripts/build_notebooks.py
├── scripts/export_reports.sh
├── pyproject.toml
└── requirements.txt
```

---

## 8. Assumptions and limitations

| Assumption | Status in this project | Risk |
|------------|------------------------|------|
| SUTVA (no interference) | Emails to distinct customers | Low |
| Random assignment | Verified by design | Low |
| i.i.d. units | Simple random sample | Low |
| Correct measurement | No nulls, consistent binaries | Low |
| External validity | Only 5k of 500k | Medium — validate in rollout |
| Calibrated CATE | ATE vs mean-CATE gap | Medium — ranking + `calibrate_cate_to_ate` |

---

## 9. Quick reference

- **ATE / diff-in-means:** primary estimator in an RCT; see `src/analysis.py`.
- **Logistic regression:** adjusted odds ratios; notebook 02.
- **S/T/X-Learner, LinearDML, CausalForestDML:** `src/causal.py`, [`CAUSAL_ML.md`](CAUSAL_ML.md).
- **Funnel mediation:** `src/mediation.py`.
- **Uplift validation & CATE CIs:** `src/uplift.py`, notebook 05.
- **Tests:** `pytest`.
- **Potential outcomes framework:** Imbens & Rubin (2015), *Causal Inference for Statistics, Social, and Biomedical Sciences*.
