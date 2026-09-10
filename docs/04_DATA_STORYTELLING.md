# 04 — Data storytelling for the client

Executive narrative of the email experiment with behavioral-science nudges.

---

## 1. The bank's challenge

The bank sends customers emails with a call to action (e.g. activate a product, complete a flow). Without behavioral nudges, engagement is low:

| Variant | Open rate | Click rate |
|---------|-----------|------------|
| Control (`ctrl`) | 28.8% | 8.7% |

**Business question:** Can an email with a behavioral-science nudge move these rates without changing the underlying product?

---

## 2. What was tested

Three variants of the **same base email**, differing only by the embedded nudge:

| Code | Role | What it represents |
|------|------|--------------------|
| `ctrl` | **Control** | Standard email, no nudge — causal reference |
| `trat1` | **Treatment 1** | Email + behavioral nudge A |
| `trat2` | **Treatment 2** | Email + behavioral nudge B |

5,000 customers randomly assigned (sample of 500,000).

---

## 3. Experiment result

| Variant | Open rate | Click rate |
|---------|-----------|------------|
| Control | 28.8% | 8.7% |
| Treatment 1 | 60.8% | 35.3% |
| Treatment 2 | 60.8% | **48.9%** |

**One-line takeaway:** the nudges **double the open rate**; Treatment 2 additionally **multiplies the click rate ~5×** versus control.

### Where the effect acts (funnel mediation)

| Comparison | % of average treatment effect (ATE) via opening | % via post-open conversion |
|------------|----------------------|----------------------------|
| Trat1 vs control | 36% | 64% |
| Trat2 vs control | 24% | **76%** |
| Trat2 vs Trat1 | ~0% | **~100%** |

Trat1 and Trat2 open equally; Trat2's advantage is **almost entirely** more clicks among those who already opened.

---

## 4. Estimated impact at scale

If we deploy **Treatment 2** to the full population (500,000 customers):

| Metric | Value |
|--------|-------|
| Absolute lift in click rate | +40.2 pp |
| Additional clicks vs control | **~200,773** |

Calculation: `(48.9% − 8.7%) × 500,000 ≈ 200,773`.

---

## 5. Personalization (Causal ML)

The heterogeneous-effects analysis (notebook 03) shows that not all customers respond equally:

| Segment | conditional average treatment effect (CATE) trat2 (ctor) | Action |
|---------|-------------------|--------|
| Age 18–35 | High (~0.67) | Prioritize trat2 |
| Age 36–50 | Moderate (~0.21) | Deploy trat2 |
| Age 51+ | Close to 0 | Evaluate an alternative |
| App users | Higher than non-app | Prioritize in digital campaigns |

The out-of-sample validation (notebook 05) confirms this ranking holds on held-out customers, and the formal confidence intervals show the 18–35 and 36–50 effects are statistically significant while the 51+ effect is not. See [`CAUSAL_ML.md`](CAUSAL_ML.md) for the technical framework.

---

## 6. Recommendation

| Action | Expected impact |
|--------|-----------------|
| Deploy **Treatment 2** as the main variant | +40.2 pp in click rate vs control |
| Prioritize high-CATE segments (younger customers, app users) | Additional gains via personalization |
| Keep a continuous A/B post-launch | Early detection of nudge fatigue |
