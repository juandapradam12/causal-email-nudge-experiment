# Causal ML en el experimento de emails

Guía de referencia para demostrar el marco causal y las técnicas de **Causal Machine Learning** aplicadas en este repositorio. Complementa [`GUIA_CONCEPTUAL_TECNICA.md`](GUIA_CONCEPTUAL_TECNICA.md) con mayor profundidad metodológica.

---

## 1. Pregunta causal vs pregunta predictiva

| Enfoque | Pregunta | Ejemplo en este proyecto |
|---------|----------|--------------------------|
| **Predictivo** | ¿Quién hará clic? | Clasificador de `ctor` sin contrafactual |
| **Causal** | ¿El nudge *causó* más clics que el control? | ATE en RCT |
| **Causal ML** | ¿Para *qué perfil* el nudge causa más clics? | CATE + personalización |

Un modelo predictivo puede asociar “edad joven” con más clics porque jóvenes ya clicaban más **sin** el nudge. El CATE intenta estimar el **incremento atribuible al tratamiento** condicionado al perfil \(X\).

---

## 2. Identificación en un RCT

Para que un estimador de efecto tenga interpretación causal, hacen falta supuestos. En un experimento aleatorizado bien ejecutado:

### 2.1 Asignación aleatoria (ignorabilidad)

\[
Y(0), Y(1) \perp T \quad \Rightarrow \quad \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0] = \mathbb{E}[Y(1) - Y(0)]
\]

La asignación a `ctrl`, `trat1` o `trat2` es independiente de los resultados potenciales. Por eso la **diferencia de medias por grupo** es un estimador válido del ATE sin ajustar por covariables.

### 2.2 SUTVA (Stable Unit Treatment Value Assumption)

El resultado de un cliente no depende del tratamiento asignado a otro cliente. En emails masivos es razonable: cada cliente recibe una variante; no hay “contagio” entre unidades.

### 2.3 Positivity

Cada cliente tiene probabilidad positiva de estar en cada brazo. Con randomización balanceada (~33% por grupo), se cumple.

### 2.4 Resultado potencial con tres brazos

Aquí hay **tres** tratamientos, no uno. Para cada cliente \(i\):

\[
Y_i(\text{ctrl}),\; Y_i(\text{trat1}),\; Y_i(\text{trat2})
\]

Solo observamos uno. Las comparaciones son **pareadas**:

- `trat1` vs `ctrl` → efecto del nudge A
- `trat2` vs `ctrl` → efecto del nudge B
- `trat2` vs `trat1` → efecto incremental de B sobre A

Los meta-learners del código binarizan: \(T=0\) si `ctrl`, \(T=1\) si el brazo de tratamiento elegido.

---

## 3. De ATE a CATE a personalización

```
RCT (grupo aleatorio)
        │
        ▼
   ATE = E[Y(trat) − Y(ctrl)]     ← “¿funciona en promedio?”
        │
        ▼
   CATE(x) = E[Y(trat) − Y(ctrl) | X=x]   ← “¿funciona para este perfil?”
        │
        ▼
   Política: enviar trat2 si CATE(x) > umbral
```

**ATE** alimenta decisiones globales (desplegar `trat2` a todos).

**CATE** alimenta **priorización** y **personalización** (quién recibe primero el nudge más fuerte).

**Importante:** en este dataset la media del CATE **no coincide** con el ATE (~0.20 vs ~0.40 en `trat2` vs `ctrl` / `ctor`). Usar CATE para **ordenar segmentos** (ranking), y ATE para **magnitudes de impacto** en negocio.

---

## 4. Meta-learners (EconML)

Todos estiman \(\hat\tau(x)\) a partir de modelos de outcome supervisados. Implementación en `src/causal.py` con `RandomForestClassifier` como modelo base (outcome binario).

### 4.1 S-Learner (Single model)

Un solo modelo predice \(Y\) usando \(X\) y \(T\):

\[
\hat\tau(x) = \hat\mu(x, T=1) - \hat\mu(x, T=0)
\]

- **Ventaja:** simple, un solo modelo.
- **Riesgo:** si el efecto del tratamiento es pequeño, el modelo puede “ignorar” \(T\) y subestimar \(\tau(x)\).

En este proyecto: media CATE S-Learner ≈ **0.21** (trat2 vs ctrl, `ctor`).

### 4.2 T-Learner (Two models)

Modelos separados por brazo:

\[
\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)
\]

- **Ventaja:** flexibilidad por brazo; bueno con heterogeneidad fuerte.
- **Riesgo:** error se acumula si un brazo tiene pocas observaciones en algún segmento.

Media CATE T-Learner ≈ **0.21**.

### 4.3 X-Learner

Combina T-Learner con **modelo de propensión** \(\hat e(x) = P(T=1 \mid X)\) e imputación cruzada de efectos individuales. Suele funcionar mejor cuando un brazo es más pequeño o la heterogeneidad es marcada.

En RCT, \(\hat e(x) \approx 0.5\) constante; el X-Learner aún puede ayudar en la etapa de regresión del efecto.

Media CATE X-Learner ≈ **0.20** (usada por defecto para segmentación en notebook 03).

### 4.4 Cuándo usar cada uno

| Situación | Learner recomendado |
|-----------|---------------------|
| RCT balanceado, exploración inicial | T-Learner o S-Learner |
| Brazo tratamiento pequeño o desbalanceado | X-Learner |
| Muchas covariables, sospecha de confusión residual | DML (ver §5) |
| Solo ranking de segmentos | Cualquiera; validar con `validate_cate_vs_ate` |

---

## 5. Double Machine Learning (LinearDML)

Los meta-learners estiman efectos **directamente** desde modelos de outcome. **DML** (Chernozhukov et al.) separa:

1. **Nuisance functions:** \(\hat\mu(x)\) (outcome) y \(\hat e(x)\) (propensión), con **cross-fitting** para evitar overfitting.
2. **Etapa final:** regresión del “residual outcome” sobre el “residual treatment” → estimador **Neyman-orthogonal** (más robusto a errores en nuisance).

En EconML:

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

**Resultado en este proyecto (trat2 vs ctrl, `ctor`):** media LinearDML ≈ **0.16** — más cerca de meta-learners que del ATE, pero con varianza distinta en segmentos.

**Ventaja de DML:** teoría sólida bajo confusión (observacional); en RCT aporta principalmente **cross-fitting** y **calibración** alternativa.

---

## 6. DR-Learner: por qué no lo usamos aquí

`DRLearner` de EconML (doubly robust) es potente en datos observacionales. En pruebas con este dataset y `discrete_treatment=True` sin configuración fina, las medias de CATE salieron ~107–120 (absurdo vs ATE 0.40).

**Causas típicas:**

- Outcome y tratamiento **binarios** requieren modelos y enlace coherentes.
- DR combina propensión y outcome; con RF no calibrado en colas, los pseudo-outcomes pueden explotar.
- En **RCT**, el ATE ya es identificado sin DR; el beneficio marginal no compensa el riesgo de mala especificación.

**Conclusión:** documentamos DR conceptualmente; para producción en este repo preferimos T/X-Learner + validación, o LinearDML con cross-fitting.

---

## 7. Validación: calibración CATE vs ATE

Bajo identificación correcta y modelo bien especificado:

\[
\frac{1}{n}\sum_i \hat\tau(x_i) \approx \widehat{ATE}
\]

Función `validate_cate_vs_ate` en `src/causal.py` compara:

| Métrica | Valor típico (trat2, ctor) |
|---------|---------------------------|
| ATE (diff medias) | **0.40** |
| Media S-Learner | ~0.21 |
| Media T-Learner | ~0.21 |
| Media X-Learner | ~0.20 |
| Media LinearDML | ~0.16 |

**Interpretación honesta (señal de madurez causal):**

1. **No** escalar “+0.20 pp por cliente” a 500k si el ATE dice +40 pp.
2. **Sí** usar CATE para: edad 18–35 CATE ≈ 0.67 vs 51+ ≈ 0 — orden relativo fiable.
3. **Mejoras implementadas:** `CausalForestDML`, `calibrate_cate_to_ate` (shift/scale), mediación del funnel en `src/mediation.py`.

---

## 8. Outcomes múltiples y mediación del funnel

El funnel impone estructura:

\[
\text{ctor} = \text{or} \times \text{click\_si\_abrió}
\quad\Rightarrow\quad
\mathbb{E}[\text{ctor}\mid T] = P(\text{or}=1\mid T)\times P(\text{ctor}=1\mid \text{or}=1, T)
\]

Un nudge puede subir **apertura** (`or`) o **clics condicionados a apertura** (CTO). Descomposición Kitagawa–Blinder–Oaxaca (pesos tratamiento en la vía de conversión), implementada en `src/mediation.py`:

\[
\Delta\text{ctor} = \underbrace{\text{CTO}_{\text{ctrl}}\cdot\Delta\text{or}}_{\text{vía apertura}}
+ \underbrace{\text{OR}_{\text{trat}}\cdot\Delta\text{CTO}}_{\text{vía conversión}}
\]

| Comparación | ATE ctor | Vía apertura | Vía conversión | Share conversión |
|-------------|----------|--------------|----------------|------------------|
| trat1 vs ctrl | +26.5 pp | +9.7 pp | +16.9 pp | **64%** |
| trat2 vs ctrl | +40.2 pp | +9.7 pp | +30.5 pp | **76%** |
| trat2 vs trat1 | +13.6 pp | ~0 | +13.6 pp | **~100%** |

**Lectura causal:** ambos nudges abren el funnel (~+32 pp en `or`). La ventaja de `trat2` sobre `trat1` es **casi solo conversión post-apertura** (CTO 58% → 80%). No hace falta un PEMs complejo: la anidación `ctor ⊂ or` permite esta descomposición exacta.

```python
from src.mediation import funnel_mediation, all_funnel_mediations

funnel_mediation(df, "trat2", "ctrl")
all_funnel_mediations(df)
```

---

## 9. CausalForestDML y calibración post-hoc

### CausalForestDML

Bosque causal con residualización DML + árboles honestos para \(\tau(x)\). En este dataset (trat2 vs ctrl, `ctor`): media ≈ **0.19** — misma orden de magnitud que meta-learners; no cierra sola la brecha vs ATE 0.40.

Activado por defecto en `fit_cate(..., include_causal_forest=True)`.

### Calibración al ATE

Para reportar magnitudes alineadas al RCT sin perder el **ranking** de segmentos:

```python
from src.causal import calibrate_cate_to_ate

# shift: cate - mean(cate) + ATE  (preserva diferencias relativas)
cate_cal = calibrate_cate_to_ate(est.cate_x, ate=0.4015, method="shift")
```

Usar CATE crudo para **quién priorizar**; CATE calibrado solo si se necesita comunicar magnitudes por segmento alineadas al ATE global.

---

## 10. Flujo de código en el repositorio

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

Tests: `pytest tests/` — ATE, mediación del funnel, calibración y validación CATE.

Notebook: [`notebooks/03_causal_ml_heterogeneity.ipynb`](../notebooks/03_causal_ml_heterogeneity.ipynb).

---

## 11. Referencias

- Imbens & Rubin (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*.
- Chernozhukov et al. (2018). Double/debiased machine learning for treatment and structural parameters.
- Athey, Tibshirani & Wager (2019). Generalized random forests.
- Kitagawa (1955) / Blinder–Oaxaca — descomposiciones de diferencias de medias.
- [EconML documentation](https://econml.azurewebsites.net/) — meta-learners y DML.
- Künzel et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning.
