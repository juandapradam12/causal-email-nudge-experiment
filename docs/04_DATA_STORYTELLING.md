# 04 — Data storytelling para el cliente

Narrativa ejecutiva del experimento de emails con nudges de ciencias del comportamiento (caso BeWay).

---

## 1. El reto del banco

El banco envía emails a clientes con una llamada a la acción (ej. activar producto, completar un flujo). Sin nudges de comportamiento, la interacción es baja:

| Variante | Open rate | Click rate |
|----------|-----------|------------|
| Control (`ctrl`) | 28.8% | 8.7% |

**Pregunta de negocio:** ¿Puede un email con nudge de ciencias del comportamiento mover estas tasas sin cambiar el producto subyacente?

---

## 2. Qué se probó

Tres variantes del **mismo email base**, diferenciadas solo por el nudge incrustado:

| Código | Rol | Qué representa |
|--------|-----|----------------|
| `ctrl` | **Control** | Email estándar, sin nudge — referencia causal |
| `trat1` | **Tratamiento 1** | Email + nudge de comportamiento A |
| `trat2` | **Tratamiento 2** | Email + nudge de comportamiento B |

5.000 clientes asignados aleatoriamente (muestra de 500.000).

---

## 3. Resultado del experimento

| Variante | Open rate | Click rate |
|----------|-----------|------------|
| Control | 28.8% | 8.7% |
| Tratamiento 1 | 60.8% | 35.3% |
| Tratamiento 2 | 60.8% | **48.9%** |

**Lectura en una frase:** Los nudges **duplican la apertura**; el Tratamiento 2 además **multiplica por ~5** la tasa de clic respecto al control.

### Dónde actúa el efecto (mediación del funnel)

| Comparación | % del ATE vía apertura | % vía conversión post-apertura |
|-------------|------------------------|--------------------------------|
| Trat1 vs control | 36% | 64% |
| Trat2 vs control | 24% | **76%** |
| Trat2 vs Trat1 | ~0% | **~100%** |

Trat1 y Trat2 abren igual; la ventaja de Trat2 es **casi solo** más clics entre quienes ya abrieron.

---

## 4. Impacto estimado al escalar

Si desplegamos **Tratamiento 2** a la población completa (500.000 clientes):

| Métrica | Valor |
|---------|-------|
| Lift absoluto en click rate | +40.2 pp |
| Clics adicionales vs control | **~200.773** |

Cálculo: `(48.9% − 8.7%) × 500.000 ≈ 200.773`.

---

## 5. Personalización (Causal ML)

El análisis de efectos heterogéneos (notebook 03) muestra que no todos los clientes responden igual:

| Segmento | CATE trat2 (ctor) | Acción |
|----------|-------------------|--------|
| Edad 18–35 | Alto (~0.67) | Priorizar trat2 |
| Edad 36–50 | Moderado (~0.21) | Desplegar trat2 |
| Edad 51+ | Cercano a 0 | Evaluar alternativa |
| Usuarios app | Mayor que sin app | Priorizar en campañas digitales |

Ver [`CAUSAL_ML.md`](CAUSAL_ML.md) para el marco técnico.

---

## 6. Recomendación

| Acción | Impacto esperado |
|--------|------------------|
| Desplegar **Tratamiento 2** como variante principal | +40.2 pp en click rate vs control |
| Priorizar segmentos con CATE alto (jóvenes, usuarios app) | Optimización adicional vía personalización |
| Mantener A/B continuo post-lanzamiento | Detección temprana de fatiga del nudge |

---

## 7. Siguiente paso

1. Rollout de trat2 a la base objetivo.
2. Monitorizar open/click por cohorte.
3. Re-estimar CATE trimestralmente para ajustar targeting.
