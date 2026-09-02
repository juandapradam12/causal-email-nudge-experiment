# causal-email-nudge-experiment

Análisis causal de un experimento A/B de emails con nudges de ciencias del comportamiento (caso BeWay). El proyecto cubre el enunciado de la prueba técnica (EDA, análisis básico, data storytelling) y una extensión con **Causal ML** para estimar efectos heterogéneos (CATE).

## Contexto del experimento

Un banco probó tres variantes de email para aumentar:

- **Open rate** (`or`): ¿el cliente abrió el email?
- **Click-to-open rate** (`ctor`): ¿clicó en el botón de acción?

| Grupo  | Descripción                          |
|--------|--------------------------------------|
| `ctrl` | Email control (sin nudge)            |
| `trat1`| Email con nudge de comportamiento 1  |
| `trat2`| Email con nudge de comportamiento 2  |

Diseño: experimento aleatorizado (RCT) con 5.000 clientes muestreados de 500.000.

Documentación de referencia en [`docs/`](docs/):

- [`GUIA_CONCEPTUAL_TECNICA.md`](docs/GUIA_CONCEPTUAL_TECNICA.md) — marco causal, resultados numéricos e interpretación
- `DOE_prueba_tecnica.docx` — diseño del experimento
- `Dic_Variables_Prueba_Tecnica.pdf` — diccionario de variables

## Diccionario de variables

| Variable         | Tipo    | Descripción                          |
|------------------|---------|--------------------------------------|
| `iid`            | ID      | Identificador del cliente            |
| `grupo`          | Factor  | Asignación: `ctrl`, `trat1`, `trat2` |
| `or`             | Binaria | Abrió el email (0/1)                 |
| `ctor`           | Binaria | Clic en botón (0/1)                  |
| `sexo`           | Binaria | Sexo (0=mujer, 1=hombre)              |
| `edad`           | Numérica| Edad (18–99)                         |
| `inve`           | Numérica| Inversión en el banco                |
| `uso_app`        | Binaria | Usa la app del banco (0/1)           |
| `tarjeta_debito` | Binaria | Tiene tarjeta de débito (0/1)        |
| `tipo_tarjeta`   | Factor  | Tipos de tarjeta (1–5)               |
| `formacion`      | Factor  | Nivel educativo (1–5)                |

## Roadmap de notebooks

| Notebook | Contenido |
|----------|-----------|
| `01_load_and_eda.ipynb` | Carga, validación, EDA, balance de randomización |
| `02_basic_experiment_analysis.ipynb` | ATE, tests, regresión, visualizaciones |
| `03_causal_ml_heterogeneity.ipynb` | CATE con EconML/causalml, segmentos |
| `04_data_storytelling.ipynb` | Narrativa para el cliente + export HTML ([`docs/04_data_storytelling.html`](docs/04_data_storytelling.html)) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Estructura del proyecto

```
causal-email-nudge-experiment/
├── data/
│   └── datos_prueba_tecnica.csv
├── docs/
├── notebooks/
├── src/
│   ├── data.py       # Carga y tipado
│   ├── analysis.py   # ATE, regresión, impacto
│   └── causal.py     # CATE con meta-learners
├── requirements.txt
└── README.md
```

## Licencia

Proyecto personal de aprendizaje. Los datos provienen de la prueba técnica BeWay.
