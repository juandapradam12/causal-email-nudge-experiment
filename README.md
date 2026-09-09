# causal-email-nudge-experiment

Causal analysis of an email A/B experiment with behavioral-science nudges. The project covers the technical-assignment brief (EDA, basic analysis, data storytelling) plus an extension with **Causal ML** to estimate heterogeneous effects (CATE).

## Experiment context

A bank tested three email variants to increase:

- **Open rate** (`or`): did the customer open the email?
- **Click-to-open rate** (`ctor`): did they click the call-to-action button?

| Group  | Description |
|--------|-------------|
| `ctrl` | Control email (no nudge) — **causal reference**, not "no email" |
| `trat1`| Email with behavioral nudge 1 |
| `trat2`| Email with behavioral nudge 2 |

Design: randomized controlled trial (RCT) with 5,000 customers sampled from 500,000.

Reference documentation in [`docs/`](docs/):

- [`GUIA_CONCEPTUAL_TECNICA.md`](docs/GUIA_CONCEPTUAL_TECNICA.md) — causal framework, `ctrl` vs `trat`, results and interpretation
- [`CAUSAL_ML.md`](docs/CAUSAL_ML.md) — identification, meta-learners, DML, CausalForest, mediation, validation
- [`04_DATA_STORYTELLING.md`](docs/04_DATA_STORYTELLING.md) — executive narrative for the client
- `DOE_prueba_tecnica.docx` — experiment design
- `Dic_Variables_Prueba_Tecnica.pdf` — variable dictionary

## Variable dictionary

| Variable         | Type      | Description                          |
|------------------|-----------|--------------------------------------|
| `iid`            | ID        | Customer identifier                  |
| `grupo`          | Factor    | Assignment: `ctrl`, `trat1`, `trat2` |
| `or`             | Binary    | Opened the email (0/1)               |
| `ctor`           | Binary    | Clicked the button (0/1)             |
| `sexo`           | Binary    | Sex (0=female, 1=male)               |
| `edad`           | Numeric   | Age (18–99)                          |
| `inve`           | Numeric   | Investment held at the bank          |
| `uso_app`        | Binary    | Uses the bank app (0/1)              |
| `tarjeta_debito` | Binary    | Has a debit card (0/1)               |
| `tipo_tarjeta`   | Factor    | Card type (1–5)                      |
| `formacion`      | Factor    | Education level (1–5)                |

## Notebook roadmap

| Notebook | Contents |
|----------|----------|
| `01_load_and_eda.ipynb` | Loading, validation, EDA, randomization balance |
| `02_basic_experiment_analysis.ipynb` | ATE, tests, regression, visualizations |
| `03_causal_ml_heterogeneity.ipynb` | CATE (S/T/X, LinearDML, CausalForest), funnel mediation |
| `04_data_storytelling.ipynb` | Client narrative → see [`docs/04_DATA_STORYTELLING.md`](docs/04_DATA_STORYTELLING.md) |

## Executed reports

Rendered HTML versions of the notebooks already executed (with charts and tables) in [`docs/reports/`](docs/reports/):

- [`01_load_and_eda.html`](docs/reports/01_load_and_eda.html)
- [`02_basic_experiment_analysis.html`](docs/reports/02_basic_experiment_analysis.html)
- [`03_causal_ml_heterogeneity.html`](docs/reports/03_causal_ml_heterogeneity.html)
- [`04_data_storytelling.html`](docs/reports/04_data_storytelling.html)

To regenerate them: `bash scripts/export_reports.sh`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/               # ATE, mediation, and CATE helpers
jupyter lab
```

## Project structure

```
causal-email-nudge-experiment/
├── data/
│   └── datos_prueba_tecnica.csv
├── docs/
├── notebooks/
├── src/
│   ├── data.py       # Loading and typing
│   ├── analysis.py   # ATE, regression, impact
│   ├── causal.py     # CATE (meta-learners + DML + CausalForest)
│   └── mediation.py  # Funnel decomposition or → ctor
├── tests/
├── requirements.txt
└── README.md
```

## License

Personal learning project. The data comes from a data-analysis technical assignment.
