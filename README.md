# Causal Email Nudge Experiment

**A single behavioral nudge nearly 5× the email click-through rate in a randomized experiment.** This project proves that effect causally, quantifies it at scale (~200K extra clicks), and uses Causal ML to pinpoint *which* customers to target — packaged as a reproducible, tested analysis.

It covers the full arc: EDA and randomization balance, classic A/B analysis (ATE), Causal ML for heterogeneous effects (CATE) with **out-of-sample uplift validation**, and an executive data-storytelling narrative.

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
- [`EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) — experiment design brief
- [`VARIABLES.md`](docs/VARIABLES.md) — variable dictionary

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
| `05_uplift_validation.ipynb` | Out-of-sample uplift (Qini/AUUC) and CATE confidence intervals |

## Executed reports

Rendered Markdown versions of the executed notebooks (with charts and tables, viewable directly on GitHub) in [`docs/reports/`](docs/reports/):

- [`01_load_and_eda.md`](docs/reports/01_load_and_eda.md)
- [`02_basic_experiment_analysis.md`](docs/reports/02_basic_experiment_analysis.md)
- [`03_causal_ml_heterogeneity.md`](docs/reports/03_causal_ml_heterogeneity.md)
- [`04_data_storytelling.md`](docs/reports/04_data_storytelling.md)
- [`05_uplift_validation.md`](docs/reports/05_uplift_validation.md)

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
│   ├── mediation.py  # Funnel decomposition or → ctor
│   └── uplift.py     # Out-of-sample uplift (Qini/AUUC) + CATE confidence intervals
├── tests/
├── requirements.txt
└── README.md
```

## License

Code released under the [MIT License](LICENSE).

Personal learning project. The dataset comes from a data-analysis technical assignment and is included for demonstration purposes only.
