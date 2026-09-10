# Variable dictionary

Columns of `data/datos_prueba_tecnica.csv` (5,000 rows). The column names are kept in their
original Spanish since they are the literal dataset identifiers used throughout the code.

| Variable | Type | Role | Description |
|----------|------|------|-------------|
| `iid` | ID | — | Unique customer identifier |
| `grupo` | Factor | Treatment | Experiment arm: `ctrl`, `trat1`, `trat2` |
| `or` | Binary (0/1) | Intermediate outcome | Opened the email (open rate) |
| `ctor` | Binary (0/1) | Final outcome | Clicked the action button (nested in `or`: `ctor=0` whenever `or=0`) |
| `sexo` | Binary (0/1) | Covariate | Sex (0 = female, 1 = male) |
| `edad` | Numeric | Covariate | Age (18–99) |
| `inve` | Numeric | Covariate | Investment held at the bank |
| `uso_app` | Binary (0/1) | Covariate | Uses the bank app |
| `tarjeta_debito` | Binary (0/1) | Covariate | Holds a debit card |
| `tipo_tarjeta` | Factor | Covariate | Card type (1–5) |
| `formacion` | Factor | Covariate | Education level (1–5) |

Notes:

- **Treatment arms** (`grupo`): `ctrl` is the causal reference (email with no nudge), not "no email".
- **Outcomes**: `or` (opening) and `ctor` (click) form a funnel — a click is only possible after an open.
- **Covariates** are measured pre-treatment, so they are valid for adjustment and for estimating heterogeneous effects (CATE).
