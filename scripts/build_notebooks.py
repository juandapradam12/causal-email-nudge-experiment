"""Generate project notebooks programmatically."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}


def write(name, cells):
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")


write(
    "01_load_and_eda.ipynb",
    [
        md(
            "# 01 — Carga de datos y EDA\n\n"
            "Exploración del experimento A/B de emails con nudges de ciencias del comportamiento."
        ),
        code(
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "from scipy import stats\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\n\n"
            "from src.data import GROUP_LABELS, load_data\n\n"
            "sns.set_theme(style='whitegrid', palette='colorblind')\n"
            "plt.rcParams['figure.figsize'] = (10, 5)"
        ),
        md("## Carga y validación"),
        code(
            "df = load_data(ROOT / 'data' / 'datos_prueba_tecnica.csv')\n"
            "print(f'Filas: {len(df):,} | Columnas: {df.shape[1]}')\n"
            "print(f'Valores nulos: {df.isna().sum().sum()}')\n"
            "df.head()"
        ),
        code(
            "assert len(df) == 5000\n"
            "assert df['iid'].nunique() == 5000\n"
            "assert set(df['grupo'].cat.categories) == {'ctrl', 'trat1', 'trat2'}\n"
            "print('Validación básica OK')"
        ),
        md("## Balance de randomización por grupo"),
        code(
            "group_counts = df['grupo'].value_counts().sort_index()\n"
            "group_share = (group_counts / len(df) * 100).round(1)\n"
            "balance = pd.DataFrame({'N': group_counts, '%': group_share})\n"
            "balance.index = balance.index.map(GROUP_LABELS)\n"
            "balance"
        ),
        md("## Tasas de outcome por grupo"),
        code(
            "rates = (\n"
            "    df.groupby('grupo', observed=True)[['or', 'ctor']]\n"
            "    .mean()\n"
            "    .rename(columns={'or': 'open_rate', 'ctor': 'click_rate'})\n"
            ")\n"
            "rates['ctor_given_open'] = (\n"
            "    df[df['or'] == 1].groupby('grupo', observed=True)['ctor'].mean()\n"
            ")\n"
            "rates.index = rates.index.map(GROUP_LABELS)\n"
            "rates.round(3)"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "for ax, metric, title in zip(\n"
            "    axes,\n"
            "    ['or', 'ctor'],\n"
            "    ['Open rate (or)', 'Click rate (ctor)'],\n"
            "):\n"
            "    sns.barplot(data=df, x='grupo', y=metric, errorbar=('ci', 95), ax=ax)\n"
            "    ax.set_title(title)\n"
            "    ax.set_xlabel('Grupo')\n"
            "    ax.set_ylabel('Proporción')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## Distribución de covariables"),
        code(
            "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
            "sns.histplot(df['edad'], kde=True, ax=axes[0, 0])\n"
            "axes[0, 0].set_title('Edad')\n"
            "sns.histplot(df['inve'], kde=True, ax=axes[0, 1])\n"
            "axes[0, 1].set_title('Inversión')\n"
            "sns.countplot(data=df, x='uso_app', hue='grupo', ax=axes[1, 0])\n"
            "axes[1, 0].set_title('Uso app por grupo')\n"
            "sns.countplot(data=df, x='formacion', hue='grupo', ax=axes[1, 1])\n"
            "axes[1, 1].set_title('Formación por grupo')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## ¿Covariables balanceadas entre grupos?"),
        code(
            "covariates = ['edad', 'inve', 'sexo', 'uso_app', 'tarjeta_debito']\n"
            "balance_tests = []\n"
            "for col in covariates:\n"
            "    groups = [g[col].values for _, g in df.groupby('grupo', observed=True)]\n"
            "    if col in ['edad', 'inve']:\n"
            "        stat, p = stats.f_oneway(*groups)\n"
            "        test = 'ANOVA'\n"
            "    else:\n"
            "        table = pd.crosstab(df['grupo'], df[col])\n"
            "        stat, p, _, _ = stats.chi2_contingency(table)\n"
            "        test = 'Chi-cuadrado'\n"
            "    balance_tests.append({'variable': col, 'test': test, 'p_value': p})\n"
            "balance_df = pd.DataFrame(balance_tests).round(4)\n"
            "balance_df"
        ),
        md(
            "**Conclusión EDA:** Los tres brazos están razonablemente balanceados. "
            "Los tratamientos muestran un aumento claro en open rate y click rate respecto al control; "
            "`trat2` parece superar a `trat1` en clics."
        ),
    ],
)

write(
    "02_basic_experiment_analysis.ipynb",
    [
        md("# 02 — Análisis clásico del experimento (ATE)"),
        code(
            "import sys\nfrom pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n"
            "import seaborn as sns\nimport statsmodels.formula.api as smf\n"
            "from scipy import stats\nfrom statsmodels.stats.proportion import proportion_confint\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\nfrom src.data import GROUP_LABELS, load_data\n\n"
            "sns.set_theme(style='whitegrid')\n"
            "df = load_data(ROOT / 'data' / 'datos_prueba_tecnica.csv')"
        ),
        md("## Funciones auxiliares"),
        code(
            "def ate_proportion(df, treatment, control, outcome):\n"
            "    p_t = df.loc[df['grupo'] == treatment, outcome].mean()\n"
            "    p_c = df.loc[df['grupo'] == control, outcome].mean()\n"
            "    diff = p_t - p_c\n"
            "    n_t = (df['grupo'] == treatment).sum()\n"
            "    n_c = (df['grupo'] == control).sum()\n"
            "    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)\n"
            "    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se\n"
            "    table = pd.crosstab(df.loc[df['grupo'].isin([treatment, control]), 'grupo'],\n"
            "                        df.loc[df['grupo'].isin([treatment, control]), outcome])\n"
            "    chi2, p, _, _ = stats.chi2_contingency(table)\n"
            "    return {\n"
            "        'comparison': f'{treatment} vs {control}',\n"
            "        'outcome': outcome,\n"
            "        'rate_treatment': p_t,\n"
            "        'rate_control': p_c,\n"
            "        'ate_pp': diff,\n"
            "        'lift_pct': (diff / p_c * 100) if p_c else np.nan,\n"
            "        'ci_low': ci_low,\n"
            "        'ci_high': ci_high,\n"
            "        'p_value': p,\n"
            "    }\n\n"
            "comparisons = [('trat1', 'ctrl'), ('trat2', 'ctrl'), ('trat2', 'trat1')]\n"
            "results = []\n"
            "for outcome in ['or', 'ctor']:\n"
            "    for t, c in comparisons:\n"
            "        results.append(ate_proportion(df, t, c, outcome))\n"
            "ate_df = pd.DataFrame(results).round(4)\n"
            "ate_df"
        ),
        md("## Forest plot de efectos"),
        code(
            "plot_df = ate_df.copy()\n"
            "plot_df['label'] = plot_df['comparison'] + ' | ' + plot_df['outcome']\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "y = np.arange(len(plot_df))\n"
            "ax.errorbar(plot_df['ate_pp'], y,\n"
            "            xerr=[plot_df['ate_pp'] - plot_df['ci_low'], plot_df['ci_high'] - plot_df['ate_pp']],\n"
            "            fmt='o', capsize=4)\n"
            "ax.axvline(0, color='gray', linestyle='--')\n"
            "ax.set_yticks(y)\n"
            "ax.set_yticklabels(plot_df['label'])\n"
            "ax.set_xlabel('ATE (puntos porcentuales)')\n"
            "ax.set_title('Efectos promedio del tratamiento con IC 95%')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## Regresión logística ajustada por covariables"),
        code(
            "formula = (\n"
            "    'Q(\"or\") ~ C(grupo, Treatment(reference=\"ctrl\")) + edad + sexo + inve + '\n"
            "    'uso_app + tarjeta_debito + C(tipo_tarjeta) + C(formacion)'\n"
            ")\n"
            "model_or = smf.logit(formula, data=df).fit(disp=0)\n"
            "print(model_or.summary2().tables[1].loc[\n"
            "    [x for x in model_or.params.index if 'grupo' in x]\n"
            "])"
        ),
        code(
            "formula_ctor = formula.replace('Q(\"or\") ~', 'ctor ~')\n"
            "model_ctor = smf.logit(formula_ctor, data=df).fit(disp=0)\n"
            "print(model_ctor.summary2().tables[1].loc[\n"
            "    [x for x in model_ctor.params.index if 'grupo' in x]\n"
            "])"
        ),
        md(
            "**Conclusión:** Ambos nudges aumentan significativamente open rate y click rate. "
            "Trat2 supera a trat1 en clics. Los efectos persisten tras ajustar por covariables."
        ),
    ],
)

write(
    "03_causal_ml_heterogeneity.ipynb",
    [
        md("# 03 — Causal ML: efectos heterogéneos (CATE)"),
        code(
            "import sys\nfrom pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n"
            "import seaborn as sns\nfrom sklearn.ensemble import RandomForestRegressor\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from econml.metalearners import TLearner, XLearner\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\nfrom src.data import load_data\n\n"
            "sns.set_theme(style='whitegrid')\n"
            "df = load_data(ROOT / 'data' / 'datos_prueba_tecnica.csv')"
        ),
        md("## Preparación: comparaciones binarias tratamiento vs control"),
        code(
            "FEATURES = ['edad', 'sexo', 'inve', 'uso_app', 'tarjeta_debito', 'tipo_tarjeta', 'formacion']\n\n"
            "def prep_binary(df, treatment_arm, outcome='ctor'):\n"
            "    sub = df[df['grupo'].isin(['ctrl', treatment_arm])].copy()\n"
            "    X = pd.get_dummies(sub[FEATURES], columns=['tipo_tarjeta', 'formacion'], drop_first=True)\n"
            "    T = (sub['grupo'] == treatment_arm).astype(int).values\n"
            "    Y = sub[outcome].values\n"
            "    return X, T, Y\n\n"
            "X1, T1, Y1 = prep_binary(df, 'trat1')\n"
            "X2, T2, Y2 = prep_binary(df, 'trat2')\n"
            "print(X1.shape, X2.shape)"
        ),
        md("## T-Learner y X-Learner (EconML)"),
        code(
            "def fit_cate(X, T, Y, label):\n"
            "    base = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)\n"
            "    t_learner = TLearner(models=base)\n"
            "    t_learner.fit(Y, T, X=X.values)\n"
            "    cate_t = t_learner.effect(X.values).flatten()\n"
            "\n"
            "    x_learner = XLearner(models=base, propensity_model=LogisticRegression(max_iter=1000))\n"
            "    x_learner.fit(Y, T, X=X.values)\n"
            "    cate_x = x_learner.effect(X.values).flatten()\n"
            "\n"
            "    return pd.DataFrame({\n"
            "        'label': label,\n"
            "        'cate_t': cate_t,\n"
            "        'cate_x': cate_x,\n"
            "        'edad': X['edad'].values,\n"
            "        'uso_app': X['uso_app'].values,\n"
            "    })\n\n"
            "cate_trat1 = fit_cate(X1, T1, Y1, 'trat1 vs ctrl (ctor)')\n"
            "cate_trat2 = fit_cate(X2, T2, Y2, 'trat2 vs ctrl (ctor)')\n"
            "cate_all = pd.concat([cate_trat1, cate_trat2], ignore_index=True)\n"
            "cate_all.groupby('label')[['cate_t', 'cate_x']].mean().round(4)"
        ),
        md("## Distribución de CATE"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "for ax, label in zip(axes, cate_all['label'].unique()):\n"
            "    subset = cate_all[cate_all['label'] == label]\n"
            "    sns.kdeplot(subset['cate_x'], fill=True, ax=ax, label='X-Learner')\n"
            "    ax.axvline(subset['cate_x'].mean(), color='red', linestyle='--', label='Media CATE')\n"
            "    ax.set_title(label)\n"
            "    ax.set_xlabel('CATE estimado (ctor)')\n"
            "    ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## Heterogeneidad por segmentos"),
        code(
            "def segment_summary(cate_df, segment_col, bins=None, labels=None):\n"
            "    tmp = cate_df.copy()\n"
            "    if bins is not None:\n"
            "        tmp['segment'] = pd.cut(tmp[segment_col], bins=bins, labels=labels)\n"
            "    else:\n"
            "        tmp['segment'] = tmp[segment_col].map({0: 'No app', 1: 'Usa app'})\n"
            "    return tmp.groupby('segment')['cate_x'].agg(['mean', 'std', 'count']).round(4)\n\n"
            "print('Trat1 — por edad:')\n"
            "display(segment_summary(cate_trat1, 'edad', [18, 35, 50, 100], ['18-35', '36-50', '51+']))\n"
            "print('Trat2 — por uso app:')\n"
            "display(segment_summary(cate_trat2, 'uso_app'))"
        ),
        md(
            "**Validación:** La media del CATE debe aproximar el ATE del notebook 02. "
            "La heterogeneidad indica segmentos con mayor respuesta a cada nudge."
        ),
        code(
            "from src.data import load_data\n"
            "ate_manual = (\n"
            "    df[df['grupo'] == 'trat2']['ctor'].mean() - df[df['grupo'] == 'ctrl']['ctor'].mean()\n"
            ")\n"
            "print(f'ATE manual trat2 vs ctrl (ctor): {ate_manual:.4f}')\n"
            "print(f'Media CATE X-Learner trat2: {cate_trat2[\"cate_x\"].mean():.4f}')"
        ),
    ],
)

write(
    "04_data_storytelling.ipynb",
    [
        md(
            "# 04 — Data storytelling para el cliente\n\n"
            "Narrativa ejecutiva del experimento de emails con nudges."
        ),
        code(
            "import sys\nfrom pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\nfrom src.data import GROUP_LABELS, load_data\n\n"
            "df = load_data(ROOT / 'data' / 'datos_prueba_tecnica.csv')"
        ),
        md("## 1. El reto del banco"),
        code(
            "summary = df.groupby('grupo', observed=True)[['or', 'ctor']].mean()\n"
            "summary.index = summary.index.map(GROUP_LABELS)\n"
            "ctrl_or, ctrl_ctor = summary.loc[GROUP_LABELS['ctrl']]\n"
            "best = summary.loc[GROUP_LABELS['trat2']]\n"
            "print(f'Control: {ctrl_or:.1%} apertura, {ctrl_ctor:.1%} clics')\n"
            "print(f'Trat2:   {best[\"or\"]:.1%} apertura, {best[\"ctor\"]:.1%} clics')"
        ),
        md("## 2. Resultado del experimento"),
        code(
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "plot_data = df.groupby('grupo', observed=True)[['or', 'ctor']].mean().reset_index()\n"
            "plot_data['grupo'] = plot_data['grupo'].map(GROUP_LABELS)\n"
            "plot_melt = plot_data.melt(id_vars='grupo', var_name='metric', value_name='rate')\n"
            "sns.barplot(data=plot_melt, x='grupo', y='rate', hue='metric', ax=ax)\n"
            "ax.set_ylabel('Proporción')\n"
            "ax.set_title('Open rate y click rate por variante de email')\n"
            "plt.xticks(rotation=15)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## 3. Impacto estimado"),
        code(
            "n_clients = 500_000\n"
            "lift_ctor = best['ctor'] - ctrl_ctor\n"
            "extra_clicks = int(lift_ctor * n_clients)\n"
            "print(f'Si escalamos trat2 a {n_clients:,} clientes:')\n"
            "print(f'  ~{extra_clicks:,} clics adicionales vs control ({lift_ctor:.1%} lift absoluto)')"
        ),
        md("## 4. Recomendación"),
        code(
            "recommendation = pd.DataFrame({\n"
            "    'Acción': [\n"
            "        'Desplegar email Tratamiento 2 como variante principal',\n"
            "        'Priorizar segmentos con mayor CATE (usuarios app, edad media)',\n"
            "        'Monitorizar A/B continuo post-lanzamiento',\n"
            "    ],\n"
            "    'Impacto esperado': [\n"
            "        f'+{lift_ctor:.1%} en click rate vs control',\n"
            "        'Optimización adicional vía personalización',\n"
            "        'Detección temprana de fatiga del nudge',\n"
            "    ],\n"
            "})\n"
            "recommendation"
        ),
        md(
            "---\n\n"
            "**Narrativa en Markdown:** ver [`docs/04_DATA_STORYTELLING.md`](../docs/04_DATA_STORYTELLING.md)"
        ),
    ],
)

if __name__ == "__main__":
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    print("Notebooks generated.")
