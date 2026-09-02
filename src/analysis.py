"""Classical experiment analysis: ATE, tests, and adjusted regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def ate_proportion(
    df: pd.DataFrame,
    treatment: str,
    control: str,
    outcome: str,
) -> dict:
    """Difference-in-proportions ATE with Wald CI and chi-square test."""
    mask = df["grupo"].isin([treatment, control])
    sub = df.loc[mask]

    p_t = sub.loc[sub["grupo"] == treatment, outcome].mean()
    p_c = sub.loc[sub["grupo"] == control, outcome].mean()
    diff = p_t - p_c

    n_t = (sub["grupo"] == treatment).sum()
    n_c = (sub["grupo"] == control).sum()
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se

    table = pd.crosstab(sub["grupo"], sub[outcome])
    _, p_value, _, _ = stats.chi2_contingency(table)

    return {
        "comparison": f"{treatment} vs {control}",
        "outcome": outcome,
        "rate_treatment": p_t,
        "rate_control": p_c,
        "ate_pp": diff,
        "lift_pct": (diff / p_c * 100) if p_c else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def all_ate_comparisons(
    df: pd.DataFrame,
    outcomes: list[str] | None = None,
    comparisons: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Run ATE for all treatment pairs and outcomes."""
    outcomes = outcomes or ["or", "ctor"]
    comparisons = comparisons or [
        ("trat1", "ctrl"),
        ("trat2", "ctrl"),
        ("trat2", "trat1"),
    ]
    rows = [
        ate_proportion(df, t, c, outcome)
        for outcome in outcomes
        for t, c in comparisons
    ]
    return pd.DataFrame(rows)


def logistic_treatment_effects(
    df: pd.DataFrame,
    outcome: str,
    covariates: str = "edad + sexo + inve + uso_app + tarjeta_debito + C(tipo_tarjeta) + C(formacion)",
) -> pd.DataFrame:
    """Adjusted log-odds and odds ratios for treatment dummies."""
    if outcome == "or":
        formula = f'Q("or") ~ C(grupo, Treatment(reference="ctrl")) + {covariates}'
    else:
        formula = f"{outcome} ~ C(grupo, Treatment(reference=\"ctrl\")) + {covariates}"

    model = smf.logit(formula, data=df).fit(disp=0)
    rows = []
    for name in model.params.index:
        if "grupo" not in name:
            continue
        rows.append(
            {
                "term": name,
                "coef_log_odds": model.params[name],
                "odds_ratio": np.exp(model.params[name]),
                "p_value": model.pvalues[name],
                "ci_low": np.exp(model.conf_int().loc[name, 0]),
                "ci_high": np.exp(model.conf_int().loc[name, 1]),
            }
        )
    return pd.DataFrame(rows)


def scale_impact(
    ate_pp: float,
    population_size: int,
    outcome_label: str = "conversions",
) -> dict:
    """Translate an absolute lift to population-level impact."""
    return {
        "population_size": population_size,
        "absolute_lift": ate_pp,
        "extra_outcomes": int(ate_pp * population_size),
        "outcome_label": outcome_label,
    }
