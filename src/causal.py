"""Causal ML utilities: CATE estimation with meta-learners and DML."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


DEFAULT_FEATURES = [
    "edad",
    "sexo",
    "inve",
    "uso_app",
    "tarjeta_debito",
    "tipo_tarjeta",
    "formacion",
]


@dataclass
class CATEEstimates:
    """Container for individual-level treatment effect estimates."""

    label: str
    cate_t: np.ndarray
    cate_x: np.ndarray
    cate_s: np.ndarray
    features: pd.DataFrame
    treatment: np.ndarray
    outcome: np.ndarray
    cate_dml: np.ndarray | None = None
    cate_cf: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        frame = self.features.copy()
        frame["cate_t"] = self.cate_t
        frame["cate_x"] = self.cate_x
        frame["cate_s"] = self.cate_s
        if self.cate_dml is not None:
            frame["cate_dml"] = self.cate_dml
        if self.cate_cf is not None:
            frame["cate_cf"] = self.cate_cf
        frame["label"] = self.label
        return frame

    def summary(self) -> pd.Series:
        values = {
            "mean_cate_t": self.cate_t.mean(),
            "mean_cate_x": self.cate_x.mean(),
            "mean_cate_s": self.cate_s.mean(),
            "std_cate_x": self.cate_x.std(),
            "min_cate_x": self.cate_x.min(),
            "max_cate_x": self.cate_x.max(),
        }
        if self.cate_dml is not None:
            values["mean_cate_dml"] = self.cate_dml.mean()
        if self.cate_cf is not None:
            values["mean_cate_cf"] = self.cate_cf.mean()
        return pd.Series(values)


def prep_binary_comparison(
    df: pd.DataFrame,
    treatment_arm: str,
    outcome: str = "ctor",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Filter to control + one treatment arm and encode covariates."""
    features = features or DEFAULT_FEATURES
    sub = df[df["grupo"].isin(["ctrl", treatment_arm])].copy()
    X = pd.get_dummies(
        sub[features],
        columns=["tipo_tarjeta", "formacion"],
        drop_first=True,
    )
    T = (sub["grupo"] == treatment_arm).astype(int).values
    Y = sub[outcome].values
    return X, T, Y


def _base_classifier(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def calibrate_cate_to_ate(
    cate: np.ndarray,
    ate: float,
    method: str = "shift",
) -> np.ndarray:
    """Post-hoc calibration so mean CATE matches ATE while preserving ranking.

    Parameters
    ----------
    cate :
        Individual CATE estimates.
    ate :
        Target ATE (e.g. difference in means from the RCT).
    method :
        ``shift`` — additive: ``cate - mean(cate) + ate`` (preserves gaps).
        ``scale`` — multiplicative: ``cate * (ate / mean(cate))`` if mean ≠ 0.
    """
    cate = np.asarray(cate, dtype=float)
    mean_cate = cate.mean()
    if method == "shift":
        return cate - mean_cate + ate
    if method == "scale":
        if abs(mean_cate) < 1e-12:
            raise ValueError("Cannot scale CATE when mean is ~0; use method='shift'.")
        return cate * (ate / mean_cate)
    raise ValueError(f"Unknown method={method!r}; use 'shift' or 'scale'.")


def fit_cate(
    X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    label: str,
    *,
    n_estimators: int = 200,
    random_state: int = 42,
    include_dml: bool = True,
    include_causal_forest: bool = True,
    dml_cv: int = 3,
    cf_n_estimators: int = 100,
) -> CATEEstimates:
    """Fit S/T/X meta-learners and optional LinearDML / CausalForestDML."""
    base = _base_classifier(n_estimators, random_state)
    x_vals = X.values

    s_learner = SLearner(overall_model=base)
    s_learner.fit(Y, T, X=x_vals)
    cate_s = s_learner.effect(x_vals).flatten()

    t_learner = TLearner(models=base)
    t_learner.fit(Y, T, X=x_vals)
    cate_t = t_learner.effect(x_vals).flatten()

    propensity = LogisticRegression(max_iter=5000, solver="lbfgs")
    x_learner = XLearner(models=base, propensity_model=propensity)
    x_learner.fit(Y, T, X=x_vals)
    cate_x = x_learner.effect(x_vals).flatten()

    cate_dml = None
    if include_dml:
        dml = LinearDML(
            model_y=_base_classifier(n_estimators, random_state),
            model_t=_base_classifier(n_estimators, random_state),
            discrete_treatment=True,
            discrete_outcome=True,
            cv=dml_cv,
            random_state=random_state,
        )
        dml.fit(Y, T, X=x_vals)
        cate_dml = dml.effect(x_vals).flatten()

    cate_cf = None
    if include_causal_forest:
        # CausalForestDML requires n_estimators divisible by subforest_size (default 4).
        n_trees = max(4, int(cf_n_estimators) // 4 * 4)
        cf = CausalForestDML(
            model_y=_base_classifier(max(50, n_estimators // 2), random_state),
            model_t=_base_classifier(max(50, n_estimators // 2), random_state),
            discrete_treatment=True,
            discrete_outcome=True,
            n_estimators=n_trees,
            cv=dml_cv,
            random_state=random_state,
        )
        cf.fit(Y, T, X=x_vals)
        cate_cf = cf.effect(x_vals).flatten()

    return CATEEstimates(
        label=label,
        cate_t=cate_t,
        cate_x=cate_x,
        cate_s=cate_s,
        cate_dml=cate_dml,
        cate_cf=cate_cf,
        features=X,
        treatment=T,
        outcome=Y,
    )


def segment_cate_summary(
    cate_df: pd.DataFrame,
    segment_col: str,
    cate_col: str = "cate_x",
    *,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate CATE by a segment column (continuous or categorical)."""
    tmp = cate_df.copy()
    if bins is not None:
        tmp["segment"] = pd.cut(tmp[segment_col], bins=bins, labels=labels)
    else:
        tmp["segment"] = tmp[segment_col]
    return tmp.groupby("segment", observed=True)[cate_col].agg(["mean", "std", "count"])


def validate_cate_vs_ate(
    df: pd.DataFrame,
    treatment_arm: str,
    outcome: str,
    cate_estimates: CATEEstimates,
) -> pd.DataFrame:
    """Compare mean CATE from each learner to the simple difference-in-means ATE."""
    ate = (
        df.loc[df["grupo"] == treatment_arm, outcome].mean()
        - df.loc[df["grupo"] == "ctrl", outcome].mean()
    )

    metrics = [
        "ATE (diff medias)",
        "Media CATE S-Learner",
        "Media CATE T-Learner",
        "Media CATE X-Learner",
    ]
    values = [
        ate,
        cate_estimates.cate_s.mean(),
        cate_estimates.cate_t.mean(),
        cate_estimates.cate_x.mean(),
    ]
    if cate_estimates.cate_dml is not None:
        metrics.append("Media CATE LinearDML")
        values.append(cate_estimates.cate_dml.mean())
    if cate_estimates.cate_cf is not None:
        metrics.append("Media CATE CausalForestDML")
        values.append(cate_estimates.cate_cf.mean())

    return pd.DataFrame(
        {
            "metric": metrics,
            "value": values,
            "gap_vs_ate": [0.0] + [abs(ate - v) for v in values[1:]],
        }
    )
