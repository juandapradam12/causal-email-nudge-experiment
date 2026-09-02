"""Causal ML utilities: CATE estimation with meta-learners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.metalearners import TLearner, XLearner
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
    features: pd.DataFrame
    treatment: np.ndarray
    outcome: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        frame = self.features.copy()
        frame["cate_t"] = self.cate_t
        frame["cate_x"] = self.cate_x
        frame["label"] = self.label
        return frame

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "mean_cate_t": self.cate_t.mean(),
                "mean_cate_x": self.cate_x.mean(),
                "std_cate_x": self.cate_x.std(),
                "min_cate_x": self.cate_x.min(),
                "max_cate_x": self.cate_x.max(),
            }
        )


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


def fit_cate(
    X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    label: str,
    *,
    n_estimators: int = 200,
    random_state: int = 42,
) -> CATEEstimates:
    """Fit T-Learner and X-Learner for a binary treatment comparison."""
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    x_vals = X.values

    t_learner = TLearner(models=base)
    t_learner.fit(Y, T, X=x_vals)
    cate_t = t_learner.effect(x_vals).flatten()

    propensity = LogisticRegression(max_iter=5000, solver="lbfgs")
    x_learner = XLearner(models=base, propensity_model=propensity)
    x_learner.fit(Y, T, X=x_vals)
    cate_x = x_learner.effect(x_vals).flatten()

    return CATEEstimates(
        label=label,
        cate_t=cate_t,
        cate_x=cate_x,
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
    """Compare mean CATE to the simple difference-in-means ATE."""
    ate = (
        df.loc[df["grupo"] == treatment_arm, outcome].mean()
        - df.loc[df["grupo"] == "ctrl", outcome].mean()
    )
    return pd.DataFrame(
        {
            "metric": ["ATE (diff medias)", "Media CATE T-Learner", "Media CATE X-Learner"],
            "value": [ate, cate_estimates.cate_t.mean(), cate_estimates.cate_x.mean()],
            "gap_vs_ate": [
                0.0,
                abs(ate - cate_estimates.cate_t.mean()),
                abs(ate - cate_estimates.cate_x.mean()),
            ],
        }
    )
