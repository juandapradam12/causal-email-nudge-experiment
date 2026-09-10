"""Out-of-sample uplift validation and CATE uncertainty.

The meta-learners / DML estimates in :mod:`src.causal` tell us *how large* the
effect is on average, but not whether a model **ranks** customers by
responsiveness well enough to target them. This module answers that with:

- an out-of-sample (train/test) protocol that fits CATE on train and scores
  test customers, then measures ranking quality with **Qini** and **AUUC**
  (via ``causalml.metrics``); and
- **confidence intervals** for the CATE from ``CausalForestDML`` so segment
  differences are not over-interpreted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from causalml.metrics import auuc_score, get_cumgain, get_qini, qini_score
from econml.dml import CausalForestDML
from econml.metalearners import TLearner, XLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.causal import DEFAULT_FEATURES, prep_binary_comparison

OUTCOME_COL = "y"
TREATMENT_COL = "w"


def _rf(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )


def fit_predict_cate_oos(
    X_train: pd.DataFrame,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    learners: tuple[str, ...] = ("t", "x", "cf"),
    n_estimators: int = 200,
    random_state: int = 42,
    dml_cv: int = 3,
) -> dict[str, np.ndarray]:
    """Fit the requested learners on train and predict CATE on test (tau_hat).

    Returns a mapping ``{model_name: tau_test}``. Supported learners: ``t``
    (T-learner), ``x`` (X-learner), ``cf`` (CausalForestDML).
    """
    xtr, xte = X_train.values, X_test.values
    out: dict[str, np.ndarray] = {}

    if "t" in learners:
        t = TLearner(models=_rf(n_estimators, random_state))
        t.fit(Y_train, T_train, X=xtr)
        out["cate_t"] = t.effect(xte).flatten()

    if "x" in learners:
        x = XLearner(
            models=_rf(n_estimators, random_state),
            propensity_model=LogisticRegression(max_iter=5000),
        )
        x.fit(Y_train, T_train, X=xtr)
        out["cate_x"] = x.effect(xte).flatten()

    if "cf" in learners:
        cf = CausalForestDML(
            model_y=_rf(max(50, n_estimators // 2), random_state),
            model_t=_rf(max(50, n_estimators // 2), random_state),
            discrete_treatment=True,
            discrete_outcome=True,
            n_estimators=max(4, n_estimators // 4 * 4),
            cv=dml_cv,
            random_state=random_state,
        )
        cf.fit(Y_train, T_train, X=xtr)
        out["cate_cf"] = cf.effect(xte).flatten()

    return out


def scored_frame(
    Y_test: np.ndarray,
    T_test: np.ndarray,
    tau_by_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble the frame expected by ``causalml.metrics`` (y, w, model cols)."""
    frame = pd.DataFrame({OUTCOME_COL: np.asarray(Y_test, dtype=float),
                          TREATMENT_COL: np.asarray(T_test, dtype=int)})
    for name, tau in tau_by_model.items():
        frame[name] = np.asarray(tau, dtype=float)
    return frame


def uplift_scores(
    scored: pd.DataFrame,
    model_cols: list[str] | None = None,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Qini and AUUC for each model column, ranked best-first.

    Higher is better; a random-targeting model scores ~0 on the normalized
    Qini. Returns one row per model with columns ``qini`` and ``auuc``.
    """
    model_cols = model_cols or [
        c for c in scored.columns if c not in (OUTCOME_COL, TREATMENT_COL)
    ]
    rows = []
    for col in model_cols:
        # causalml treats every column other than outcome/treatment (and the
        # optional ground-truth ``treatment_effect_col``, which we don't have)
        # as a model to rank by, so pass exactly one model column at a time.
        sub = scored[[OUTCOME_COL, TREATMENT_COL, col]]
        q = qini_score(
            sub, outcome_col=OUTCOME_COL, treatment_col=TREATMENT_COL,
            normalize=normalize,
        )
        a = auuc_score(
            sub, outcome_col=OUTCOME_COL, treatment_col=TREATMENT_COL,
            normalize=normalize,
        )
        rows.append({"model": col, "qini": float(q[col]), "auuc": float(a[col])})
    return (
        pd.DataFrame(rows)
        .sort_values("qini", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_uplift(
    df: pd.DataFrame,
    treatment_arm: str,
    outcome: str = "ctor",
    *,
    features: list[str] | None = None,
    learners: tuple[str, ...] = ("t", "x", "cf"),
    test_size: float = 0.3,
    n_estimators: int = 200,
    random_state: int = 42,
    dml_cv: int = 3,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end out-of-sample uplift validation for one treatment arm.

    Splits control + ``treatment_arm`` into train/test (stratified by
    treatment), fits the learners on train, scores the held-out test set, and
    returns ``(scores, scored)`` where ``scores`` holds Qini/AUUC per model and
    ``scored`` is the test frame with predicted uplift columns.
    """
    features = features or DEFAULT_FEATURES
    X, T, Y = prep_binary_comparison(df, treatment_arm, outcome, features)

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=test_size, random_state=random_state, stratify=T
    )
    tau_by_model = fit_predict_cate_oos(
        X_train, T_train, Y_train, X_test,
        learners=learners, n_estimators=n_estimators,
        random_state=random_state, dml_cv=dml_cv,
    )
    scored = scored_frame(Y_test, T_test, tau_by_model)
    scores = uplift_scores(scored, normalize=normalize)
    return scores, scored


def uplift_curve_points(
    scored: pd.DataFrame,
    model_col: str,
    *,
    kind: str = "qini",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Curve points for plotting (``kind='qini'`` or ``'gain'``)."""
    getter = get_qini if kind == "qini" else get_cumgain
    return getter(
        scored[[OUTCOME_COL, TREATMENT_COL, model_col]],
        outcome_col=OUTCOME_COL,
        treatment_col=TREATMENT_COL,
        random_seed=random_seed,
    )


def cate_with_confidence(
    df: pd.DataFrame,
    treatment_arm: str,
    outcome: str = "ctor",
    *,
    features: list[str] | None = None,
    n_estimators: int = 200,
    dml_cv: int = 3,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Per-customer CATE with a (1-alpha) confidence interval from CausalForest.

    Returns the covariate frame plus ``cate``, ``ci_lower``, ``ci_upper`` and a
    boolean ``significant`` (the interval excludes zero).
    """
    features = features or DEFAULT_FEATURES
    X, T, Y = prep_binary_comparison(df, treatment_arm, outcome, features)

    cf = CausalForestDML(
        model_y=_rf(max(50, n_estimators // 2), random_state),
        model_t=_rf(max(50, n_estimators // 2), random_state),
        discrete_treatment=True,
        discrete_outcome=True,
        n_estimators=max(4, n_estimators // 4 * 4),
        cv=dml_cv,
        random_state=random_state,
    )
    cf.fit(Y, T, X=X.values)
    point = np.asarray(cf.effect(X.values)).reshape(-1)
    lower, upper = cf.effect_interval(X.values, alpha=alpha)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    frame = X.copy()
    frame["cate"] = point
    frame["ci_lower"] = lower
    frame["ci_upper"] = upper
    frame["significant"] = (lower > 0) | (upper < 0)
    return frame


def summarize_cate_ci(cate_ci: pd.DataFrame) -> pd.Series:
    """Headline numbers for a CATE-with-CI frame."""
    width = cate_ci["ci_upper"] - cate_ci["ci_lower"]
    return pd.Series(
        {
            "n": float(len(cate_ci)),
            "mean_cate": float(cate_ci["cate"].mean()),
            "mean_ci_width": float(width.mean()),
            "frac_significant": float(cate_ci["significant"].mean()),
        }
    )
