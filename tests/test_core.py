"""Unit tests for analysis, mediation, and causal helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    all_ate_comparisons,
    ate_proportion,
    logistic_treatment_effects,
    scale_impact,
)
from src.causal import (
    CATEEstimates,
    calibrate_cate_to_ate,
    fit_cate,
    prep_binary_comparison,
    segment_cate_summary,
    validate_cate_vs_ate,
)
from src.data import load_data, treatment_dummies
from src.mediation import all_funnel_mediations, funnel_mediation, funnel_rates


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


def test_load_data_shape_and_groups(df: pd.DataFrame) -> None:
    assert len(df) == 5000
    assert df["iid"].nunique() == 5000
    assert set(df["grupo"].unique()) == {"ctrl", "trat1", "trat2"}
    assert df[["or", "ctor"]].isna().sum().sum() == 0


def test_ctor_nested_in_or(df: pd.DataFrame) -> None:
    """If the email was not opened, click must be zero."""
    assert (df.loc[df["or"] == 0, "ctor"] == 0).all()


def test_ate_proportion_trat2_ctor(df: pd.DataFrame) -> None:
    result = ate_proportion(df, "trat2", "ctrl", "ctor")
    assert result["ate_pp"] == pytest.approx(0.4015, abs=0.01)
    assert result["p_value"] < 0.001
    assert result["ci_low"] < result["ate_pp"] < result["ci_high"]


def test_all_ate_comparisons_shape(df: pd.DataFrame) -> None:
    ate_df = all_ate_comparisons(df)
    assert len(ate_df) == 6  # 2 outcomes × 3 pairs
    assert {"ate_pp", "p_value", "comparison", "outcome"}.issubset(ate_df.columns)


def test_scale_impact() -> None:
    impact = scale_impact(0.4, 500_000, outcome_label="clics")
    assert impact["extra_outcomes"] == 200_000
    assert impact["outcome_label"] == "clics"


def test_funnel_rates(df: pd.DataFrame) -> None:
    rates = funnel_rates(df, "ctrl")
    assert 0 < rates["open_rate"] < 1
    assert rates["click_rate"] <= rates["open_rate"]
    assert 0 <= rates["click_given_open"] <= 1


def test_funnel_mediation_reconstructs_ate(df: pd.DataFrame) -> None:
    med = funnel_mediation(df, "trat2", "ctrl").iloc[0]
    assert med["reconstructed"] == pytest.approx(med["ate_ctor"], abs=1e-10)
    assert med["share_via_open"] + med["share_via_conversion"] == pytest.approx(1.0, abs=1e-10)
    # trat2 vs ctrl: both paths should contribute positively
    assert med["effect_via_open"] > 0
    assert med["effect_via_conversion"] > 0


def test_funnel_mediation_trat2_vs_trat1_is_conversion(df: pd.DataFrame) -> None:
    """trat2 vs trat1: nearly identical open rates → effect is almost all conversion."""
    med = funnel_mediation(df, "trat2", "trat1").iloc[0]
    assert abs(med["delta_or"]) < 0.01
    assert med["share_via_conversion"] > 0.95


def test_funnel_mediation_trat2_conversion_dominates(df: pd.DataFrame) -> None:
    """trat2 vs ctrl: both paths help, but post-open conversion dominates."""
    med = funnel_mediation(df, "trat2", "ctrl").iloc[0]
    assert med["share_via_conversion"] > med["share_via_open"]
    assert med["share_via_conversion"] == pytest.approx(0.76, abs=0.05)


def test_all_funnel_mediations(df: pd.DataFrame) -> None:
    table = all_funnel_mediations(df)
    assert len(table) == 3


def test_prep_binary_comparison(df: pd.DataFrame) -> None:
    X, T, Y = prep_binary_comparison(df, "trat2", "ctor")
    assert len(X) == len(T) == len(Y)
    assert set(np.unique(T)) == {0, 1}
    assert X.shape[1] >= 7


def test_calibrate_cate_shift() -> None:
    cate = np.array([0.1, 0.2, 0.3])
    calibrated = calibrate_cate_to_ate(cate, ate=0.5, method="shift")
    assert calibrated.mean() == pytest.approx(0.5)
    # ranking preserved
    assert np.argsort(calibrated).tolist() == np.argsort(cate).tolist()


def test_calibrate_cate_scale() -> None:
    cate = np.array([0.1, 0.2, 0.3])
    calibrated = calibrate_cate_to_ate(cate, ate=0.4, method="scale")
    assert calibrated.mean() == pytest.approx(0.4)


def test_validate_cate_vs_ate_structure(df: pd.DataFrame) -> None:
    from src.causal import CATEEstimates

    X, T, Y = prep_binary_comparison(df, "trat2", "ctor")
    n = len(Y)
    fake = CATEEstimates(
        label="fake",
        cate_t=np.full(n, 0.2),
        cate_x=np.full(n, 0.19),
        cate_s=np.full(n, 0.21),
        cate_dml=np.full(n, 0.16),
        cate_cf=np.full(n, 0.18),
        features=X,
        treatment=T,
        outcome=Y,
    )
    table = validate_cate_vs_ate(df, "trat2", "ctor", fake)
    assert table.iloc[0]["metric"] == "ATE (diff medias)"
    assert len(table) == 6
    assert table.iloc[0]["gap_vs_ate"] == 0.0


def test_treatment_dummies(df: pd.DataFrame) -> None:
    dummies = treatment_dummies(df)
    assert {"grupo_ctrl", "grupo_trat1", "grupo_trat2"}.issubset(dummies.columns)
    assert len(dummies) == len(df)
    # every client belongs to exactly one arm
    assert (dummies.sum(axis=1) == 1).all()


def test_calibrate_cate_scale_zero_mean_raises() -> None:
    cate = np.array([-1.0, 1.0])  # mean == 0 -> scaling is undefined
    with pytest.raises(ValueError):
        calibrate_cate_to_ate(cate, ate=0.5, method="scale")


def test_calibrate_cate_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        calibrate_cate_to_ate(np.array([0.1, 0.2]), ate=0.5, method="bogus")


@pytest.mark.parametrize("outcome", ["or", "ctor"])
def test_logistic_treatment_effects(df: pd.DataFrame, outcome: str) -> None:
    res = logistic_treatment_effects(df, outcome)
    # two treatment dummies vs the ctrl reference
    assert len(res) == 2
    assert {
        "term",
        "coef_log_odds",
        "odds_ratio",
        "p_value",
        "ci_low",
        "ci_high",
    }.issubset(res.columns)
    assert res["term"].str.contains("grupo").all()
    assert (res["odds_ratio"] > 0).all()
    assert (res["ci_low"] <= res["odds_ratio"]).all()
    assert (res["odds_ratio"] <= res["ci_high"]).all()


def test_segment_cate_summary_continuous_and_categorical() -> None:
    cate_df = pd.DataFrame(
        {
            "edad": [20, 30, 40, 60, 70],
            "uso_app": [0, 1, 0, 1, 1],
            "cate_x": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    binned = segment_cate_summary(
        cate_df, "edad", bins=[18, 35, 50, 100], labels=["18-35", "36-50", "51+"]
    )
    assert {"mean", "std", "count"}.issubset(binned.columns)
    assert binned.loc["18-35", "count"] == 2
    assert binned.loc["51+", "count"] == 2

    categorical = segment_cate_summary(cate_df, "uso_app")
    assert set(categorical.index) == {0, 1}
    assert categorical.loc[1, "count"] == 3


@pytest.fixture(scope="module")
def small_binary(df: pd.DataFrame):
    """Small balanced ctrl-vs-trat2 slice to keep model-fitting tests fast."""
    sub = pd.concat(
        [
            df[df["grupo"] == "ctrl"].head(250),
            df[df["grupo"] == "trat2"].head(250),
        ]
    )
    return prep_binary_comparison(sub, "trat2", "ctor")


def test_fit_cate_metalearners_only(small_binary) -> None:
    X, T, Y = small_binary
    est = fit_cate(
        X, T, Y, "trat2 vs ctrl",
        n_estimators=10,
        include_dml=False,
        include_causal_forest=False,
    )
    n = len(Y)
    assert isinstance(est, CATEEstimates)
    assert est.cate_s.shape == (n,)
    assert est.cate_t.shape == (n,)
    assert est.cate_x.shape == (n,)
    assert est.cate_dml is None
    assert est.cate_cf is None

    frame = est.to_frame()
    assert len(frame) == n
    assert {"cate_s", "cate_t", "cate_x", "label"}.issubset(frame.columns)
    assert "cate_dml" not in frame.columns
    assert "cate_cf" not in frame.columns

    summary = est.summary()
    assert "mean_cate_x" in summary.index
    assert "mean_cate_dml" not in summary.index


def test_fit_cate_with_dml_and_causal_forest(small_binary) -> None:
    X, T, Y = small_binary
    est = fit_cate(
        X, T, Y, "trat2 vs ctrl",
        n_estimators=10,
        dml_cv=2,
        cf_n_estimators=8,
    )
    n = len(Y)
    assert est.cate_dml is not None and est.cate_dml.shape == (n,)
    assert est.cate_cf is not None and est.cate_cf.shape == (n,)

    frame = est.to_frame()
    assert {"cate_dml", "cate_cf"}.issubset(frame.columns)

    summary = est.summary()
    assert "mean_cate_dml" in summary.index
    assert "mean_cate_cf" in summary.index
    assert np.isfinite(summary["mean_cate_dml"])
    assert np.isfinite(summary["mean_cate_cf"])
