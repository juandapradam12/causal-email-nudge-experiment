"""Unit tests for analysis, mediation, and causal helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import all_ate_comparisons, ate_proportion, scale_impact
from src.causal import calibrate_cate_to_ate, prep_binary_comparison, validate_cate_vs_ate
from src.data import load_data
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
