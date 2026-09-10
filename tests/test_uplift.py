"""Tests for out-of-sample uplift validation and CATE confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import load_data
from src.uplift import (
    cate_with_confidence,
    evaluate_uplift,
    scored_frame,
    summarize_cate_ci,
    uplift_curve_points,
    uplift_scores,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


def test_uplift_scores_perfect_beats_random() -> None:
    """A model that scores by the true responder flag must beat random ranking."""
    rng = np.random.default_rng(0)
    n = 4000
    w = rng.integers(0, 2, n)
    responder = rng.integers(0, 2, n)
    base = (rng.random(n) < 0.1).astype(int)
    # responders click iff treated; everyone else clicks at a treatment-independent base rate
    y = np.where(responder == 1, w, base)

    scored = pd.DataFrame(
        {
            "y": y,
            "w": w,
            "perfect": responder.astype(float),
            "rand": rng.random(n),
        }
    )
    scores = uplift_scores(scored, model_cols=["perfect", "rand"])

    assert list(scores.columns) == ["model", "qini", "auuc"]
    # sorted best-first
    assert scores.iloc[0]["qini"] >= scores.iloc[1]["qini"]
    q = scores.set_index("model")["qini"]
    a = scores.set_index("model")["auuc"]
    assert q["perfect"] > q["rand"]
    assert a["perfect"] > a["rand"]
    # normalized random ranking is ~0
    assert abs(q["rand"]) < 0.2


def test_scored_frame_shapes() -> None:
    frame = scored_frame(
        np.array([0, 1, 1]), np.array([0, 1, 0]), {"cate_x": np.array([0.1, 0.2, 0.3])}
    )
    assert list(frame.columns) == ["y", "w", "cate_x"]
    assert len(frame) == 3
    assert frame["w"].tolist() == [0, 1, 0]


def test_evaluate_uplift_real_out_of_sample(df: pd.DataFrame) -> None:
    scores, scored = evaluate_uplift(
        df,
        "trat2",
        "ctor",
        learners=("t", "x", "cf"),
        n_estimators=20,
        dml_cv=2,
        test_size=0.3,
        random_state=0,
    )
    # one row per learner, finite metrics, sorted by qini desc
    assert set(scores["model"]) == {"cate_t", "cate_x", "cate_cf"}
    assert np.isfinite(scores[["qini", "auuc"]].values).all()
    assert scores["qini"].is_monotonic_decreasing
    # trat2's effect is large, so a real model must rank clearly better than random OOS
    assert scores["qini"].max() > 0.2

    assert {"y", "w", "cate_t", "cate_x", "cate_cf"}.issubset(scored.columns)
    assert set(np.unique(scored["w"])) == {0, 1}
    assert len(scored) > 0


@pytest.mark.parametrize("kind", ["qini", "gain"])
def test_uplift_curve_points(df: pd.DataFrame, kind: str) -> None:
    _, scored = evaluate_uplift(
        df,
        "trat2",
        "ctor",
        learners=("x",),
        n_estimators=20,
        test_size=0.3,
        random_state=0,
    )
    curve = uplift_curve_points(scored, "cate_x", kind=kind)
    assert len(curve) > 0
    assert "cate_x" in curve.columns


def test_cate_with_confidence(df: pd.DataFrame) -> None:
    ci = cate_with_confidence(
        df, "trat2", "ctor", n_estimators=20, dml_cv=2, random_state=0
    )
    assert {"cate", "ci_lower", "ci_upper", "significant"}.issubset(ci.columns)
    # point estimate always inside its own interval
    assert (ci["ci_lower"] <= ci["cate"]).all()
    assert (ci["cate"] <= ci["ci_upper"]).all()
    assert ci["significant"].dtype == bool
    # significance flag is consistent with the interval excluding zero
    expected = (ci["ci_lower"] > 0) | (ci["ci_upper"] < 0)
    assert (ci["significant"] == expected).all()


def test_summarize_cate_ci(df: pd.DataFrame) -> None:
    ci = cate_with_confidence(
        df, "trat2", "ctor", n_estimators=20, dml_cv=2, random_state=0
    )
    summary = summarize_cate_ci(ci)
    assert set(summary.index) == {"n", "mean_cate", "mean_ci_width", "frac_significant"}
    assert summary["n"] == len(ci)
    assert summary["mean_ci_width"] > 0
    assert 0.0 <= summary["frac_significant"] <= 1.0
