"""Package exports for the email nudge causal analysis."""

from src.analysis import all_ate_comparisons, ate_proportion, scale_impact
from src.causal import (
    CATEEstimates,
    calibrate_cate_to_ate,
    fit_cate,
    prep_binary_comparison,
)
from src.data import GROUP_LABELS, load_data
from src.mediation import all_funnel_mediations, funnel_mediation

__all__ = [
    "GROUP_LABELS",
    "load_data",
    "ate_proportion",
    "all_ate_comparisons",
    "scale_impact",
    "prep_binary_comparison",
    "fit_cate",
    "calibrate_cate_to_ate",
    "CATEEstimates",
    "funnel_mediation",
    "all_funnel_mediations",
]
