"""Causal email nudge experiment analysis package."""

from src.analysis import all_ate_comparisons, ate_proportion, scale_impact
from src.causal import CATEEstimates, fit_cate, prep_binary_comparison
from src.data import GROUP_LABELS, load_data

__all__ = [
    "GROUP_LABELS",
    "load_data",
    "ate_proportion",
    "all_ate_comparisons",
    "scale_impact",
    "prep_binary_comparison",
    "fit_cate",
    "CATEEstimates",
]
