"""Module to compute mAP-based metrics."""

from . import multilabel
from .map import get_map_pvalue, mean_average_precision
from .hierarchical_fdr import apply_fdr_correction, apply_hierarchical_fdr_correction
from .average_precision import average_precision

__all__ = [
    "mean_average_precision",
    "get_map_pvalue",
    "apply_fdr_correction",
    "apply_hierarchical_fdr_correction",
    "multilabel",
    "average_precision",
]
