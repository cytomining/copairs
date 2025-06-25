"""Module to compute mAP-based metrics."""

from . import multilabel
from .map import mean_average_precision
from .average_precision import average_precision

__all__ = ["mean_average_precision", "multilabel", "average_precision"]
