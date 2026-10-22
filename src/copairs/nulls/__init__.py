"""Null distributions for mAP p-values, drawn from counter-based random streams.

:mod:`copairs.nulls.philox` generates every draw as a pure function of a seed and
a counter, so a draw does not depend on which other draws are computed, or in
which order. :mod:`copairs.nulls.common` holds the helpers the nulls share.
"""

from copairs.nulls.common import TIE_TOL, tie_thresholds

__all__ = ["TIE_TOL", "tie_thresholds"]
