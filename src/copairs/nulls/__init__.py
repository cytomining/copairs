"""Null distributions for mAP p-values, drawn from counter-based random streams.

:mod:`copairs.nulls.philox` generates every draw as a pure function of a seed and
a counter, so a draw does not depend on which other draws are computed, or in
which order. :mod:`copairs.nulls.common` holds the helpers the nulls share.
:mod:`copairs.nulls.joint` and :mod:`copairs.nulls.permutation` are the joint and
label-permutation mAP nulls. Everything here runs on NumPy.
"""

from copairs.nulls import joint, permutation
from copairs.nulls.joint import joint_pvalues
from copairs.nulls.common import TIE_TOL, tie_thresholds

__all__ = ["TIE_TOL", "joint", "joint_pvalues", "permutation", "tie_thresholds"]
