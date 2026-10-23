"""Null distributions for mAP p-values, drawn from counter-based random streams.

:mod:`copairs.nulls.philox` generates every draw as a pure function of a seed and
a counter, so a draw does not depend on which other draws are computed, or in
which order. :mod:`copairs.nulls.common` holds the helpers the nulls share.
:mod:`copairs.nulls.joint` and :mod:`copairs.nulls.permutation` are the joint and
label-permutation mAP nulls.

:mod:`copairs.nulls.sampler` samples AP nulls exactly, and
:mod:`copairs.nulls.pvalues` streams AP and mAP p-values against them in
bounded memory. Every null has two CPU backends: ``"numba"`` (optional,
``pip install 'copairs[numba]'``; kernels in :mod:`copairs.nulls._numba`)
and ``"numpy"``, bitwise identical; ``"auto"`` picks Numba when it imports
(:func:`~copairs.nulls.backend.resolve_backend`). The AP nulls and their
p-values also run on ``"cuda"`` (optional, ``pip install 'copairs[gpu]'``),
only when asked for (see :mod:`copairs.nulls.cuda`); the joint and
permutation nulls raise ValueError under it.
"""

from copairs.nulls import joint, permutation
from copairs.nulls.joint import joint_pvalues
from copairs.nulls.common import TIE_TOL, tie_thresholds
from copairs.nulls.backend import resolve_backend, available_backends
from copairs.nulls.pvalues import ap_pvalues, map_pvalues
from copairs.nulls.sampler import ap_nulls

__all__ = [
    "TIE_TOL",
    "ap_nulls",
    "ap_pvalues",
    "available_backends",
    "joint",
    "joint_pvalues",
    "map_pvalues",
    "permutation",
    "resolve_backend",
    "tie_thresholds",
]
