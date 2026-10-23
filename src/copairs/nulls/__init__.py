"""Null distributions for mAP p-values, drawn from counter-based random streams.

:mod:`copairs.nulls.philox` generates every draw as a pure function of a seed and
a counter, so a draw does not depend on which other draws are computed, or in
which order. :mod:`copairs.nulls.common` holds the helpers the nulls share.

:mod:`copairs.nulls.sampler` samples AP nulls exactly, and
:mod:`copairs.nulls.pvalues` streams AP and mAP p-values against them in
bounded memory. Backends: ``"cuda"`` (optional, ``pip install
'copairs[gpu]'``), ``"numba"`` (optional, ``pip install 'copairs[numba]'``)
and ``"numpy"``. ``"auto"`` picks the first usable one, and under it all three
are bitwise identical; an explicit ``"cuda"`` also steps long gaps in float32
(see :mod:`copairs.nulls.cuda`).
"""

from copairs.nulls.common import TIE_TOL, tie_thresholds
from copairs.nulls.backend import resolve_backend, available_backends
from copairs.nulls.pvalues import ap_pvalues, map_pvalues
from copairs.nulls.sampler import ap_nulls

__all__ = [
    "TIE_TOL",
    "ap_nulls",
    "ap_pvalues",
    "available_backends",
    "map_pvalues",
    "resolve_backend",
    "tie_thresholds",
]
