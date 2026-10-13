"""Fast exact sampling of AP null distributions and streamed p-values.

See :mod:`copairs.nulls.sampler` for the sampler and :mod:`copairs.nulls.pvalues`
for p-values. Backends: ``"cuda"`` (CuPy), ``"numba"`` and ``"numpy"``, all
bitwise identical; ``"auto"`` picks the fastest available. The joint mAP null
(:mod:`copairs.nulls.joint`) runs on the CPU with Numba.
"""

from copairs.nulls import joint
from copairs.nulls.joint import joint_pvalues
from copairs.nulls.pvalues import TIE_TOL, ap_pvalues, map_pvalues, tie_thresholds
from copairs.nulls.sampler import ap_nulls, resolve_backend, available_backends

__all__ = [
    "TIE_TOL",
    "ap_nulls",
    "ap_pvalues",
    "available_backends",
    "joint",
    "joint_pvalues",
    "map_pvalues",
    "resolve_backend",
    "tie_thresholds",
]
