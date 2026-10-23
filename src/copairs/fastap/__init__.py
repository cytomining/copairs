"""Numba kernels behind ``backend="numba"``: similarities, AP by counting.

This package needs the optional ``numba`` extra (``pip install
'copairs[numba]'``). copairs imports it only when ``backend="numba"`` is
requested, through ``copairs.compute._import_numba_backend``. AP values match
copairs' rank-list implementation to float64 rounding given the same
similarities.
"""

from typing import Callable

import numpy as np

from copairs.fastap.ranking import ap_from_pairs
from copairs.fastap.similarity import FAST_METRICS, PairSimilarity
from copairs.fastap.normalization import expected_ap_array

__all__ = [
    "FAST_METRICS",
    "PairSimilarity",
    "ap_from_pairs",
    "expected_ap_array",
    "pair_similarity",
    "similarity_fn",
]


def pair_similarity(feats: np.ndarray, distance) -> PairSimilarity | None:
    """Return a ``pairs -> float32 similarities`` kernel, or None without one."""
    if not isinstance(distance, str) or distance not in FAST_METRICS:
        return None
    return PairSimilarity(feats, distance)


def similarity_fn(feats: np.ndarray, distance, generic: Callable) -> Callable:
    """Return ``generic``, or a kernel for ``distance`` on ``feats`` in its place.

    ``generic`` is ``copairs.compute.get_similarity_fn(distance)``, called as
    ``generic(feats, pairs, batch_size)``. The replacement takes the same
    arguments, but the features it uses are the ``feats`` given here, which
    the kernel normalizes once.
    """
    kernel = pair_similarity(np.asarray(feats), distance)
    if kernel is None:
        return generic

    def similarity(_feats, pairs, _batch_size):
        return kernel(pairs)

    return similarity
