"""Binary search shared by the Numba AP kernels and the Numba null counts.

A module of its own, outside :mod:`copairs.fastap`, so that the null counts
(:mod:`copairs.nulls._numba`) do not import the AP kernels. Only the Numba
backends import it.
"""

import numba
import numpy as np


@numba.njit(inline="always")
def upper_bound(vals, lo, hi, key):
    """First index in sorted ``vals[lo:hi]`` after ``key`` (NumPy order, NaN last)."""
    key_nan = np.isnan(key)
    while lo < hi:
        mid = (lo + hi) >> 1
        v = vals[mid]
        if key_nan or (not np.isnan(v) and v <= key):
            lo = mid + 1
        else:
            hi = mid
    return lo
