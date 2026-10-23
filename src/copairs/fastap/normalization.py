"""Expected AP under random ranking for many configurations at once."""

import sys

import numba
import numpy as np

# Largest rank-list length whose harmonic numbers are tabulated at once.
_MAX_TABLE = 2**24


@numba.njit(cache=True)
def _harmonic_table(n: int, compensated: bool) -> np.ndarray:
    """``H_1..H_n`` exactly as ``harmonic_number`` computes them with ``sum()``.

    From Python 3.12, ``sum()`` adds floats with Neumaier compensation and adds
    the compensation once at the end; earlier versions add sequentially.
    """
    out = np.empty(n)
    total = 0.0
    comp = 0.0
    for k in range(1, n + 1):
        x = 1.0 / k
        if k == 1 or not compensated:
            total += x
        else:
            t = total + x
            if abs(total) >= abs(x):
                comp += (total - t) + x
            else:
                comp += (x - t) + total
            total = t
        out[k - 1] = total + comp
    return out


def expected_ap_array(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Vectorized :func:`copairs.map.normalization.expected_ap`, equal bit for bit.

    Each distinct ``(M, N)`` is evaluated once, with harmonic numbers tabulated
    by the same summation that ``harmonic_number`` performs.
    """
    M = np.asarray(M, dtype=np.int64)
    N = np.asarray(N, dtype=np.int64)
    L = M + N
    bad = (L < 1) | (M < 0) | (N < 0)
    if bad.any():
        i = np.flatnonzero(bad)[0]
        raise ValueError(f"Invalid inputs: M={M[i]}, N={N[i]}")
    if not len(L):
        return np.zeros(0)
    # Distinct (M, L) through one 1-D key M * width + L; M <= L < width.
    width = int(L.max()) + 1
    if width > 2**32:  # width**2 must fit in uint64
        raise ValueError(f"rank lists of up to 2**32 - 1 items, got {width - 1}")
    width = np.uint64(width)
    keys, inv = np.unique(
        M.astype(np.uint64) * width + L.astype(np.uint64), return_inverse=True
    )
    Mu, Lu = (keys // width).astype(np.int64), (keys % width).astype(np.int64)
    # Without positives the expectation is 0, with only positives 1 (L == 1
    # included); only configurations holding both need harmonic numbers.
    mu0 = (Mu == Lu).astype(np.float64)
    mixed = (Mu > 0) & (Mu < Lu)
    if mixed.any():
        Mm, Lm = Mu[mixed], Lu[mixed]
        if Lm.max() <= _MAX_TABLE:
            table = _harmonic_table(int(Lm.max()), sys.version_info >= (3, 12))
            H = table[Lm - 1]
        else:
            # Imported here: the copairs.map package loads statsmodels and DuckDB.
            from copairs.map.normalization import harmonic_number

            H = np.array([harmonic_number(int(n)) for n in Lm])
        mu0[mixed] = (1.0 / Lm) * (((Mm - 1.0) / (Lm - 1.0)) * (Lm - H) + H)
    return mu0[inv.ravel()]
