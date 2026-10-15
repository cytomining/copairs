"""Functions for normalizing Average Precision scores."""

import sys
from typing import Tuple, Union

import numba
import numpy as np


def harmonic_number(n: int) -> float:
    """Compute the n-th harmonic number H_n = Σ(1/k) for k=1 to n.

    Parameters
    ----------
    n : int
        The index of the harmonic number to compute.

    Returns
    -------
    float
        The n-th harmonic number.
    """
    if n <= 0:
        return 0.0
    return sum(1.0 / k for k in range(1, n + 1))


def expected_ap(M: int, N: int) -> float:
    """Compute the expected Average Precision under random ranking.

    This implements the exact finite-sample formula for expected AP when
    items are randomly ranked.

    Parameters
    ----------
    M : int
        Number of positive items (relevant documents).
    N : int
        Number of negative items (irrelevant documents).

    Returns
    -------
    float
        The expected Average Precision under random ranking.

    Notes
    -----
    Formula: E[AP] = (1/L) × [(M-1)/(L-1) × (L - H_L) + H_L]
    where L = M + N and H_L is the L-th harmonic number.
    """
    L = M + N

    # Handle edge cases
    if L < 1 or M < 0 or N < 0:
        raise ValueError(f"Invalid inputs: M={M}, N={N}")
    if L == 1:
        return 1.0 if M == 1 else 0.0
    if M == 0:
        return 0.0
    if M == L:  # All items are positive
        return 1.0

    # Compute the L-th harmonic number
    H_L = harmonic_number(L)

    # Apply the exact formula
    mu0 = (1.0 / L) * (((M - 1.0) / (L - 1.0)) * (L - H_L) + H_L)

    return mu0


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
    """Vectorized :func:`expected_ap`, equal to it bit for bit.

    Each distinct ``(M, N)`` is evaluated once, with harmonic numbers tabulated
    by the same summation that :func:`harmonic_number` performs.
    """
    M = np.asarray(M, dtype=np.int64)
    N = np.asarray(N, dtype=np.int64)
    L = M + N
    bad = (L < 1) | (M < 0) | (N < 0)
    if bad.any():
        i = np.flatnonzero(bad)[0]
        raise ValueError(f"Invalid inputs: M={M[i]}, N={N[i]}")
    (Mu, Lu), inv = np.unique(np.stack([M, L]), axis=1, return_inverse=True)
    if Lu.max() <= _MAX_TABLE:
        table = _harmonic_table(int(Lu.max()), sys.version_info >= (3, 12))
        H = table[Lu - 1]
    else:
        H = np.array([harmonic_number(int(n)) for n in Lu])
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = (1.0 / Lu) * (((Mu - 1.0) / (Lu - 1.0)) * (Lu - H) + H)
    mu0 = np.where(Lu == 1, (Mu == 1).astype(float), mu0)
    mu0 = np.where(Mu == 0, 0.0, mu0)
    mu0 = np.where(Mu == Lu, 1.0, mu0)
    return mu0[inv.ravel()]


def normalize_ap(
    ap: Union[float, np.ndarray],
    M: Union[int, np.ndarray],
    N: Union[int, np.ndarray],
    eps: float = 1e-10,
) -> Union[float, np.ndarray]:
    """Normalize Average Precision scores to be scale-independent.

    Computes the normalized AP as (AP - μ₀) / (1 - μ₀) where μ₀ is the
    expected AP under random ranking.

    Parameters
    ----------
    ap : float or np.ndarray
        The Average Precision score(s) to normalize.
    M : int or np.ndarray
        Number of positive items for each AP score.
    N : int or np.ndarray
        Number of negative items for each AP score.
    eps : float
        Small epsilon to avoid division by zero when μ₀ ≈ 1.

    Returns
    -------
    float or np.ndarray
        The normalized Average Precision score(s).

    Notes
    -----
    - Normalized AP = 0 when performance equals random chance
    - Normalized AP = 1 when performance is perfect
    - Negative values indicate worse-than-random performance
    """
    # Handle scalar or array inputs
    is_scalar = np.isscalar(ap)

    ap = np.atleast_1d(ap)
    M = np.atleast_1d(M)
    N = np.atleast_1d(N)

    # Validate that all arrays have compatible lengths
    lengths = [
        len(ap),
        len(M) if len(M) > 1 else len(ap),
        len(N) if len(N) > 1 else len(ap),
    ]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Array lengths must match: ap={len(ap)}, M={len(M)}, N={len(N)}"
        )

    # Compute expected AP for each configuration
    M = M if len(M) > 1 else np.repeat(M[:1], len(ap))
    N = N if len(N) > 1 else np.repeat(N[:1], len(ap))
    mu0 = expected_ap_array(M, N) if len(ap) else np.zeros(0)

    # Normalize: (AP - μ₀) / (1 - μ₀)
    # Use eps to avoid division by zero when μ₀ ≈ 1
    denominator = np.maximum(1 - mu0, eps)
    normalized = (ap - mu0) / denominator

    # Clip to [-1, 1] range to handle numerical edge cases
    normalized = np.clip(normalized, -1.0, 1.0)

    return float(normalized[0]) if is_scalar else normalized


def compute_normalized_ap_scores(
    ap_scores: np.ndarray, null_confs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute both raw and normalized Average Precision scores.

    Parameters
    ----------
    ap_scores : np.ndarray
        Array of raw Average Precision scores.
    null_confs : np.ndarray
        Array of configurations where each row is [n_pos_pairs, n_total_pairs].

    Returns
    -------
    ap_scores : np.ndarray
        The original raw AP scores.
    normalized_ap_scores : np.ndarray
        The normalized AP scores.
    """
    # Extract M (positive pairs) and compute N (negative pairs)
    M = null_confs[:, 0].astype(int)
    L = null_confs[:, 1].astype(int)
    N = L - M

    # Compute normalized scores
    normalized_ap_scores = normalize_ap(ap_scores, M, N)

    return ap_scores, normalized_ap_scores
