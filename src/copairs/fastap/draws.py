"""Average precision of many query-vs-reference draws, batched.

A draw has ``k`` query profiles and ``m`` reference profiles. Each query
ranks the other ``k - 1`` queries (positives) and the ``m`` references
(negatives) by cosine similarity; a positive ties ahead of a negative with
the same similarity. Similarities of a chunk of draws come from one batched
matrix product, and each query row's AP from counting the negatives above
each of its sorted positives, so no row is fully sorted.
"""

import numba
import numpy as np

from copairs.fastap.ranking import ap_from_counts
from copairs.fastap.similarity import _ROW_BLOCK

DEFAULT_BUDGET = 2**29


def unit_rows(feats, normalized: bool = False) -> np.ndarray:
    """float32 rows of unit norm; raises on non-finite or zero-norm rows.

    With ``normalized``, rows already have unit norm and are only checked.
    Norms are computed in float64 one block of rows at a time, so rows whose
    squared norm overflows float32 keep their direction.
    """
    x = np.asarray(feats, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite features; clean them first")
    if normalized:
        return x
    out = np.empty_like(x)
    for start in range(0, len(x), _ROW_BLOCK):
        block = x[start : start + _ROW_BLOCK].astype(np.float64)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("zero-norm feature vector; cosine similarity is undefined")
        out[start : start + _ROW_BLOCK] = block / norms
    return out


@numba.njit(parallel=True, cache=True)
def _row_ap(sims, k, out):
    """AP of each ``(draw, query)`` row of ``sims`` with shape ``(n_rows, k + m)``."""
    n_rows, width = sims.shape
    m = width - k
    for row in numba.prange(n_rows):
        q = row % k
        s = sims[row]
        pos = np.empty(k - 1, dtype=sims.dtype)
        j = 0
        for i in range(k):
            if i != q:
                pos[j] = s[i]
                j += 1
        pos = np.sort(pos)[::-1]  # descending
        hist = np.zeros(k, dtype=np.int64)
        for e in range(m):
            v = s[k + e]
            lo, hi = 0, k - 1  # first t with pos[t] < v
            while lo < hi:
                mid = (lo + hi) >> 1
                if pos[mid] >= v:
                    lo = mid + 1
                else:
                    hi = mid
            hist[lo] += 1
        out[row] = ap_from_counts(hist, 0, k - 1)


def _chunk(n_draws: int, k: int, m: int, d: int, budget_bytes: int) -> int:
    per_draw = 4 * (k + m) * (d + k)
    return int(max(1, min(n_draws, budget_bytes // per_draw)))


def draw_average_precisions(
    feats,
    queries,
    references,
    normalized: bool = False,
    budget_bytes: int = DEFAULT_BUDGET,
) -> np.ndarray:
    """Average precision of every query of every draw.

    Parameters
    ----------
    feats : np.ndarray
        ``(n, d)`` profiles.
    queries : np.ndarray
        ``(n_draws, k)`` row indices of each draw's queries, ``k >= 2``.
    references : np.ndarray
        ``(n_draws, m)`` row indices of each draw's references.
    normalized : bool
        Whether ``feats`` rows already have unit norm (skips normalization;
        features are still checked to be finite).
    budget_bytes : int
        Upper bound on gathered features and similarities held at once.

    Returns
    -------
    np.ndarray
        ``(n_draws, k)`` float64 APs.
    """
    queries = np.asarray(queries, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    if queries.ndim != 2 or references.ndim != 2 or len(queries) != len(references):
        raise ValueError("queries and references must be (n_draws, k) and (n_draws, m)")
    n_draws, k = queries.shape
    m = references.shape[1]
    if k < 2:
        raise ValueError(f"need at least 2 queries per draw, got {k}")
    for idx in (queries, references):
        if idx.size and (idx.min() < 0 or idx.max() >= len(feats)):
            raise ValueError(f"indices must address the {len(feats)} rows of feats")
    x = unit_rows(feats, normalized)
    idx = np.concatenate([queries, references], axis=1)
    out = np.empty((n_draws, k), dtype=np.float64)
    step = _chunk(n_draws, k, m, x.shape[1], budget_bytes)
    for start in range(0, n_draws, step):
        sl = slice(start, start + step)
        rows = x[idx[sl]]  # (b, k + m, d)
        sims = np.matmul(rows[:, :k], rows.transpose(0, 2, 1))  # (b, k, k + m)
        _row_ap(sims.reshape(-1, k + m), k, out[sl].reshape(-1))
    return out
