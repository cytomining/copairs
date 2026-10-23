"""Pairwise similarities computed in parallel kernels.

Cosine-type metrics normalize each profile once and then take one dot
product per pair, instead of gathering and normalizing both rows of every
pair. Normalization and dot products run in float64 and results are returned
as float32, like ``copairs.compute.get_similarity_fn``; results match it to
float32 rounding, not bitwise. One exception: the generic functions compute in
the input's precision, so float32 rows whose norms overflow float32 (entries
around 1e19 and up) get similarity 0 or NaN there and their actual value here.
"""

import numba
import numpy as np

FAST_METRICS = (
    "cosine",
    "abs_cosine",
    "correlation",
    "euclidean",
    "manhattan",
    "chebyshev",
)


_ROW_BLOCK = 8192
# Pairs per kernel call; bounds the int64 copy of the pair indices.
PAIR_CHUNK = 1 << 24


def _unit_rows(feats: np.ndarray, center: bool) -> np.ndarray:
    """Rows scaled to unit norm (after centering for correlation).

    Computed in float64 one block of rows at a time, so the temporaries stay
    small, and stored as float32 unless ``feats`` is float64.
    """
    out = np.empty(
        feats.shape, dtype=np.float64 if feats.dtype == np.float64 else np.float32
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, len(feats), _ROW_BLOCK):
            x = np.asarray(feats[start : start + _ROW_BLOCK], dtype=np.float64)
            if center:
                x = x - x.mean(axis=1, keepdims=True)
            out[start : start + _ROW_BLOCK] = x / np.linalg.norm(
                x, axis=1, keepdims=True
            )
    return out


@numba.njit(parallel=True, fastmath={"reassoc", "contract"}, cache=True)
def _dot_pairs(x, pairs, out):
    for p in numba.prange(len(pairs)):
        a, b = x[pairs[p, 0]], x[pairs[p, 1]]
        acc = 0.0
        for t in range(len(a)):
            acc += np.float64(a[t]) * b[t]
        out[p] = acc


@numba.njit(parallel=True, fastmath={"reassoc", "contract"}, cache=True)
def _minkowski_pairs(x, pairs, kind, out):
    """``1 / (1 + distance)`` for kind 0 = euclidean, 1 = manhattan, 2 = chebyshev."""
    for p in numba.prange(len(pairs)):
        a, b = x[pairs[p, 0]], x[pairs[p, 1]]
        acc = 0.0
        for t in range(len(a)):
            diff = abs(np.float64(a[t]) - b[t])
            if kind == 0:
                acc += diff * diff
            elif kind == 1:
                acc += diff
            elif diff > acc or np.isnan(diff):  # NaN propagates, as in np.max
                acc = diff
        if kind == 0:
            acc = np.sqrt(acc)
        out[p] = 1.0 / (1.0 + acc)


class PairSimilarity:
    """Similarity of indexed profile pairs for one feature matrix.

    Parameters
    ----------
    feats : np.ndarray
        ``(n, d)`` profiles.
    metric : str
        One of :data:`FAST_METRICS`.
    """

    def __init__(self, feats: np.ndarray, metric: str):
        if metric not in FAST_METRICS:
            raise ValueError(f"no fast kernel for {metric!r}; expected {FAST_METRICS}")
        feats = np.asarray(feats)
        if feats.dtype not in (np.float32, np.float64):
            feats = feats.astype(np.float64)
        self.metric = metric
        if metric in ("cosine", "abs_cosine", "correlation"):
            self.x = np.ascontiguousarray(_unit_rows(feats, metric == "correlation"))
        else:
            self.x = np.ascontiguousarray(feats)

    def __call__(self, pairs: np.ndarray) -> np.ndarray:
        """float32 similarity of each ``(i, j)`` row of ``pairs``.

        Pairs are converted to int64 a chunk at a time, and the kernels round
        each float64 result into the float32 output, so no full-size temporary
        is built.
        """
        pairs = np.asarray(pairs).reshape(-1, 2)
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), PAIR_CHUNK):
            chunk = np.ascontiguousarray(
                pairs[start : start + PAIR_CHUNK], dtype=np.int64
            )
            dest = out[start : start + PAIR_CHUNK]
            if self.metric in ("cosine", "abs_cosine", "correlation"):
                _dot_pairs(self.x, chunk, dest)
            else:
                kind = ("euclidean", "manhattan", "chebyshev").index(self.metric)
                _minkowski_pairs(self.x, chunk, kind, dest)
        if self.metric == "abs_cosine":
            np.abs(out, out=out)
        return out
