"""Pairwise similarities computed in parallel kernels.

Cosine-type metrics normalize each profile once and then take one dot
product per pair, instead of gathering and normalizing both rows of every
pair. Normalization and dot products run in float64 and results are returned
as float32, like ``copairs.compute.get_similarity_fn``; results match it to
float32 rounding, not bitwise. One exception: the generic functions compute in
the input's precision, so float32 rows whose norms overflow float32 (entries
around 1e19 and up) get similarity 0 or NaN there and their actual value here.
"""

from typing import Callable

import numba
import numpy as np

from copairs._numba_cache import launch
from copairs.fastap.ranking import index_pairs

# Dot products of unit rows; 1 / (1 + distance), in _minkowski_pairs' order.
COSINE_METRICS = ("cosine", "abs_cosine", "correlation")
MINKOWSKI_METRICS = ("euclidean", "manhattan", "chebyshev")
FAST_METRICS = COSINE_METRICS + MINKOWSKI_METRICS

_ROW_BLOCK = 8192


def _unit_rows(
    feats: np.ndarray, center: bool, dtype=None, strict: bool = False
) -> np.ndarray:
    """Rows scaled to unit norm (after centering for correlation).

    Computed in float64 one block of rows at a time, so the temporaries stay
    small, and stored as ``dtype``: by default float32 unless ``feats`` is
    float64. With ``strict``, non-finite and zero norms raise ValueError
    instead of giving NaN rows.
    """
    if dtype is None:
        dtype = np.float64 if feats.dtype == np.float64 else np.float32
    out = np.empty(feats.shape, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, len(feats), _ROW_BLOCK):
            x = np.asarray(feats[start : start + _ROW_BLOCK], dtype=np.float64)
            if center:
                x = x - x.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            if strict and not np.isfinite(norms).all():
                raise ValueError("non-finite features; clean them first")
            if strict and (norms == 0).any():
                raise ValueError(
                    "zero-norm feature vector; cosine similarity is undefined"
                )
            out[start : start + _ROW_BLOCK] = x / norms
    return out


# One pair's similarity, shared by the parallel and serial kernels, which
# therefore give the same bits.
@numba.njit(fastmath={"reassoc", "contract"}, cache=True)
def _dot(a, b):
    acc = 0.0
    for t in range(len(a)):
        acc += np.float64(a[t]) * b[t]
    return acc


@numba.njit(fastmath={"reassoc", "contract"}, cache=True)
def _minkowski(a, b, kind):
    """``1 / (1 + distance)`` for kind 0 = euclidean, 1 = manhattan, 2 = chebyshev."""
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
    return 1.0 / (1.0 + acc)


@numba.njit(parallel=True, cache=True)
def _dot_pairs(x, pairs, out):
    for p in numba.prange(len(pairs)):
        out[p] = _dot(x[pairs[p, 0]], x[pairs[p, 1]])


@numba.njit(parallel=True, cache=True)
def _minkowski_pairs(x, pairs, kind, out):
    for p in numba.prange(len(pairs)):
        out[p] = _minkowski(x[pairs[p, 0]], x[pairs[p, 1]], kind)


@numba.njit(nogil=True, cache=True)
def _dot_pairs_serial(x, pairs, out):
    for p in range(len(pairs)):
        out[p] = _dot(x[pairs[p, 0]], x[pairs[p, 1]])


@numba.njit(nogil=True, cache=True)
def _minkowski_pairs_serial(x, pairs, kind, out):
    for p in range(len(pairs)):
        out[p] = _minkowski(x[pairs[p, 0]], x[pairs[p, 1]], kind)


def pair_similarity(
    feats: np.ndarray, metric, parallel: bool = True
) -> Callable | None:
    """Return a ``pairs -> float32 similarities`` kernel, or None without one.

    Parameters
    ----------
    feats : np.ndarray
        ``(n, d)`` profiles; the kernel uses them as given here.
    metric : str or callable
        A distance name; only :data:`FAST_METRICS` have a kernel.
    parallel : bool
        Run on Numba's threads (default), or serially on the calling
        thread, releasing the GIL, for many small calls from several
        threads. Both give the same bits.

    Returns
    -------
    callable or None
        Takes ``(m, 2)`` integer row indices of ``feats`` and returns ``m``
        float32 similarities, rounded from float64 into the output without a
        full-size temporary. uint32 pairs are used without a copy. Raises
        ValueError for other shapes or dtypes and IndexError for indices
        outside ``feats``, checked once per call before the kernel runs.
    """
    rows = kernel_rows(feats, metric)
    if rows is None:
        return None
    x, kind = rows

    def similarity(pairs):
        pairs = index_pairs(pairs, len(x))
        out = np.empty(len(pairs), dtype=np.float32)
        if kind < 0 and parallel:
            launch(_dot_pairs, x, pairs, out)
        elif kind < 0:
            _dot_pairs_serial(x, pairs, out)
        elif parallel:
            launch(_minkowski_pairs, x, pairs, kind, out)
        else:
            _minkowski_pairs_serial(x, pairs, kind, out)
        if metric == "abs_cosine":
            np.abs(out, out=out)
        return out

    return similarity


def kernel_rows(feats: np.ndarray, metric) -> tuple[np.ndarray, int] | None:
    """Return the rows and kind that the similarity kernels take for ``metric``.

    Returns None for metrics without a kernel. Rows are contiguous float32
    or float64 (other dtypes become float64): unit rows for the cosine
    metrics (centered for correlation), kind -1 for their dot products;
    else the features as given, kind their index in
    :data:`MINKOWSKI_METRICS`.
    """
    if not isinstance(metric, str) or metric not in FAST_METRICS:
        return None
    feats = np.asarray(feats)
    if feats.dtype not in (np.float32, np.float64):
        feats = feats.astype(np.float64)
    if metric in MINKOWSKI_METRICS:
        return np.ascontiguousarray(feats), MINKOWSKI_METRICS.index(metric)
    return np.ascontiguousarray(_unit_rows(feats, metric == "correlation")), -1


def PairSimilarity(feats: np.ndarray, metric: str) -> Callable:
    """:func:`pair_similarity`'s kernel, raising ValueError for metrics without one."""
    kernel = pair_similarity(feats, metric)
    if kernel is None:
        raise ValueError(f"no fast kernel for {metric!r}; expected {FAST_METRICS}")
    return kernel
