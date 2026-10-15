"""Average precision from pair lists without building rank lists.

In copairs' rank list of profile ``i``, pairs are ordered by the float32 key
``1 - similarity``, and a positive ties ahead of a negative with the same key.
The ``t``-th positive (1-based, by key) therefore sits at rank
``t + #{negatives with key < key_t}``, so AP needs only each profile's sorted
positive keys and, for every negative, a binary search into them: no global
sort of all pairs.
"""

import numba
import numpy as np

from copairs.nulls import resolve_backend
from copairs.nulls.pvalues import upper_bound


def resolve_fast_backend(backend: str) -> str:
    """Concrete backend of the fast AP stage, ``"cuda"`` or ``"numba"``."""
    backend = resolve_backend(backend)
    if backend == "numpy":
        raise ValueError(
            "the fast AP stage has no NumPy backend; use backend='numba', or "
            "method='legacy' for copairs' NumPy implementation"
        )
    return backend


def array_module(x):
    """``cupy`` for a CuPy array, else ``numpy`` (CuPy stays an optional import)."""
    if type(x).__module__.startswith("cupy"):
        import cupy

        return cupy
    return np


def sortable_keys(keys):
    """uint64 that orders like float32 ``keys`` in NumPy (NaN last); NumPy or CuPy."""
    xp = array_module(keys)
    keys = xp.where(xp.isnan(keys), xp.float32(np.nan), keys).astype(xp.float32)
    bits = keys.view(xp.uint32)
    flipped = xp.where(bits >> 31 == 1, ~bits, bits | xp.uint32(0x80000000))
    return flipped.astype(xp.uint64)


def rank_keys(sims):
    """Ranking keys of similarities, as computed by ``build_rank_lists`` (NumPy or CuPy)."""
    xp = array_module(sims)
    return xp.float32(1) - xp.asarray(sims, dtype=xp.float32)


@numba.njit(inline="always")
def _rank_key(sim):
    """:func:`rank_keys` of one similarity."""
    return np.float32(1) - np.float32(sim)


def index_pairs(pairs):
    """Contiguous ``(n, 2)`` pairs for the kernels; uint32 pairs are not copied."""
    pairs = np.ascontiguousarray(pairs).reshape(-1, 2)
    return pairs if pairs.dtype == np.uint32 else pairs.astype(np.int64, copy=False)


@numba.njit(cache=True)
def pair_csr(pairs, sims, n):
    """Rank keys of each profile's pairs (both endpoints) in CSR layout, unsorted."""
    ptr = np.zeros(n + 1, dtype=np.int64)
    for p in range(len(pairs)):
        ptr[pairs[p, 0] + 1] += 1
        ptr[pairs[p, 1] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    fill = ptr[:-1].copy()
    vals = np.empty(2 * len(pairs), dtype=np.float32)
    for p in range(len(pairs)):
        key = _rank_key(sims[p])
        for side in range(2):
            i = pairs[p, side]
            vals[fill[i]] = key
            fill[i] += 1
    return ptr, vals


@numba.njit(parallel=True, cache=True)
def _sort_segments(ptr, vals):
    for i in numba.prange(len(ptr) - 1):
        vals[ptr[i] : ptr[i + 1]] = np.sort(vals[ptr[i] : ptr[i + 1]])


def _positive_csr(pairs, sims, n):
    """Positive rank keys of each profile, sorted, in CSR layout."""
    ptr, vals = pair_csr(pairs, sims, n)
    _sort_segments(ptr, vals)
    return ptr, vals


@numba.njit(inline="always")
def ap_from_counts(hist, base, num_pos):
    """AP given ``hist[base + t]`` negatives ranked between positives t - 1 and t."""
    before = 0
    acc = 0.0
    for t in range(num_pos):
        before += hist[base + t]
        acc += (t + 1) / (t + 1 + before)
    return acc / num_pos


@numba.njit(parallel=True, cache=True)
def _negative_hist(neg_pairs, neg_sims, ptr, vals, n, n_chunks):
    """Per-profile histograms of negatives over the gaps between positive keys.

    Bin ``ptr[i] + i + q`` counts the negatives of profile ``i`` whose key lies
    between its ``q``-th and ``q + 1``-th smallest positive keys (ties go after
    the positive). Each chunk of pairs fills its own histogram row.
    """
    size = len(neg_pairs)
    hist = np.zeros((n_chunks, ptr[n] + n), dtype=np.int64)
    n_neg = np.zeros((n_chunks, n), dtype=np.int64)
    step = (size + n_chunks - 1) // n_chunks
    for c in numba.prange(n_chunks):
        for p in range(c * step, min(size, (c + 1) * step)):
            key = _rank_key(neg_sims[p])
            for side in range(2):
                i = neg_pairs[p, side]
                hist[c, upper_bound(vals, ptr[i], ptr[i + 1], key) + i] += 1
                n_neg[c, i] += 1
    return hist.sum(axis=0), n_neg.sum(axis=0)


@numba.njit(parallel=True, cache=True)
def _ap_from_hist(ptr, hist, n):
    ap = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        num_pos = ptr[i + 1] - ptr[i]
        if num_pos == 0:
            ap[i] = np.nan
            continue
        ap[i] = ap_from_counts(hist, ptr[i] + i, num_pos)
    return ap


def _host(x):
    """NumPy array of ``x``, copied from the GPU if it is a CuPy array."""
    return x.get() if array_module(x) is not np else np.asarray(x)


def ap_from_pairs(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
    backend: str = "numba",
    budget_bytes: int = 2**30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average precision of every profile that appears in a pair.

    Parameters
    ----------
    pos_pairs, neg_pairs : np.ndarray
        ``(n, 2)`` profile indices of positive and negative pairs.
    pos_sims, neg_sims : np.ndarray
        Similarities of the pairs (NumPy, or CuPy to stay on the GPU).
    backend : str
        ``"auto"``, ``"numba"`` or ``"cuda"``.
    budget_bytes : int
        Upper bound on the per-thread negative histograms (Numba).

    Returns
    -------
    paired_ix : np.ndarray
        Sorted indices of profiles in at least one pair.
    ap_scores : np.ndarray
        float64 AP of each profile in ``paired_ix`` (NaN without positives).
    null_confs : np.ndarray
        ``(n, 2)`` uint32 ``(num_pos, total)`` of each profile in ``paired_ix``.
    """
    backend = resolve_fast_backend(backend)
    n = 1 + max(
        int(pairs.max()) if pairs.size else -1 for pairs in (pos_pairs, neg_pairs)
    )
    if backend == "cuda" and n >= 2**32:
        backend = "numba"  # the GPU sort packs profile indices into 32 bits
    if backend == "cuda":
        from copairs.fastap import cuda

        ap, num_pos, n_neg = cuda.ap_from_pairs(
            pos_pairs, neg_pairs, rank_keys(pos_sims), neg_sims, n
        )
    else:
        pos_pairs, neg_pairs = index_pairs(pos_pairs), index_pairs(neg_pairs)
        ptr, vals = _positive_csr(pos_pairs, _host(pos_sims), n)
        bins = int(ptr[n]) + n
        n_chunks = budget_bytes // (8 * (bins + n))
        n_chunks = max(1, min(numba.get_num_threads(), n_chunks))
        hist, n_neg = _negative_hist(neg_pairs, _host(neg_sims), ptr, vals, n, n_chunks)
        ap = _ap_from_hist(ptr, hist, n)
        num_pos = np.diff(ptr)
    total = num_pos + n_neg
    paired_ix = np.flatnonzero(total)
    null_confs = np.stack([num_pos[paired_ix], total[paired_ix]], axis=1)
    return paired_ix, ap[paired_ix], null_confs.astype(np.uint32)
