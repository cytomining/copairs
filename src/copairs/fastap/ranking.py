"""Average precision from pair lists without building rank lists.

In copairs' rank list of profile ``i``, pairs are ordered by the float32 key
``1 - similarity``, and a positive ties ahead of a negative with the same key.
The ``t``-th positive (1-based, by key) therefore sits at rank
``t + #{negatives with key < key_t}``, so AP needs only each profile's sorted
positive keys and, for every negative, a binary search into them: no global
sort of all pairs.
"""

import operator

import numba
import numpy as np

from copairs._search import upper_bound
from copairs._numba_cache import launch, cached_njit


def sortable_keys(keys: np.ndarray, xp=np) -> np.ndarray:
    """uint64 that orders like float32 ``keys`` in NumPy (NaN last).

    ``xp`` is the array module of ``keys``: NumPy, or CuPy on the GPU.
    """
    keys = xp.where(xp.isnan(keys), xp.float32(np.nan), keys).astype(xp.float32)
    bits = keys.view(xp.uint32)
    flipped = xp.where(bits >> 31 == 1, ~bits, bits | xp.uint32(0x80000000))
    return flipped.astype(xp.uint64)


def rank_keys(sims, xp=np) -> np.ndarray:
    """Ranking keys of similarities, as computed by ``build_rank_lists``.

    ``xp`` is the array module of the result: NumPy, or CuPy on the GPU.
    """
    return xp.float32(1) - xp.asarray(sims, dtype=xp.float32)


@numba.njit(inline="always")
def _rank_key(sim):
    """:func:`rank_keys` of one similarity."""
    return np.float32(1) - np.float32(sim)


def index_pairs(pairs, n: int | None = None) -> np.ndarray:
    """Contiguous ``(k, 2)`` pairs for the kernels; uint32 pairs are not copied.

    The kernels index without bounds checks, so this raises instead:
    ValueError unless ``pairs`` is a ``(k, 2)`` integer array (or empty), and
    IndexError for an index outside ``range(n)``, or a negative one when
    ``n`` is None. One vectorized pass over the pairs.
    """
    pairs = np.asarray(pairs)
    if pairs.shape in ((0,), (0, 2)):
        dtype = np.uint32 if pairs.dtype == np.uint32 else np.int64
        return np.empty((0, 2), dtype=dtype)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (k, 2), got {pairs.shape}")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError(f"pairs must hold integer row indices, got {pairs.dtype}")
    lo, hi = pairs.min(), pairs.max()
    if lo < 0:
        raise IndexError(f"pair index {lo} is out of bounds: indices must be >= 0")
    if n is not None and hi >= n:
        raise IndexError(f"pair index {hi} is out of bounds for {n} profiles")
    pairs = np.ascontiguousarray(pairs)
    return pairs if pairs.dtype == np.uint32 else pairs.astype(np.int64, copy=False)


def pair_sims(sims, pairs: np.ndarray, xp=np) -> np.ndarray:
    """``sims`` as an ``xp`` array, one per pair; ValueError otherwise."""
    sims = xp.asarray(sims)
    if sims.shape != (len(pairs),):
        raise ValueError(
            f"need one similarity per pair: {len(pairs)} pairs, sims of shape "
            f"{sims.shape}"
        )
    return sims


def profile_count(n) -> int:
    """``n`` as a number of profiles: a nonnegative integer."""
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"n must be a nonnegative number of profiles, got {n}")
    return n


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
        vals[ptr[i] : ptr[i + 1]].sort()


@numba.njit(inline="always")
def ap_from_counts(hist, base, num_pos):
    """AP given ``hist[base + t]`` negatives ranked between positives t - 1 and t."""
    before = 0
    acc = 0.0
    for t in range(num_pos):
        before += hist[base + t]
        acc += (t + 1) / (t + 1 + before)
    return acc / num_pos


@cached_njit(parallel=True)
def _negative_hist(neg_pairs, neg_sims, ptr, vals, hist, n_neg):
    """Per-profile histograms of negatives over the gaps between positive keys.

    Bin ``ptr[i] + i + q`` counts the negatives of profile ``i`` whose key lies
    between its ``q``-th and ``q + 1``-th smallest positive keys (ties go after
    the positive). Each chunk of pairs fills its own row of the zeroed
    ``hist`` and ``n_neg``.
    """
    size = len(neg_pairs)
    n_chunks = len(hist)
    step = (size + n_chunks - 1) // n_chunks
    for c in numba.prange(n_chunks):
        for p in range(c * step, min(size, (c + 1) * step)):
            key = _rank_key(neg_sims[p])
            for side in range(2):
                i = neg_pairs[p, side]
                hist[c, upper_bound(vals, ptr[i], ptr[i + 1], key) + i] += 1
                n_neg[c, i] += 1


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


def ap_from_pairs(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
    budget_bytes: int = 2**30,
    n: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average precision of every profile that appears in a pair.

    Returns what ``build_rank_lists`` followed by ``compute.ap_contiguous``
    returns, with APs equal to float64 rounding.

    Parameters
    ----------
    pos_pairs, neg_pairs : np.ndarray
        ``(k, 2)`` integer profile indices of positive and negative pairs.
    pos_sims, neg_sims : np.ndarray
        One similarity per pair, ranked by its float32 ``1 - similarity``.
    budget_bytes : int
        Upper bound on the per-thread negative histograms.
    n : int, optional
        Number of profiles, more than any index in the pairs; found from the
        pairs when not given.

    Raises
    ------
    ValueError
        For pairs that are not ``(k, 2)`` integers, similarities that are not
        one per pair, or a negative ``n``.
    IndexError
        For a negative pair index, or one of at least ``n``.

    Returns
    -------
    paired_ix : np.ndarray
        Sorted indices of profiles in at least one pair.
    ap_scores : np.ndarray
        float64 AP of each profile in ``paired_ix`` (NaN without positives).
    null_confs : np.ndarray
        ``(n, 2)`` uint32 ``(num_pos, total)`` of each profile in ``paired_ix``.
    """
    pos_pairs, neg_pairs, n = checked_pairs(pos_pairs, neg_pairs, n)
    pos_sims, neg_sims = pair_sims(pos_sims, pos_pairs), pair_sims(neg_sims, neg_pairs)
    ptr, vals = pair_csr(pos_pairs, pos_sims, n)
    launch(_sort_segments, ptr, vals)
    bins = int(ptr[n]) + n
    # A bin or count of one chunk is at most 2 per negative pair (a self-pair).
    dtype = np.int32 if 2 * len(neg_pairs) <= np.iinfo(np.int32).max else np.int64
    # Without profiles (no pairs, n 0 or omitted) a chunk has no bins at all.
    n_chunks = budget_bytes // (np.dtype(dtype).itemsize * max(1, bins + n))
    n_chunks = max(1, min(numba.get_num_threads(), n_chunks))
    hist = np.zeros((n_chunks, bins), dtype=dtype)
    n_neg = np.zeros((n_chunks, n), dtype=dtype)
    launch(_negative_hist, neg_pairs, neg_sims, ptr, vals, hist, n_neg)
    # Summed in int64
    hist, n_neg = hist.sum(axis=0, dtype=np.int64), n_neg.sum(axis=0, dtype=np.int64)
    ap = launch(_ap_from_hist, ptr, hist, n)
    return paired_results(ap, np.diff(ptr), n_neg)


def checked_pairs(pos_pairs, neg_pairs, n: int | None):
    """:func:`ap_from_pairs`' pairs, checked by :func:`index_pairs`, and ``n``.

    ``n`` defaults to one more than the largest index in the pairs.
    """
    n = None if n is None else profile_count(n)
    pos_pairs, neg_pairs = index_pairs(pos_pairs, n), index_pairs(neg_pairs, n)
    if n is None:
        n = 1 + max(int(p.max()) if len(p) else -1 for p in (pos_pairs, neg_pairs))
    return pos_pairs, neg_pairs, n


def paired_results(ap, num_pos, n_neg):
    """:func:`ap_from_pairs`' results from every profile's AP and int64 pair counts."""
    total = num_pos + n_neg
    paired_ix = np.flatnonzero(total)
    null_confs = np.stack([num_pos[paired_ix], total[paired_ix]], axis=1)
    return paired_ix, ap[paired_ix], null_confs.astype(np.uint32)
