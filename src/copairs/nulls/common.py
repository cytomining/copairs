"""Helpers shared by the mAP nulls: tie thresholds, seeds and null sizes.

A null value counts against an observed score when ``null >= score - tol``
(see :func:`tie_thresholds`). The tolerance absorbs rounding between the two
computations, so a null rank list identical to the observed one is counted as
a tie, as in the ``>=`` definition of a permutation p-value.
"""

import hashlib

import numpy as np

# Far above float64 summation-order differences of an AP (~k * 2**-52). Distinct
# AP values can lie closer than this in larger configurations (e.g. 9e-13 apart
# for 2 positives among 5000), so a null value just below a score may count as
# a tie; each such value has null probability ~1/C(total, num_pos).
TIE_TOL = 1e-10
# Rounding, in units of eps * |score|, allowed for scores below float64
# precision: float32 APs are within 0.48 of the exact value (measured for k up
# to 20000 positives), and a float32 mean adds at most 0.5.
TIE_ULPS = 8


def tie_thresholds(scores) -> np.ndarray:
    """float64 ``scores - tol``; null values at or above a threshold count.

    ``tol`` is :data:`TIE_TOL` for float64 scores. Lower-precision float scores,
    such as float32 APs, carry their own rounding, so ``tol`` also covers
    ``TIE_ULPS * eps * |score|`` of their dtype.
    """
    scores = np.asarray(scores)
    dtype = scores.dtype if scores.dtype.kind == "f" else np.dtype(np.float64)
    scores = scores.astype(np.float64)
    eps = float(np.finfo(dtype).eps)
    return scores - np.maximum(TIE_TOL, TIE_ULPS * eps * np.abs(scores))


def resolve_seed(seed: int | None) -> int:
    """Return ``seed`` as an int, or fresh OS entropy when it is ``None``.

    The range is checked where the seed is used, by
    :func:`copairs.nulls.philox.config_key_arrays`.
    """
    if seed is None:
        return int(np.random.SeedSequence().entropy % 2**64)
    return int(seed)


def stream_key(seed: int, obj) -> tuple[np.uint64, np.uint64]:
    """Philox key ``(k0, k1)`` of ``obj``'s draws: a pure function of ``(seed, obj)``.

    The key is the 64-bit blake2b digest of ``repr((seed, obj))``, so ``obj``
    must have a deterministic ``repr`` (tuples of ints).
    """
    digest = hashlib.blake2b(repr((int(seed), obj)).encode(), digest_size=8)
    word = int.from_bytes(digest.digest(), "little")
    return np.uint64(word & 0xFFFFFFFF), np.uint64(word >> 32)


def check_null_size(null_size: int) -> int:
    """Return ``null_size`` as an int; raise unless it is a non-negative integer."""
    if int(null_size) != null_size or null_size < 0:
        raise ValueError(f"null_size must be a non-negative integer, got {null_size}")
    return int(null_size)


def chunk_size(null_size: int, rows: int, budget_bytes: int) -> int:
    """Return the samples per chunk that fit ``rows`` float64 nulls in the budget."""
    return max(1, min(null_size, budget_bytes // (8 * max(rows, 1))))


def chunks(null_size: int, size: int):
    """``(start, size)`` sample chunks, generated lazily."""
    return ((s, min(size, null_size - s)) for s in range(0, null_size, size))


def progress(items, total: int, enabled: bool, desc: str):
    """Wrap ``items`` in a progress bar when ``enabled`` and there are 2+ items."""
    if not enabled or total < 2:
        return items
    from tqdm.autonotebook import tqdm

    return tqdm(items, total=total, desc=desc, leave=False)


def check_pair_counts(count, member, expected, kind: str) -> None:
    """Raise unless profile ``member[i]`` has ``expected[i]`` pairs in ``count``.

    ``count`` is indexed by profile; ``kind`` names the pairs ("positive").
    """
    wrong = np.flatnonzero(count[member] != expected)
    if len(wrong):
        m = member[wrong[0]]
        raise ValueError(
            f"profile {m} has {count[m]} {kind} pairs but {expected[wrong[0]]} in "
            "the AP table; the pairs must index the rows of the frame "
            "average_precision returned, as average_precision(..., "
            "return_pairs=True) gives them, and the AP scores must keep that "
            "frame's index (no reset_index)"
        )


def group_edges(groups, profile, num_pos, pos_pairs):
    """Positive pairs inside each group, as local member pairs.

    Parameters
    ----------
    groups : list of np.ndarray
        Each group's rows of the AP table.
    profile : np.ndarray
        Profile index (row of the pairs' frame) of each AP row.
    num_pos : np.ndarray
        ``n_pos_pairs`` of each AP row; a member whose positive pairs in
        ``pos_pairs`` differ from it raises (:func:`check_pair_counts`).
    pos_pairs : np.ndarray
        The ``(n, 2)`` positive pairs the APs were computed from.

    Returns
    -------
    edges : list of np.ndarray
        Per group, an ``(k, 2)`` int64 array of the local member indices of
        its positive pairs that join two of its members, in ``pos_pairs``
        order.
    leaves : np.ndarray
        Per group, whether a positive pair joins one of its members to a
        profile outside it.
    """
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64).reshape(-1, 2)
    profile = np.asarray(profile, dtype=np.int64)
    size = 1 + max(int(profile.max(initial=-1)), int(pos_pairs.max(initial=-1)))
    rows = np.concatenate(groups) if len(groups) else np.zeros(0, dtype=np.int64)
    degree = np.bincount(pos_pairs.ravel(), minlength=size)
    check_pair_counts(degree, profile[rows], np.asarray(num_pos)[rows], "positive")
    group_of = np.full(size, -1, dtype=np.int64)
    local = np.zeros(size, dtype=np.int64)
    for g, ix in enumerate(groups):
        group_of[profile[ix]] = g
        local[profile[ix]] = np.arange(len(ix))
    ga, gb = group_of[pos_pairs[:, 0]], group_of[pos_pairs[:, 1]]
    inside = (ga >= 0) & (ga == gb)
    leaves = np.zeros(len(groups), dtype=np.bool_)
    leaves[ga[~inside & (ga >= 0)]] = True
    leaves[gb[~inside & (gb >= 0)]] = True
    order = np.argsort(ga[inside], kind="stable")
    ptr = np.searchsorted(ga[inside][order], np.arange(len(groups) + 1))
    edges = local[pos_pairs[inside][order]]
    return [edges[ptr[g] : ptr[g + 1]] for g in range(len(groups))], leaves


def make_head_sort(threshold: int, jit=lambda f: f):
    """Build ``sort_head(vals, n)``, which sorts ``vals[:n]`` in place.

    Up to ``threshold`` entries (the usual few) it sorts by insertion, more
    with NumPy's sort; both give the same sorted values. ``jit`` compiles it
    (e.g. ``numba.njit``) or leaves it Python.
    """

    @jit
    def sort_head(vals, n):
        """Sort ``vals[:n]`` in place."""
        if n > threshold:
            vals[:n].sort()
            return
        for i in range(1, n):
            v = vals[i]
            k = i - 1
            while k >= 0 and vals[k] > v:
                vals[k + 1] = vals[k]
                k -= 1
            vals[k + 1] = v

    return sort_head
