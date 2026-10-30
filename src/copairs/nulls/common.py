"""Helpers shared by the mAP nulls: tie thresholds, seeds and null sizes.

A null value counts against an observed score when ``null >= score - tol``
(see :func:`tie_thresholds`). The tolerance absorbs rounding between the two
computations, so a null rank list identical to the observed one is counted as
a tie, as in the ``>=`` definition of a permutation p-value.
"""

import hashlib
from operator import index as integer_index

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

    Raises unless ``seed`` is ``None`` or in ``[0, 2**64)`` (:func:`check_seed`).
    """
    check_seed(seed)
    if seed is None:
        return int(np.random.SeedSequence().entropy % 2**64)
    return int(seed)


def check_seed(seed: int | None) -> None:
    """Raise unless ``seed`` is ``None`` or a Philox seed, an integer in ``[0, 2**64)``."""
    if seed is not None and (int(seed) != seed or not 0 <= seed < 2**64):
        raise ValueError(f"seed must be an integer in [0, 2**64), got {seed}")


def stream_key(seed: int, obj, suffix: str = "") -> tuple[np.uint64, np.uint64]:
    """Philox key from the deterministic representation of ``(seed, obj)`` and suffix."""
    text = repr((int(seed), obj)) + suffix
    digest = hashlib.blake2b(text.encode(), digest_size=8)
    word = int.from_bytes(digest.digest(), "little")
    return np.uint64(word & 0xFFFFFFFF), np.uint64(word >> 32)


def check_max_workers(max_workers):
    """Validate concurrency before dispatch, including scores-only calls."""
    if max_workers is None:
        return None
    try:
        if isinstance(max_workers, (bool, np.bool_)):
            raise TypeError
        workers = integer_index(max_workers)
        if workers <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_workers must be a positive integer or None") from None
    return workers


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


def check_pairs(pairs, name="pairs") -> np.ndarray:
    """Validate an undirected pair array and return its canonical ordering."""
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must be a (n, 2) integer array indexing the original rows"
        )
    if pairs.size and (pairs.min() < 0 or pairs.max() > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must index the rows with non-negative int64 values")
    # Preserve the usual already-canonical arrays and their compact dtype.
    # Large negative-pair tables must not acquire full int64 copies just to validate.
    if pairs.dtype == np.uint64:
        pairs = pairs.astype(np.int64)
    ordered = pairs if (pairs[:, 0] <= pairs[:, 1]).all() else np.sort(pairs, axis=1)
    a, b = ordered.T
    if (a == b).any():
        raise ValueError(f"{name} cannot contain self pairs")
    increasing = (a[1:] > a[:-1]) | ((a[1:] == a[:-1]) & (b[1:] > b[:-1]))
    if increasing.all():
        return ordered
    unique = np.unique(ordered, axis=0)
    if len(unique) != len(ordered):
        raise ValueError(f"{name} cannot contain duplicate pairs")
    return unique


def check_pair_counts(count, member, expected, kind: str, label=None) -> None:
    """Validate member-aligned pair counts, optionally identifying a multilabel group."""
    wrong = np.flatnonzero(count != expected)
    if not len(wrong):
        return
    i = wrong[0]
    suffix = "" if label is None else f" for {label!r}"
    raise ValueError(
        f"profile {member[i]} has {count[i]} {kind} pairs{suffix} but {expected[i]} in "
        "the AP table; use the pairs returned with this AP frame and preserve "
        "its original index (no reset_index)"
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
    pos_pairs = check_pairs(pos_pairs, "positive pairs")
    profile = np.asarray(profile, dtype=np.int64)
    size = 1 + max(
        int(profile.max(initial=-1)), int(pos_pairs.max()) if pos_pairs.size else -1
    )
    rows = np.concatenate(groups) if len(groups) else np.zeros(0, dtype=np.int64)
    degree = np.bincount(pos_pairs.ravel(), minlength=size)
    check_pair_counts(
        degree[profile[rows]], profile[rows], np.asarray(num_pos)[rows], "positive"
    )
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
