"""Multilabel pair matching and AP through an inverted label index.

Pairs that share a label are enumerated per label from the label's members,
so their cost is the number of such pairs. Pairs that share no label are all
other candidate pairs; candidates are every pair, or the monolabel pairs of
the remaining ``sameby``/``diffby`` columns. Rows whose labels cannot be
indexed exactly (missing values inside lists, duplicate labels in a row,
mixed label types, a non-default index) return None so callers can fall back
to the SQL implementation.
"""

import numba
import numpy as np
import pandas as pd

from copairs.fastap import ranking
from copairs.matching import _key_pairs, _pair_keys, find_pairs
from copairs._numba_cache import cached_njit
from copairs.fastap.ranking import (
    pair_csr,
    pair_sims,
    rank_keys,
    index_pairs,
    upper_bound,
    profile_count,
    sortable_keys,
    ap_from_counts,
)


def label_members(labels: pd.Series):
    """Sorted distinct labels and the CSR of rows holding each, or None."""
    if not all(isinstance(v, (list, tuple, np.ndarray)) for v in labels):
        return None
    flat = labels.reset_index(drop=True).explode()
    values = flat.to_numpy()
    missing = pd.isna(values)
    # explode gives one missing value per empty cell; any other is a None/NaN label.
    if missing.sum() != (labels.map(len) == 0).sum():
        return None
    values, rows = values[~missing], flat.index.to_numpy()[~missing]
    try:
        inv, keys = pd.factorize(values, sort=True)
    except TypeError:  # labels of mixed, unorderable types
        return None
    keys = np.asarray(keys, dtype=object)
    if len({type(k) for k in keys}) > 1:
        return None
    if len(keys) and not isinstance(keys[0], str):
        # Numeric labels as a numeric array, like SQL; DuckDB stores Python ints
        # as INTEGER while they all fit.
        fits = type(keys[0]) is int and -(2**31) <= keys[0] and keys[-1] < 2**31
        keys = np.array(keys.tolist(), dtype=np.int32 if fits else None)
    row_key = inv.astype(np.int64) * len(labels) + rows
    order = np.argsort(row_key, kind="stable")
    if (np.diff(row_key[order]) == 0).any():
        return None  # a row lists the same label twice
    ptr = np.searchsorted(inv[order], np.arange(len(keys) + 1))
    return keys, ptr, rows[order]


def _label_pairs(ptr, rows):
    """Pairs ``(i < j)`` of rows sharing each label, grouped by label.

    Within a label, pairs are in ``np.triu_indices`` order of its sorted rows.
    """
    size = np.diff(ptr)
    local = np.arange(len(rows)) - np.repeat(ptr[:-1], size)
    later = np.repeat(size, size) - 1 - local  # members after each one in its label
    first = np.repeat(np.arange(len(rows)), later)
    step = np.arange(len(first)) - np.repeat(np.cumsum(later) - later, later)
    pairs = np.stack([rows[first], rows[first + 1 + step]], axis=1)
    label = np.repeat(np.repeat(np.arange(len(size)), size), later)
    return pairs, label


def _in_sorted(keys, sorted_keys):
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    loc = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[loc] == keys


def multilabel_pairs(dframe, multilabel_col):
    """``(keys, pairs, label)`` of rows sharing each label, or None if unsupported.

    Pairs are grouped by label (see :func:`_label_pairs`).
    """
    if not dframe.index.equals(pd.RangeIndex(len(dframe))):
        return None
    members = label_members(dframe[multilabel_col])
    if members is None:
        return None
    keys, ptr, rows = members
    pairs, label = _label_pairs(ptr, rows)
    return keys, pairs, label


def _monolabel_keys(dframe, sameby, diffby):
    """Sorted :func:`~copairs.matching._pair_keys` of ``find_pairs(dframe, ...)``."""
    # Only the compared columns: DuckDB would convert the label lists too.
    columns = list(dict.fromkeys(sameby + diffby))
    return np.sort(_pair_keys(find_pairs(dframe[columns], sameby, diffby)))


def _column_codes(dframe, columns, multilabel_col):
    """Integer codes of each column's values, or None if SQL must compare them.

    find_pairs compares in DuckDB, where missing values never match and mixed
    Python types are converted, so such columns keep the SQL comparison.
    """
    codes = []
    for c in columns:
        col = dframe[c]
        if c == multilabel_col:  # label lists, already checked, compare whole
            col = col.map(tuple)
        elif col.isna().any() or pd.api.types.infer_dtype(col).startswith("mixed"):
            return None
        codes.append(pd.factorize(col)[0])
    return codes


def _monolabel_mask(dframe, sameby, diffby, multilabel_col, pairs):
    """Which ``pairs`` also satisfy the monolabel ``sameby``/``diffby`` columns."""
    same = _column_codes(dframe, sameby, multilabel_col)
    diff = _column_codes(dframe, diffby, multilabel_col)
    if same is None or diff is None:
        mono = _monolabel_keys(dframe, sameby, diffby)
        return _in_sorted(_pair_keys(pairs), mono)
    keep = np.ones(len(pairs), dtype=bool)
    for code in same:
        keep &= code[pairs[:, 0]] == code[pairs[:, 1]]
    for code in diff:
        keep &= code[pairs[:, 0]] != code[pairs[:, 1]]
    return keep


def indexed_pairs(dframe, sameby, diffby, multilabel_col, found):
    """:func:`copairs.matching.find_pairs_multilabel` through a label index.

    ``found`` is ``multilabel_pairs(dframe, multilabel_col)``, not None.
    """
    rest_same = [c for c in sameby if c != multilabel_col]
    if multilabel_col in sameby:
        # Also in diffby, as in SQL, the label lists must differ as a whole.
        return shared_label_pairs(dframe, rest_same, diffby, multilabel_col, found)
    rest_diff = [c for c in diffby if c != multilabel_col]
    return disjoint_label_pairs(dframe, rest_same, rest_diff, multilabel_col, found)


def shared_label_pairs(dframe, sameby, diffby, multilabel_col, found):
    """``(pairs, keys, counts)`` of rows sharing a label.

    ``found`` is ``multilabel_pairs(dframe, multilabel_col)``. ``sameby``
    excludes ``multilabel_col``; ``diffby`` may hold it, so that paired rows'
    label lists differ. Returns the same pairs as the SQL implementation,
    grouped by label.
    """
    keys, pairs, label = found
    if len(sameby) or len(diffby):
        keep = _monolabel_mask(dframe, sameby, diffby, multilabel_col, pairs)
        pairs, label = pairs[keep], label[keep]
    counts = np.bincount(label, minlength=len(keys))
    present = counts > 0
    return pairs.astype(np.uint32), keys[present], counts[present]


@numba.njit(parallel=True, cache=True)
def _all_pairs_except(n, shared_ptr, shared_j, offsets):
    """Pairs ``(i < j)`` of ``range(n)`` except the shared pairs, by row in parallel.

    Row ``i``'s shared pairs are ``(i, shared_j[s])`` for ``s`` in
    ``shared_ptr[i]:shared_ptr[i + 1]``, sorted by ``j``; its candidates are
    generated in the same order and merged against them, so no candidate
    array is materialized. Its output starts at ``offsets[i]``.
    """
    out = np.empty((offsets[n], 2), dtype=np.uint32)
    for row in numba.prange(n):
        i = np.int64(row)  # prange's index is unsigned; uint64 with int64 is float
        s, end, w = shared_ptr[i], shared_ptr[i + 1], offsets[i]
        for j in range(i + 1, n):
            if s < end and shared_j[s] == j:
                s += 1
                continue
            out[w, 0] = i
            out[w, 1] = j
            w += 1
    return out


def disjoint_label_pairs(dframe, sameby, diffby, multilabel_col, found):
    """Sorted unique pairs ``(i < j)`` of rows sharing no label.

    ``found`` is ``multilabel_pairs(dframe, multilabel_col)``.
    """
    _, pairs, _ = found
    shared = np.unique(_pair_keys(pairs))
    if not (len(sameby) or len(diffby)):
        n = len(dframe)
        # Row i's shared keys are those in [i << 32, (i + 1) << 32).
        ptr = np.searchsorted(
            shared, np.arange(n + 1, dtype=np.uint64) << np.uint64(32)
        )
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum((n - 1 - np.arange(n)) - np.diff(ptr), out=offsets[1:])
        return _all_pairs_except(n, ptr, _key_pairs(shared)[:, 1], offsets)
    mono = _monolabel_keys(dframe, sameby, diffby)
    return _key_pairs(mono[~_in_sorted(mono, shared)])


# _rows_ap compiles functions of ranking.py, which Numba's cache would not see
# (see copairs._numba_cache).
@cached_njit((__file__, ranking.__file__), parallel=True)
def _rows_ap(pos_ptr, pos_vals, row_profile, neg_ptr, neg_vals, max_pos, n_tasks):
    """AP of rows with sorted positive keys and their profile's negatives.

    Each of ``n_tasks`` parallel tasks scores a block of rows with one
    histogram of ``max_pos + 1`` bins, the most positives of any row.
    """
    n_rows = len(row_profile)
    ap = np.empty(n_rows, dtype=np.float64)
    step = (n_rows + n_tasks - 1) // n_tasks
    for task in numba.prange(n_tasks):
        hist = np.empty(max_pos + 1, dtype=np.int64)
        for r in range(task * step, min(n_rows, (task + 1) * step)):
            lo, hi = pos_ptr[r], pos_ptr[r + 1]
            num_pos = hi - lo
            i = row_profile[r]
            hist[: num_pos + 1] = 0
            for e in range(neg_ptr[i], neg_ptr[i + 1]):
                hist[upper_bound(pos_vals, lo, hi, neg_vals[e]) - lo] += 1
            ap[r] = ap_from_counts(hist, 0, num_pos)
    return ap


def multilabel_ap(pos_pairs, pos_sims, pos_counts, neg_pairs, neg_sims, n):
    """AP of every (label, profile) with positives for that label.

    Parameters
    ----------
    pos_pairs, pos_sims : np.ndarray
        Positive pairs grouped by label, ``pos_counts[c]`` pairs for label ``c``.
    neg_pairs, neg_sims : np.ndarray
        Unique negative pairs; a profile's negatives are shared by its labels.
    n : int
        Number of profiles, more than any index in the pairs.

    Returns
    -------
    tuple of np.ndarray
        ``label``, ``profile``, ``ap``, ``num_pos`` and ``total`` per row,
        ordered by label and then profile, like copairs' per-label loop.

    Raises
    ------
    ValueError, IndexError
        For malformed pairs, similarities or ``n``, as in
        :func:`copairs.fastap.ap_from_pairs`, or ``pos_counts`` that do not
        sum to the number of positive pairs.
    """
    n = profile_count(n)
    pos_pairs, neg_pairs = index_pairs(pos_pairs, n), index_pairs(neg_pairs, n)
    pos_sims, neg_sims = pair_sims(pos_sims, pos_pairs), pair_sims(neg_sims, neg_pairs)
    pos_counts = np.asarray(pos_counts)
    if pos_counts.ndim != 1 or pos_counts.sum() != len(pos_pairs):
        raise ValueError(
            f"pos_counts must split the {len(pos_pairs)} positive pairs by label"
        )
    pos_pairs = pos_pairs.astype(np.int64, copy=False)
    label = np.repeat(np.arange(len(pos_counts)), pos_counts)
    ends = pos_pairs.ravel()
    end_label = np.repeat(label, 2)
    end_keys = np.repeat(rank_keys(pos_sims), 2)
    row_key = end_label * n + ends
    # One argsort of packed (row, key) words: 0.9 ms instead of lexsort's 7.1 ms
    # on 78k positive ends (2000 profiles, pr4_bench multi), 11 vs 103 ms at 10x.
    if len(row_key) == 0 or row_key.max() < 2**32:
        order = np.argsort((row_key.astype(np.uint64) << 32) | sortable_keys(end_keys))
    else:
        order = np.lexsort((end_keys, row_key))
    row_key, end_keys = row_key[order], end_keys[order]
    rows, pos_start = np.unique(row_key, return_index=True)
    pos_ptr = np.append(pos_start, len(row_key)).astype(np.int64)
    neg_ptr, neg_vals = pair_csr(neg_pairs, neg_sims, n)
    profile = rows % n
    num_pos = np.diff(pos_ptr)
    max_pos = int(num_pos.max()) if len(num_pos) else 0
    n_tasks = max(1, min(len(rows), 4 * numba.get_num_threads()))
    ap = _rows_ap(pos_ptr, end_keys, profile, neg_ptr, neg_vals, max_pos, n_tasks)
    return rows // n, profile, ap, num_pos, num_pos + np.diff(neg_ptr)[profile]
