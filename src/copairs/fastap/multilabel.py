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

from copairs.matching import find_pairs
from copairs.nulls.pvalues import upper_bound
from copairs.fastap.ranking import (
    pair_csr,
    rank_keys,
    index_pairs,
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
    """Sorted keys ``i * n + j`` of ``find_pairs(dframe, sameby, diffby)``."""
    # Only the compared columns: DuckDB would convert the label lists too.
    columns = list(dict.fromkeys(sameby + diffby))
    mono = find_pairs(dframe[columns], sameby, diffby).astype(np.int64)
    return np.sort(mono[:, 0] * len(dframe) + mono[:, 1])


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
        return _in_sorted(pairs[:, 0] * len(dframe) + pairs[:, 1], mono)
    keep = np.ones(len(pairs), dtype=bool)
    for code in same:
        keep &= code[pairs[:, 0]] == code[pairs[:, 1]]
    for code in diff:
        keep &= code[pairs[:, 0]] != code[pairs[:, 1]]
    return keep


def shared_label_pairs(dframe, sameby, diffby, multilabel_col):
    """``(pairs, keys, counts)`` of rows sharing a label, or None.

    ``sameby`` excludes ``multilabel_col``; ``diffby`` may hold it, so that
    paired rows' label lists differ. Returns the same pairs as the SQL
    implementation, grouped by label.
    """
    found = multilabel_pairs(dframe, multilabel_col)
    if found is None:
        return None
    keys, pairs, label = found
    if len(sameby) or len(diffby):
        keep = _monolabel_mask(dframe, sameby, diffby, multilabel_col, pairs)
        pairs, label = pairs[keep], label[keep]
    counts = np.bincount(label, minlength=len(keys))
    present = counts > 0
    return pairs.astype(np.uint32), keys[present], counts[present]


@numba.njit(cache=True)
def _all_pairs_except(n, shared):
    """Pairs ``(i < j)`` of ``range(n)`` whose key ``i * n + j`` is not in ``shared``.

    ``shared`` is sorted and unique; candidates are generated in key order and
    merged against it, so no candidate array is materialized.
    """
    out = np.empty((n * (n - 1) // 2 - len(shared), 2), dtype=np.uint32)
    s = 0
    w = 0
    for i in range(n):
        for j in range(i + 1, n):
            if s < len(shared) and shared[s] == i * n + j:
                s += 1
                continue
            out[w, 0] = i
            out[w, 1] = j
            w += 1
    return out


@numba.njit(cache=True)
def _sorted_difference(candidates, shared):
    """Elements of sorted ``candidates`` not in sorted ``shared`` (linear merge)."""
    keep = np.ones(len(candidates), dtype=np.bool_)
    s = 0
    for c in range(len(candidates)):
        while s < len(shared) and shared[s] < candidates[c]:
            s += 1
        if s < len(shared) and shared[s] == candidates[c]:
            keep[c] = False
    return candidates[keep]


def disjoint_label_pairs(dframe, sameby, diffby, multilabel_col):
    """Sorted unique pairs ``(i < j)`` of rows sharing no label, or None."""
    found = multilabel_pairs(dframe, multilabel_col)
    if found is None:
        return None
    _, pairs, _ = found
    n = len(dframe)
    shared = np.unique(pairs[:, 0] * n + pairs[:, 1])
    if not (len(sameby) or len(diffby)):
        return _all_pairs_except(n, shared)
    kept = _sorted_difference(_monolabel_keys(dframe, sameby, diffby), shared)
    return np.stack([kept // n, kept % n], axis=1).astype(np.uint32)


@numba.njit(parallel=True, cache=True)
def _rows_ap(pos_ptr, pos_vals, row_profile, neg_ptr, neg_vals):
    """AP of rows with sorted positive keys and their profile's negatives."""
    n_rows = len(row_profile)
    ap = np.empty(n_rows, dtype=np.float64)
    for r in numba.prange(n_rows):
        lo, hi = pos_ptr[r], pos_ptr[r + 1]
        num_pos = hi - lo
        i = row_profile[r]
        hist = np.zeros(num_pos + 1, dtype=np.int64)
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
        Number of profiles.

    Returns
    -------
    tuple of np.ndarray
        ``label``, ``profile``, ``ap``, ``num_pos`` and ``total`` per row,
        ordered by label and then profile, like copairs' per-label loop.
    """
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64).reshape(-1, 2)
    label = np.repeat(np.arange(len(pos_counts)), pos_counts)
    ends = pos_pairs.ravel()
    end_label = np.repeat(label, 2)
    end_keys = np.repeat(rank_keys(pos_sims), 2)
    row_key = end_label * n + ends
    if len(row_key) == 0 or row_key.max() < 2**32:
        order = np.argsort((row_key.astype(np.uint64) << 32) | sortable_keys(end_keys))
    else:
        order = np.lexsort((end_keys, row_key))
    row_key, end_keys = row_key[order], end_keys[order]
    rows, pos_start = np.unique(row_key, return_index=True)
    pos_ptr = np.append(pos_start, len(row_key)).astype(np.int64)
    neg_ptr, neg_vals = pair_csr(index_pairs(neg_pairs), np.asarray(neg_sims), n)
    profile = rows % n
    ap = _rows_ap(pos_ptr, end_keys, profile, neg_ptr, neg_vals)
    num_pos = np.diff(pos_ptr)
    return rows // n, profile, ap, num_pos, num_pos + np.diff(neg_ptr)[profile]
