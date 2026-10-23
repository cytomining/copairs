"""Permutation p-values against the fast AP null, streamed in sample chunks.

Nulls are regenerated chunk by chunk instead of being stored. The chunk of
null samples, the NumPy sampler's temporaries and those of the counts each
stay within ``budget_bytes``, and a chunk is freed before the next is sampled,
so peak memory stays within 2.5x ``budget_bytes`` (tested) whatever
``null_size`` and the number of configurations. Null samples are kept in
float64, the precision copairs computes observed APs in, and a null value
counts against an observed score when ``null >= score - tol`` (see
:func:`tie_thresholds`). The tolerance absorbs rounding between the two
computations, so a null rank list identical to the observed one is counted as
a tie, as in the paper's ``>=`` definition.
"""

import numpy as np

from copairs.nulls.common import (
    chunks,
    progress,
    chunk_size,
    resolve_seed,
    tie_thresholds,
    check_null_size,
)
from copairs.nulls.backend import resolve_backend, _import_numba_backend
from copairs.nulls.sampler import DEFAULT_BUDGET, _ap_nulls, null_plan


def _count_ge_host(null, thr, ptr, counts):
    """``counts[p] += #{x in null[c] : x >= thr[p]}`` for each segment ``c``."""
    for c in range(null.shape[0]):
        lo, hi = ptr[c], ptr[c + 1]
        if hi == lo:
            continue
        q = np.searchsorted(thr[lo:hi], null[c], side="right")
        hist = np.bincount(q, minlength=hi - lo + 1)
        counts[lo:hi] += np.cumsum(hist[::-1])[::-1][1:]


def _prepare(scores, conf_ix, confs, null_size, seed, backend):
    """Thresholds, checked inputs and the null plan shared by the p-value functions."""
    thr = tie_thresholds(scores)
    conf_ix = np.asarray(conf_ix, dtype=np.int64)
    if len(conf_ix) and (conf_ix.min() < 0 or conf_ix.max() >= len(confs)):
        raise ValueError(f"conf_ix must index the {len(confs)} rows of confs")
    null_size = check_null_size(null_size)
    backend = resolve_backend(backend)
    plan = null_plan(confs, resolve_seed(seed))
    return thr, conf_ix, null_size, backend, plan


def ap_pvalues(
    scores: np.ndarray,
    conf_ix: np.ndarray,
    confs: np.ndarray,
    null_size: int,
    seed: int | None,
    backend: str = "auto",
    budget_bytes: int = DEFAULT_BUDGET,
    progress_bar: bool = False,
) -> np.ndarray:
    """P-value of each AP score against the null of its configuration.

    ``p_i = (1 + #{null >= thr_i}) / (null_size + 1)``, where the null is that
    of ``confs[conf_ix[i]]`` and ``thr_i`` is ``scores[i]`` minus the tie
    tolerance of :func:`tie_thresholds`.

    Parameters
    ----------
    scores : np.ndarray
        Observed AP scores.
    conf_ix : np.ndarray
        Row of ``confs`` holding each score's ``(num_pos, total)`` configuration.
    confs : np.ndarray
        ``(n_conf, 2)`` configurations.
    null_size, seed, backend
        See :func:`copairs.nulls.sampler.ap_nulls`.
    budget_bytes : int
        Upper bound on the null samples held in memory at once, and on the
        NumPy sampler's temporaries.
    progress_bar : bool
        Show a progress bar over chunks.
    """
    if len(conf_ix) != len(scores):
        raise ValueError(f"{len(scores)} scores but {len(conf_ix)} conf_ix")
    thr, conf_ix, null_size, backend, plan = _prepare(
        scores, conf_ix, confs, null_size, seed, backend
    )
    order = np.lexsort((thr, conf_ix))
    thr = thr[order]
    ptr = np.searchsorted(conf_ix[order], np.arange(len(confs) + 1))
    # Process configurations in batches whose chunk of samples fits the budget.
    chunk = chunk_size(null_size, 1, budget_bytes)
    batch = chunk_size(len(confs), chunk, budget_bytes)
    batches = range(0, len(confs), batch)
    work = (
        (b, start, size) for b in batches for start, size in chunks(null_size, chunk)
    )
    n_work = len(batches) * len(range(0, null_size, chunk))
    if backend == "numba":
        count = _import_numba_backend().count_ge
    else:
        count = _count_ge_host
    counts = np.zeros(len(thr), dtype=np.int64)
    for b, start, size in progress(work, n_work, progress_bar, "AP null"):
        sl = slice(b, min(b + batch, len(confs)))
        part = tuple(a[sl] for a in plan)
        null = _ap_nulls(part, size, start, backend, np.float64, budget_bytes)
        count(null, thr, ptr[sl.start : sl.stop + 1], counts)
        del null  # free this chunk before sampling the next
    pvals = np.empty(len(thr), dtype=np.float64)
    pvals[order] = (counts + 1) / (null_size + 1)
    return pvals


# Rows of the chunk's size that _group_ge_host holds next to the null chunk:
# the group sum, one member's term and the comparison's booleans.
_GROUP_WORK_ROWS = 3


def _group_ge_host(null, ptr, conf_ix, conf_cnt, n_group, thr, counts):
    acc = np.empty(null.shape[1])
    term = np.empty(null.shape[1])
    for g in range(len(ptr) - 1):
        acc.fill(0.0)
        for e in range(ptr[g], ptr[g + 1]):
            np.multiply(conf_cnt[e], null[conf_ix[e]], out=term)
            acc += term
        np.divide(acc, n_group[g], out=acc)
        counts[g] += np.count_nonzero(acc >= thr[g])


def map_pvalues(
    map_scores: np.ndarray,
    ptr: np.ndarray,
    conf_ix: np.ndarray,
    conf_cnt: np.ndarray,
    confs: np.ndarray,
    null_size: int,
    seed: int | None,
    backend: str = "auto",
    budget_bytes: int = DEFAULT_BUDGET,
    progress_bar: bool = False,
) -> np.ndarray:
    """P-value of each group's mAP against the mean of its members' AP nulls.

    Group ``g`` has ``conf_cnt[e]`` members with configuration
    ``confs[conf_ix[e]]`` for ``e`` in ``ptr[g]:ptr[g + 1]``. Members sharing a
    configuration share null samples, as in copairs' original implementation,
    so the group null at sample ``j`` is
    ``sum_e conf_cnt[e] * AP_{conf_ix[e]}(j) / sum_e conf_cnt[e]`` and
    ``p_g = (1 + #{null_g >= thr_g}) / (null_size + 1)``, with ``thr_g`` from
    :func:`tie_thresholds`.
    """
    ptr = np.asarray(ptr, dtype=np.int64)
    conf_cnt = np.asarray(conf_cnt, dtype=np.int64)
    if len(conf_cnt) != len(conf_ix):
        raise ValueError(f"{len(conf_ix)} conf_ix but {len(conf_cnt)} conf_cnt")
    if (
        len(ptr) < 1
        or ptr[0] != 0
        or ptr[-1] != len(conf_ix)
        or (np.diff(ptr) < 1).any()
    ):
        raise ValueError(
            "ptr must start at 0, end at len(conf_ix) and give every group a member"
        )
    if len(map_scores) != len(ptr) - 1:
        raise ValueError(
            f"{len(map_scores)} mAP scores but {len(ptr) - 1} groups in ptr"
        )
    thr, conf_ix, null_size, backend, plan = _prepare(
        map_scores, conf_ix, confs, null_size, seed, backend
    )
    if len(thr) == 0:
        return np.zeros(0, dtype=np.float64)
    n_group = np.add.reduceat(conf_cnt, ptr[:-1])
    if backend == "numba":
        kernel = _import_numba_backend().group_ge
    else:
        kernel = _group_ge_host
    counts = np.zeros(len(thr), dtype=np.int64)
    # The chunk and the counts' workspace rows share the budget.
    chunk = chunk_size(null_size, len(confs) + _GROUP_WORK_ROWS, budget_bytes)
    n_work = len(range(0, null_size, chunk))
    for start, size in progress(
        chunks(null_size, chunk), n_work, progress_bar, "mAP null"
    ):
        null = _ap_nulls(plan, size, start, backend, np.float64, budget_bytes)
        kernel(null, ptr, conf_ix, conf_cnt, n_group, thr, counts)
        del null  # free this chunk before sampling the next
    return (counts + 1) / (null_size + 1)
