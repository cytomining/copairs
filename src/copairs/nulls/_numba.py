"""Numba kernels of the nulls and their p-value counts (optional dependency).

Imported only through :func:`copairs.nulls.backend._import_numba_backend`. The
AP null sampler, the joint mAP null and the label-permutation mAP null compile
the scalar reference functions of :mod:`copairs.nulls.sampler`,
:mod:`copairs.nulls.joint` and :mod:`copairs.nulls.permutation` unchanged, so
their samples are bitwise identical to the NumPy backend's.
"""

import threading

import numba
import numpy as np

from copairs.nulls import joint, permutation
from copairs.nulls.philox import uniform53, philox4x32
from copairs.nulls.sampler import gap_loop, _make_ap_sample, _make_gap_guided

_philox_nb = numba.njit(inline="always")(philox4x32)
_uniform_nb = numba.njit(inline="always")(uniform53)
_gap_loop_nb = numba.njit(cache=True)(gap_loop)
_ap_sample_nb = numba.njit(cache=True)(
    _make_ap_sample(
        _philox_nb,
        _uniform_nb,
        _gap_loop_nb,
        numba.njit(cache=True)(_make_gap_guided(_gap_loop_nb, numba.njit)),
    )
)
_jit = numba.njit(cache=True)
_joint_sample_nb, _, _ = joint._make_sampler(_philox_nb, _uniform_nb, _jit)
_draw_nb, _group_map_nb = permutation._make_kernels(_philox_nb, _uniform_nb, _jit)
# Samples or draws per parallel task; each task allocates its scratch once.
_JOINT_TASK = 256
_PERMUTATION_TASK = 128
# Held while a parallel kernel that callers may launch from several threads
# runs: Numba's workqueue threading layer aborts on concurrent launches, and
# the others would oversubscribe the CPUs.
_LAUNCH = threading.Lock()


@numba.njit(parallel=True, cache=True)
def ap_nulls(num_pos, total, k0, k1, start, out):
    """Fill ``out[c, t]`` with sample ``start + t`` of configuration ``c``."""
    n_conf, size = out.shape
    for flat in numba.prange(n_conf * size):
        c = flat // size
        t = flat - c * size
        out[c, t] = _ap_sample_nb(num_pos[c], total[c], start + t, k0[c], k1[c])


@numba.njit(inline="always")
def upper_bound(vals, lo, hi, key):
    """First index in sorted ``vals[lo:hi]`` after ``key`` (NumPy order, NaN last)."""
    key_nan = np.isnan(key)
    while lo < hi:
        mid = (lo + hi) >> 1
        v = vals[mid]
        if key_nan or (not np.isnan(v) and v <= key):
            lo = mid + 1
        else:
            hi = mid
    return lo


# Upper bound on the per-block histograms of count_ge.
_HIST_BYTES = 2**28


def count_ge(null, thr, ptr, counts):
    """:func:`copairs.nulls.pvalues._count_ge_host` in parallel over blocks of samples.

    A segment's histogram has a bin per score, so the scores are counted in
    batches of at most ``_HIST_BYTES`` of bins, splitting a configuration's
    sorted scores where they alone would exceed it. Each piece of a segment
    counts its own scores exactly, so the counts do not depend on the batches.
    """
    null = np.ascontiguousarray(null)
    ptr = np.asarray(ptr, dtype=np.int64)
    rows = np.arange(len(ptr) - 1, dtype=np.int64)
    max_bins = max(2, _HIST_BYTES // 8)
    batches = [(0, len(rows))]
    if ptr[-1] - ptr[0] + len(rows) > max_bins:
        rows, ptr = _split_segments(rows, ptr, max_bins - 1)
        batches = _batches(ptr, max_bins)
    for first, end in batches:
        bins = int(ptr[end] - ptr[first]) + end - first
        max_blocks = min(4 * numba.get_num_threads(), max(1, _HIST_BYTES // (8 * bins)))
        _count_ge_kernel(
            null, rows[first:end], thr, ptr[first : end + 1], counts, max_blocks
        )


def _batches(ptr, max_bins):
    """Consecutive segments ``[first, end)`` with at most ``max_bins`` bins each.

    A segment has a bin per score plus one; each must fit ``max_bins`` alone.
    """
    out, first = [], 0
    for end in range(1, len(ptr)):
        if ptr[end] - ptr[first] + end - first > max_bins:
            out.append((first, end - 1))
            first = end - 1
    out.append((first, len(ptr) - 1))
    return out


def _split_segments(rows, ptr, max_scores):
    """Split each segment ``ptr[s]:ptr[s + 1]`` into pieces of ``max_scores`` or fewer."""
    lengths = np.diff(ptr)
    pieces = np.maximum(1, -(-lengths // max_scores))
    piece_rows = np.repeat(rows, pieces)
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    starts = (
        np.repeat(ptr[:-1], pieces) + (np.arange(len(piece_rows)) - first) * max_scores
    )
    return piece_rows, np.append(starts, ptr[-1])


@numba.njit(parallel=True, cache=True)
def _count_ge_kernel(null, rows, thr, ptr, counts, max_blocks):
    """Count segment ``s``'s scores ``thr[ptr[s]:ptr[s + 1]]`` on ``null[rows[s]]``."""
    n_seg = len(rows)
    size = null.shape[1]
    n_blocks = max(1, min((size + (1 << 16) - 1) >> 16, max_blocks))
    block = (size + n_blocks - 1) // n_blocks
    base = ptr[0]
    partial = np.zeros((n_blocks, ptr[n_seg] - base + n_seg), dtype=np.int64)
    for task in numba.prange(n_seg * n_blocks):
        s = task // n_blocks
        b = task - s * n_blocks
        lo, hi = ptr[s], ptr[s + 1]
        if hi == lo:
            continue
        off = s - base
        r = rows[s]
        for t in range(b * block, min(size, (b + 1) * block)):
            partial[b, off + upper_bound(thr, lo, hi, null[r, t])] += 1
    for s in numba.prange(n_seg):
        lo, hi = ptr[s], ptr[s + 1]
        off = lo - base + s
        run = 0
        for q in range(hi - lo - 1, -1, -1):
            for b in range(n_blocks):
                run += partial[b, off + q + 1]
            counts[lo + q] += run


@numba.njit(parallel=True, cache=True)
def group_ge(null, ptr, conf_ix, conf_cnt, n_group, thr, counts):
    """:func:`copairs.nulls.pvalues._group_ge_host` in parallel over sample blocks."""
    n_groups = len(ptr) - 1
    size = null.shape[1]
    block = 1 << 14
    n_blocks = (size + block - 1) // block
    partial = np.zeros(n_groups * n_blocks, dtype=np.int64)
    for flat in numba.prange(n_groups * n_blocks):
        g = flat // n_blocks
        t0 = (flat - g * n_blocks) * block
        t1 = min(t0 + block, size)
        hits = 0
        for t in range(t0, t1):
            acc = 0.0
            for e in range(ptr[g], ptr[g + 1]):
                acc += conf_cnt[e] * null[conf_ix[e], t]
            if acc / n_group[g] >= thr[g]:
                hits += 1
        partial[flat] = hits
    for g in range(n_groups):
        counts[g] += partial[g * n_blocks : (g + 1) * n_blocks].sum()


@numba.njit(parallel=True, cache=True)
def joint_nulls(
    mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1, sigs, start, out
):
    """Fill ``out[r, t]`` with joint sample ``start + t`` of signature ``sigs[r]``."""
    n_sig, size = out.shape
    n_tasks = (size + _JOINT_TASK - 1) // _JOINT_TASK
    max_edge = max(1, np.max(np.diff(edge_ptr)))
    max_pos = max(1, np.max(num_pos))
    for flat in numba.prange(n_sig * n_tasks):
        r = flat // n_tasks
        t0 = (flat - r * n_tasks) * _JOINT_TASK
        ev = np.empty(max_edge, dtype=np.float64)
        vals = np.empty(max_pos, dtype=np.float64)
        for t in range(t0, min(size, t0 + _JOINT_TASK)):
            out[r, t] = _joint_sample_nb(
                sigs[r], start + t, mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr,
                inc_edge, k0, k1, ev, vals,
            )  # fmt: skip


@numba.njit(parallel=True, nogil=True, cache=True)
def _permutation_maps(
    n, n_block, bm_ptr, bm_slot, cr_lo, cr_hi, k0, k1, slot_block, adjm, deg, kmu,
    krr, gptr, grow, smr_ptr, smr, srr_ptr, srr, rmu, rcm, rrr, ranked, start, out,
):  # fmt: skip
    """Fill ``out[t]`` with the mAP of label-permutation draw ``start + t``."""
    size = len(out)
    max_k = max(1, np.max(np.diff(bm_ptr)))
    n_tasks = (size + _PERMUTATION_TASK - 1) // _PERMUTATION_TASK
    for task in numba.prange(n_tasks):
        pm = np.empty(n, dtype=np.int64)
        chosen = np.empty(max_k, dtype=np.int64)
        members = np.empty(n, dtype=np.int64)
        refs = np.empty(n, dtype=np.int64)
        ent = np.empty(n, dtype=np.uint64)
        rank = np.empty(n, dtype=np.int32)
        term = np.empty(n, dtype=np.float64)
        t0 = task * _PERMUTATION_TASK
        for t in range(t0, min(size, t0 + _PERMUTATION_TASK)):
            _draw_nb(
                start + t, n, n_block, bm_ptr, bm_slot, cr_lo, cr_hi, k0, k1, pm, chosen
            )
            out[t] = _group_map_nb(
                pm, n, slot_block, adjm, deg, kmu, krr, gptr, grow, smr_ptr, smr,
                srr_ptr, srr, rmu, rcm, rrr, ranked, members, refs, ent, rank, term,
            )  # fmt: skip


def permutation_maps(layout, start, size, k0, k1):
    """float64 mAPs of draws ``start, ..., start + size - 1`` of ``layout``'s group.

    Threads may call it concurrently (the permutation null runs groups on a
    thread pool); the kernels run one at a time, each on Numba's threads.
    """
    out = np.empty(size, dtype=np.float64)
    map_args = layout.map_args()[1:]  # after n, which draw_args starts with
    with _LAUNCH:
        _permutation_maps(*layout.draw_args(), k0, k1, *map_args, start, out)
    return out
