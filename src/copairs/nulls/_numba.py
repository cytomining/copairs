"""Numba kernels of the AP null sampler and p-value counts (optional dependency).

Imported only through :func:`copairs.nulls.backend._import_numba_backend`. The
sampler compiles the scalar reference functions of
:mod:`copairs.nulls.sampler` unchanged, so its samples are bitwise identical to
the NumPy backend's. The counts' binary search is the AP kernels'
:func:`copairs.fastap.ranking.upper_bound` (nulls may import fastap, not the
reverse).
"""

import numba
import numpy as np

from copairs.nulls import philox, sampler
from copairs.fastap import ranking
from copairs._numba_cache import cached_njit
from copairs.nulls.common import split_segments, segment_batches
from copairs.nulls.philox import uniform53, philox4x32
from copairs.nulls.sampler import gap_loop, _make_ap_sample, _make_gap_guided
from copairs.fastap.ranking import upper_bound

# The kernels compile functions of these files (the counts' upper_bound is in
# ranking.py); Numba's cache would only see this one (see copairs._numba_cache).
_SOURCES = (__file__, philox.__file__, sampler.__file__, ranking.__file__)

_gap_loop_nb = cached_njit(_SOURCES)(gap_loop)
_ap_sample_nb = cached_njit(_SOURCES)(
    _make_ap_sample(
        numba.njit(inline="always")(philox4x32),
        numba.njit(inline="always")(uniform53),
        _gap_loop_nb,
        cached_njit(_SOURCES)(_make_gap_guided(_gap_loop_nb, numba.njit)),
    )
)


@cached_njit(_SOURCES, parallel=True)
def ap_nulls(num_pos, total, k0, k1, start, out):
    """Fill ``out[c, t]`` with sample ``start + t`` of configuration ``c``."""
    n_conf, size = out.shape
    for flat in numba.prange(n_conf * size):
        c = flat // size
        t = flat - c * size
        out[c, t] = _ap_sample_nb(num_pos[c], total[c], start + t, k0[c], k1[c])


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
        rows, ptr = split_segments(rows, ptr, max_bins - 1)
        batches = segment_batches(ptr, max_bins)
    for first, end in batches:
        bins = int(ptr[end] - ptr[first]) + end - first
        max_blocks = min(4 * numba.get_num_threads(), max(1, _HIST_BYTES // (8 * bins)))
        _count_ge_kernel(
            null, rows[first:end], thr, ptr[first : end + 1], counts, max_blocks
        )


@cached_njit(_SOURCES, parallel=True)
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


@cached_njit(_SOURCES, parallel=True)
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
