"""Numba kernels of the AP null sampler and p-value counts (optional dependency).

Imported only through :func:`copairs.nulls.backend._import_numba_backend`. The
sampler compiles the scalar reference functions of
:mod:`copairs.nulls.sampler` unchanged, so its samples are bitwise identical to
the NumPy backend's.
"""

import numba
import numpy as np

from copairs.nulls.philox import uniform53, philox4x32
from copairs.nulls.sampler import gap_loop, _make_ap_sample, _make_gap_guided

_gap_loop_nb = numba.njit(cache=True)(gap_loop)
_ap_sample_nb = numba.njit(cache=True)(
    _make_ap_sample(
        numba.njit(inline="always")(philox4x32),
        numba.njit(inline="always")(uniform53),
        _gap_loop_nb,
        numba.njit(cache=True)(_make_gap_guided(_gap_loop_nb, numba.njit)),
    )
)


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
    """:func:`copairs.nulls.pvalues._count_ge_host` in parallel over blocks of samples."""
    bins = int(ptr[-1] - ptr[0]) + null.shape[0]
    max_blocks = min(4 * numba.get_num_threads(), max(1, _HIST_BYTES // (8 * bins)))
    _count_ge_kernel(np.ascontiguousarray(null), thr, ptr, counts, max_blocks)


@numba.njit(parallel=True, cache=True)
def _count_ge_kernel(null, thr, ptr, counts, max_blocks):
    n_conf, size = null.shape
    n_blocks = max(1, min((size + (1 << 16) - 1) >> 16, max_blocks))
    block = (size + n_blocks - 1) // n_blocks
    base = ptr[0]
    partial = np.zeros((n_blocks, ptr[n_conf] - base + n_conf), dtype=np.int64)
    for task in numba.prange(n_conf * n_blocks):
        c = task // n_blocks
        b = task - c * n_blocks
        lo, hi = ptr[c], ptr[c + 1]
        if hi == lo:
            continue
        off = c - base
        for t in range(b * block, min(size, (b + 1) * block)):
            partial[b, off + upper_bound(thr, lo, hi, null[c, t])] += 1
    for c in numba.prange(n_conf):
        lo, hi = ptr[c], ptr[c + 1]
        off = lo - base + c
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
