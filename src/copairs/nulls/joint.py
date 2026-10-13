"""The joint mAP null: a group's member APs sampled together.

Under the null every pair's similarity is an independent uniform. A positive
pair between two members of a group (an edge) is one uniform that sits in both
members' rank lists; everything else a member ranks is private to it. Each
member's marginal is the AP null of its configuration, but members sharing
edges move together (see :func:`copairs.compute.random_map`, the NumPy
reference this module reproduces in distribution).

A sample draws the group's edge uniforms, then for each member its private
positives and, walking down its sorted positives, the number of its negatives
between consecutive positives as a binomial draw. The cost per sample is
therefore independent of how many negatives a member ranks. All randomness
comes from Philox counters: sample ``j`` of a signature is a pure function of
``(seed, signature, j)``, so results do not depend on threads or chunking.
"""

import math
import hashlib

import numba
import numpy as np

from copairs.nulls.philox import M32, S32, uniform53, philox4x32
from copairs.nulls.pvalues import (
    DEFAULT_BUDGET,
    _chunks,
    _progress,
    _chunk_size,
    resolve_seed,
    tie_thresholds,
    _count_ge_numba,
    _check_null_size,
)

_philox = numba.njit(inline="always")(philox4x32)
_uniform = numba.njit(inline="always")(uniform53)

# Below this mean, binomials are drawn by inversion; above it by BTRS.
_INVERSION_MEAN = 10.0
# log(k!) for small k; larger k use Stirling's series. libm's lgamma is not
# used because it writes the global signgam, which serializes threads.
_LOG_FACTORIAL = np.array([math.lgamma(k + 1.0) for k in range(256)])
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
# Samples per parallel task; each task allocates its scratch once.
_TASK = 256


@numba.njit(inline="always")
def _pair(c0, c1, block, stream, k0, k1):
    """Two uniforms of block ``block`` of stream ``stream`` of sample ``(c0, c1)``."""
    w0, w1, w2, w3 = _philox(c0, c1, np.uint64(block), np.uint64(stream), k0, k1)
    return _uniform(w0, w1), _uniform(w2, w3)


@numba.njit(inline="always")
def _log_factorial(k):
    """log(k!) of an integer-valued float ``k >= 0`` (relative error < 1e-15)."""
    if k < 256:
        return _LOG_FACTORIAL[int(k)]
    x = k + 1.0
    r = 1.0 / (x * x)
    return (
        (x - 0.5) * math.log(x)
        - x
        + _HALF_LOG_2PI
        + (1.0 / 12.0 - r * (1.0 / 360.0 - r / 1260.0)) / x
    )


@numba.njit(cache=True)
def binomial(n, p, c0, c1, block, stream, k0, k1):
    """Binomial(``n``, ``p``) from the stream's blocks starting at ``block``.

    Returns ``(draw, next_block)``. Inversion for small means (BINV), else
    Hörmann's (1993) transformed rejection with squeeze (BTRS); both exact up
    to floating-point rounding.
    """
    if n <= 0 or p <= 0.0:
        return 0, block
    if p >= 1.0:
        return n, block
    flip = p > 0.5
    q = 1.0 - p if flip else p
    if n * q < _INVERSION_MEAN:
        u, _ = _pair(c0, c1, block, stream, k0, k1)
        block += 1
        s = q / (1.0 - q)
        a = (n + 1) * s
        r = (1.0 - q) ** n
        x = 0
        while u > r and x < n:
            u -= r
            x += 1
            r *= a / x - s
    else:
        spq = math.sqrt(n * q * (1.0 - q))
        b = 1.15 + 2.53 * spq
        a = -0.0873 + 0.0248 * b + 0.01 * q
        c = n * q + 0.5
        v_r = 0.92 - 4.2 / b
        alpha = (2.83 + 5.1 / b) * spq
        lpq = math.log(q / (1.0 - q))
        m = math.floor((n + 1) * q)
        h = _log_factorial(m) + _log_factorial(n - m)
        while True:
            u, v = _pair(c0, c1, block, stream, k0, k1)
            block += 1
            u -= 0.5
            us = 0.5 - abs(u)
            if us <= 0.0:
                continue
            kf = math.floor((2.0 * a / us + b) * u + c)
            if kf < 0.0 or kf > n:
                continue
            if us >= 0.07 and v <= v_r:
                break
            v = math.log(v * alpha / (a / (us * us) + b))
            if v <= h - _log_factorial(kf) - _log_factorial(n - kf) + (kf - m) * lpq:
                break
        x = int(kf)
    return (n - x if flip else x), block


@numba.njit(inline="always")
def _sort_head(vals, n):
    """Sort ``vals[:n]`` in place (insertion sort for the usual few positives)."""
    if n > 32:
        vals[:n].sort()
        return
    for i in range(1, n):
        v = vals[i]
        k = i - 1
        while k >= 0 and vals[k] > v:
            vals[k + 1] = vals[k]
            k -= 1
        vals[k + 1] = v


@numba.njit(inline="always")
def _member_ap(vals, num_pos, num_neg, c0, c1, stream, k0, k1):
    """AP of a member whose positive uniforms are ``vals[:num_pos]`` (sorted here)."""
    _sort_head(vals, num_pos)
    remaining = num_neg
    prev = 1.0
    above = 0
    acc = 0.0
    block = 0
    for t in range(num_pos):
        v = vals[num_pos - 1 - t]  # descending
        if remaining > 0 and prev > 0.0:
            x, block = binomial(
                remaining, (prev - v) / prev, c0, c1, block, stream, k0, k1
            )
            above += x
            remaining -= x
        prev = v
        acc += (t + 1) / (t + 1 + above)
    return acc / num_pos


@numba.njit(cache=True)
def joint_sample(
    s, j, mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1, ev, vals
):
    """Return the mAP of sample ``j`` of signature ``s`` (``ev``, ``vals``: scratch)."""
    c0 = np.uint64(j) & M32
    c1 = np.uint64(j) >> S32
    key0, key1 = k0[s], k1[s]
    n_edge = edge_ptr[s + 1] - edge_ptr[s]
    for e in range(0, n_edge, 2):
        u0, u1 = _pair(c0, c1, e // 2, 0, key0, key1)
        ev[e] = u0
        if e + 1 < n_edge:
            ev[e + 1] = u1
    m0, m1 = mem_ptr[s], mem_ptr[s + 1]
    total = 0.0
    for i in range(m0, m1):
        p = num_pos[i]
        shared = inc_ptr[i + 1] - inc_ptr[i]
        for t in range(shared):
            vals[t] = ev[inc_edge[inc_ptr[i] + t]]
        # Private positives come first in the member's stream, then binomials.
        stream = 1 + 2 * (i - m0)
        for t in range(shared, p, 2):
            u0, u1 = _pair(c0, c1, (t - shared) // 2, stream, key0, key1)
            vals[t] = u0
            if t + 1 < p:
                vals[t + 1] = u1
        total += _member_ap(vals, p, num_neg[i], c0, c1, stream + 1, key0, key1)
    return total / (m1 - m0)


@numba.njit(parallel=True, cache=True)
def _joint_nulls(
    mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1, s0, start, out
):
    n_sig, size = out.shape
    n_tasks = (size + _TASK - 1) // _TASK
    max_edge = max(1, np.max(np.diff(edge_ptr)))
    max_pos = max(1, np.max(num_pos))
    for flat in numba.prange(n_sig * n_tasks):
        r = flat // n_tasks
        t0 = (flat - r * n_tasks) * _TASK
        ev = np.empty(max_edge, dtype=np.float64)
        vals = np.empty(max_pos, dtype=np.float64)
        for t in range(t0, min(size, t0 + _TASK)):
            out[r, t] = joint_sample(
                s0 + r,
                start + t,
                mem_ptr,
                num_pos,
                num_neg,
                edge_ptr,
                inc_ptr,
                inc_edge,
                k0,
                k1,
                ev,
                vals,
            )


def canonical_signature(members, edges):
    """``(members, edges)`` with members sorted when every pair is an edge.

    A complete edge set is unchanged by reordering members, so replicate
    groups that differ only in row order share a signature (and a null).
    """
    members = tuple((int(a), int(b)) for a, b in members)
    edges = tuple(
        sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in edges})
    )
    n = len(members)
    if len(edges) == n * (n - 1) // 2:
        members = tuple(sorted(members))
    return members, edges


def signature_keys(seed: int, signatures) -> tuple[np.ndarray, np.ndarray]:
    """Philox keys of each signature: a pure function of ``(seed, signature)``."""
    k0 = np.empty(len(signatures), dtype=np.uint64)
    k1 = np.empty(len(signatures), dtype=np.uint64)
    for s, sig in enumerate(signatures):
        digest = hashlib.blake2b(repr((int(seed), sig)).encode(), digest_size=8)
        word = int.from_bytes(digest.digest(), "little")
        k0[s], k1[s] = word & 0xFFFFFFFF, word >> 32
    return k0, k1


def joint_plan(signatures, seed: int) -> tuple[np.ndarray, ...]:
    """Flat arrays of canonical ``signatures`` for :func:`joint_nulls`."""
    mem_ptr, edge_ptr, inc_ptr = [0], [0], [0]
    num_pos, num_neg, inc_edge = [], [], []
    for members, edges in signatures:
        n = len(members)
        incident = [[] for _ in range(n)]
        for e, (i, j) in enumerate(edges):
            if i == j or not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"edge {(i, j)} does not join two distinct members")
            incident[i].append(e)
            incident[j].append(e)
        for i, (p, m) in enumerate(members):
            if p < 1 or len(incident[i]) > p or m < 0:
                raise ValueError(
                    f"member {i} has {p} positives, {m} negatives and "
                    f"{len(incident[i])} positive pairs inside the group"
                )
            num_pos.append(p)
            num_neg.append(m)
            inc_edge.extend(incident[i])
            inc_ptr.append(len(inc_edge))
        mem_ptr.append(len(num_pos))
        edge_ptr.append(edge_ptr[-1] + len(edges))
    k0, k1 = signature_keys(seed, signatures)
    as64 = lambda a: np.asarray(a, dtype=np.int64)  # noqa: E731
    return (
        as64(mem_ptr),
        as64(num_pos),
        as64(num_neg),
        as64(edge_ptr),
        as64(inc_ptr),
        as64(inc_edge),
        k0,
        k1,
    )


def joint_nulls(plan, null_size: int, start: int = 0, rows=None) -> np.ndarray:
    """float64 mAP samples ``start, ..., start + null_size - 1`` of each signature.

    ``rows`` is a ``slice`` of the plan's signatures (default all); the result
    has one row per signature in it.
    """
    n_sig = len(plan[0]) - 1
    lo, hi, _ = (rows or slice(0, n_sig)).indices(n_sig)
    out = np.empty((max(0, hi - lo), null_size), dtype=np.float64)
    if out.size:
        _joint_nulls(*plan, lo, start, out)
    return out


def joint_pvalues(
    scores: np.ndarray,
    sig_ix: np.ndarray,
    signatures,
    null_size: int,
    seed: int | None,
    budget_bytes: int = DEFAULT_BUDGET,
    progress_bar: bool = False,
) -> np.ndarray:
    """P-value of each group's mAP against the joint null of its signature.

    Group ``g`` has signature ``signatures[sig_ix[g]]``, a canonical
    ``(members, edges)`` pair (:func:`canonical_signature`) of its members'
    ``(num_pos, num_neg)`` and its positive pairs between local member
    indices. ``p_g = (1 + #{null >= thr_g}) / (null_size + 1)`` with ``thr_g``
    from :func:`~copairs.nulls.tie_thresholds`. Nulls are generated in chunks
    of at most ``budget_bytes`` and not stored.
    """
    thr = tie_thresholds(scores)
    sig_ix = np.asarray(sig_ix, dtype=np.int64)
    if len(sig_ix) != len(thr):
        raise ValueError(f"{len(thr)} scores but {len(sig_ix)} sig_ix")
    if len(sig_ix) and (sig_ix.min() < 0 or sig_ix.max() >= len(signatures)):
        raise ValueError(f"sig_ix must index the {len(signatures)} signatures")
    null_size = _check_null_size(null_size)
    plan = joint_plan(signatures, resolve_seed(seed))
    order = np.lexsort((thr, sig_ix))
    thr = thr[order]
    ptr = np.searchsorted(sig_ix[order], np.arange(len(signatures) + 1))
    chunk = _chunk_size(null_size, 1, budget_bytes)
    batch = max(1, budget_bytes // (8 * chunk))
    batches = range(0, len(signatures), batch)
    work = (
        (b, start, size) for b in batches for start, size in _chunks(null_size, chunk)
    )
    n_work = len(batches) * -(-null_size // chunk)
    counts = np.zeros(len(thr), dtype=np.int64)
    for b, start, size in _progress(work, n_work, progress_bar, "joint mAP null"):
        rows = slice(b, min(b + batch, len(signatures)))
        null = joint_nulls(plan, size, start, rows)
        _count_ge_numba(null, thr, ptr[rows.start : rows.stop + 1], counts)
    pvals = np.empty(len(thr), dtype=np.float64)
    pvals[order] = (counts + 1) / (null_size + 1)
    return pvals
