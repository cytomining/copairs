"""The joint mAP null: a group's member APs sampled together.

Under the null every pair's similarity is an independent uniform. A positive
pair between two members of a group (an edge) is one uniform that sits in both
members' rank lists; everything else a member ranks is private to it. Each
member's marginal is the AP null of its configuration, but members sharing
edges move together. The tests keep #151's sampler, which draws from
``numpy.random.Generator``, as the reference this module reproduces in
distribution.

A sample draws the group's edge uniforms, then for each member its private
positives and, walking down its sorted positives, the number of its negatives
between consecutive positives as a binomial draw. The cost per sample is
therefore independent of how many negatives a member ranks. All randomness
comes from Philox counters: sample ``j`` of a signature is a pure function of
``(seed, signature, j)``, so results do not depend on chunking or order.

Sample ``j`` is defined by the scalar functions :func:`_make_sampler` builds,
which use only IEEE additions, subtractions, multiplications, divisions,
square roots and floors, in a fixed order: logarithms come from a fixed series
and integer powers from repeated squaring, not from the math library, whose
last bits can differ between libm, NumPy's SIMD loops and compiled code.
:func:`joint_nulls` performs the same operations vectorized over samples, so
both give identical samples, as would the scalar functions compiled.
"""

import math

import numpy as np

from copairs.nulls.common import (
    chunks,
    progress,
    stream_key,
    group_edges,
    resolve_seed,
    make_head_sort,
    tie_thresholds,
    check_null_size,
)
from copairs.nulls.philox import (
    M32,
    S32,
    uniform53,
    philox4x32,
    uniform_pair,
    make_uniform_pair,
)

# Below this mean, binomials are drawn by inversion; above it by BTRS.
_INVERSION_MEAN = 10.0
# log(k!) for small k; larger k use Stirling's series. The table is data, the
# same for every implementation.
_LOG_FACTORIAL = np.array([math.lgamma(k + 1.0) for k in range(256)])
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
# log(m) = 2 atanh(s), s = (m - 1) / (m + 1): the series' coefficients 1 / (2k + 1).
# For m in [sqrt(1/2), sqrt(2)), |s| <= 0.172 and 12 terms reach double precision.
_LOG_SERIES = np.array([1.0 / (2 * k + 1) for k in range(12)])
_SQRT_HALF = math.sqrt(0.5)
_LN2 = math.log(2.0)
# A member sorts up to this many positives by insertion, more by quicksort.
_INSERTION = 32
# Samples per vectorized call.
_CHUNK = 2**16


def _make_sampler(philox, uniform, jit=lambda f: f):
    """Build the scalar joint sampler around ``philox`` and ``uniform``.

    ``jit`` compiles each part (e.g. ``numba.njit``) or leaves it Python.
    Returns ``(joint_sample, binomial, log)``.
    """
    # Two uniforms of block ``block`` of stream ``stream`` of sample ``(c0, c1)``.
    pair = make_uniform_pair(philox, uniform, jit)

    @jit
    def log(x):
        """Natural log of a float ``x >= 0`` (within a few ulp of the exact value)."""
        if x == 0.0:
            return -math.inf
        m, e = math.frexp(x)  # x = m * 2**e, 0.5 <= m < 1
        if m < _SQRT_HALF:
            m *= 2.0
            e -= 1
        s = (m - 1.0) / (m + 1.0)
        z = s * s
        p = _LOG_SERIES[11]
        for k in range(10, -1, -1):
            p = p * z + _LOG_SERIES[k]
        return e * _LN2 + 2.0 * s * p

    @jit
    def powi(x, n):
        """``x ** n`` of an integer ``n >= 0`` by repeated squaring."""
        r = 1.0
        while n > 0:
            if n & 1:
                r *= x
            x *= x
            n >>= 1
        return r

    @jit
    def log_factorial(k):
        """log(k!) of an integer-valued ``k >= 0`` (relative error < 1e-15)."""
        if k < 256:
            return _LOG_FACTORIAL[int(k)]
        x = k + 1.0
        r = 1.0 / (x * x)
        return (
            (x - 0.5) * log(x)
            - x
            + _HALF_LOG_2PI
            + (1.0 / 12.0 - r * (1.0 / 360.0 - r / 1260.0)) / x
        )

    @jit
    def binomial(n, p, c0, c1, block, stream, k0, k1):
        """Binomial(``n``, ``p``) from the stream's blocks starting at ``block``.

        Returns ``(draw, next_block)``. Inversion for small means (BINV), else
        Hörmann's (1993) transformed rejection with squeeze (BTRS); both exact
        up to floating-point rounding.
        """
        if n <= 0 or p <= 0.0:
            return 0, block
        if p >= 1.0:
            return n, block
        flip = p > 0.5
        q = 1.0 - p if flip else p
        if n * q < _INVERSION_MEAN:
            u, _ = pair(c0, c1, block, stream, k0, k1)
            block += 1
            s = q / (1.0 - q)
            a = (n + 1) * s
            r = powi(1.0 - q, n)
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
            lpq = log(q / (1.0 - q))
            m = math.floor((n + 1) * q)
            h = log_factorial(m) + log_factorial(n - m)
            while True:
                u, v = pair(c0, c1, block, stream, k0, k1)
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
                v = log(v * alpha / (a / (us * us) + b))
                if v <= h - log_factorial(kf) - log_factorial(n - kf) + (kf - m) * lpq:
                    break
            x = int(kf)
        return (n - x if flip else x), block

    sort_head = make_head_sort(_INSERTION, jit)

    @jit
    def member_ap(vals, num_pos, num_neg, c0, c1, stream, k0, k1):
        """AP of a member whose positive uniforms are ``vals[:num_pos]`` (sorted here)."""
        sort_head(vals, num_pos)
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

    @jit
    def joint_sample(
        s, j, mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1, ev, vals
    ):
        """Return the mAP of sample ``j`` of signature ``s`` (``ev``, ``vals``: scratch)."""
        c0 = np.uint64(j) & M32
        c1 = np.uint64(j) >> S32
        key0, key1 = k0[s], k1[s]
        n_edge = edge_ptr[s + 1] - edge_ptr[s]
        for e in range(0, n_edge, 2):
            u0, u1 = pair(c0, c1, e // 2, 0, key0, key1)
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
                u0, u1 = pair(c0, c1, (t - shared) // 2, stream, key0, key1)
                vals[t] = u0
                if t + 1 < p:
                    vals[t + 1] = u1
            total += member_ap(vals, p, num_neg[i], c0, c1, stream + 1, key0, key1)
        return total / (m1 - m0)

    return joint_sample, binomial, log


# The scalar sampler in Python: the definition of every sample.
joint_sample, binomial, _log = _make_sampler(philox4x32, uniform53)


def _log_np(x):
    """:func:`_log` elementwise, with the same operations in the same order."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    small = m < _SQRT_HALF
    m = np.where(small, m * 2.0, m)
    e = np.where(small, e - 1, e)
    s = (m - 1.0) / (m + 1.0)
    z = s * s
    p = np.full(x.shape, _LOG_SERIES[11])
    for k in range(10, -1, -1):
        p = p * z + _LOG_SERIES[k]
    out = e * _LN2 + 2.0 * s * p
    out[x == 0.0] = -np.inf
    return out


def _powi_np(x, n):
    """``x ** n`` elementwise by repeated squaring, as in :func:`_make_sampler`."""
    r = np.ones(x.shape)
    n = n.copy()
    with np.errstate(under="ignore"):
        while (n > 0).any():
            odd = (n & 1) == 1
            r = np.where(odd, r * x, r)
            x = x * x
            n >>= 1
    return r


def _log_factorial_np(k):
    """log(k!) of integer-valued floats ``k >= 0``, as in :func:`_make_sampler`."""
    x = np.maximum(k, 256.0) + 1.0
    r = 1.0 / (x * x)
    big = (
        (x - 0.5) * _log_np(x)
        - x
        + _HALF_LOG_2PI
        + (1.0 / 12.0 - r * (1.0 / 360.0 - r / 1260.0)) / x
    )
    small = _LOG_FACTORIAL[np.minimum(k, 255).astype(np.int64)]
    return np.where(k < 256, small, big)


def _binomial_np(n, p, c0, c1, block, stream, k0, k1):
    """:func:`binomial` elementwise: ``(draws, next_blocks)`` of int64 arrays."""
    n = np.asarray(n, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    block = np.array(block, dtype=np.int64)
    x = np.zeros(n.shape, dtype=np.int64)
    flip = p > 0.5
    q = np.where(flip, 1.0 - p, p)
    live = (n > 0) & (p > 0.0) & (p < 1.0)
    inv = np.flatnonzero(live & (n * q < _INVERSION_MEAN))
    if inv.size:
        ni, qi = n[inv], q[inv]
        u, _ = uniform_pair(c0[inv], c1[inv], block[inv], stream, k0, k1)
        block[inv] += 1
        s = qi / (1.0 - qi)
        a = (ni + 1) * s
        r = _powi_np(1.0 - qi, ni)
        xi = np.zeros(inv.size, dtype=np.int64)
        run = np.flatnonzero((u > r) & (xi < ni))
        while run.size:
            u[run] -= r[run]
            xi[run] += 1
            r[run] *= a[run] / xi[run] - s[run]
            run = run[(u[run] > r[run]) & (xi[run] < ni[run])]
        x[inv] = xi
    rej = np.flatnonzero(live & ~(n * q < _INVERSION_MEAN))
    if rej.size:
        x[rej] = _btrs_np(n[rej], q[rej], c0[rej], c1[rej], block, rej, stream, k0, k1)
    x = np.where(flip & live, n - x, x)
    x[(n > 0) & (p >= 1.0)] = n[(n > 0) & (p >= 1.0)]
    return x, block


def _btrs_np(n, q, c0, c1, block, ix, stream, k0, k1):
    """Draw the BTRS branch of :func:`binomial`, advancing ``block[ix]`` in place."""
    spq = np.sqrt(n * q * (1.0 - q))
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * q
    c = n * q + 0.5
    v_r = 0.92 - 4.2 / b
    alpha = (2.83 + 5.1 / b) * spq
    lpq = _log_np(q / (1.0 - q))
    m = np.floor((n + 1) * q)
    h = _log_factorial_np(m) + _log_factorial_np(n - m)
    out = np.zeros(n.shape, dtype=np.int64)
    run = np.arange(n.size)
    while run.size:
        u, v = uniform_pair(c0[run], c1[run], block[ix[run]], stream, k0, k1)
        block[ix[run]] += 1
        u = u - 0.5
        us = 0.5 - np.abs(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            kf = np.floor((2.0 * a[run] / us + b[run]) * u + c[run])
        ok = (us > 0.0) & (kf >= 0.0) & (kf <= n[run])
        done = ok & (us >= 0.07) & (v <= v_r[run])
        test = np.flatnonzero(ok & ~done)
        if test.size:
            r, kt = run[test], kf[test]
            ut = us[test]
            lhs = _log_np(v[test] * alpha[r] / (a[r] / (ut * ut) + b[r]))
            rhs = (
                h[r]
                - _log_factorial_np(kt)
                - _log_factorial_np(n[r] - kt)
                + (kt - m[r]) * lpq[r]
            )
            done[test] = lhs <= rhs
        out[run[done]] = kf[done].astype(np.int64)
        run = run[~done]
    return out


def _member_ap_np(vals, num_neg, c0, c1, stream, k0, k1):
    """:func:`member_ap` over samples: ``vals`` holds each sample's positives, sorted."""
    size, num_pos = vals.shape
    remaining = np.full(size, num_neg, dtype=np.int64)
    prev = np.ones(size)
    above = np.zeros(size, dtype=np.int64)
    acc = np.zeros(size)
    block = np.zeros(size, dtype=np.int64)
    for t in range(num_pos):
        v = vals[:, num_pos - 1 - t]  # descending
        ix = np.flatnonzero((remaining > 0) & (prev > 0.0))
        if ix.size:
            x, block[ix] = _binomial_np(
                remaining[ix],
                (prev[ix] - v[ix]) / prev[ix],
                c0[ix],
                c1[ix],
                block[ix],
                stream,
                k0,
                k1,
            )
            above[ix] += x
            remaining[ix] -= x
        prev = v
        acc += (t + 1) / (t + 1 + above)
    return acc / num_pos


def _joint_null_np(plan, s, start, size):
    """Return samples ``start, ..., start + size - 1`` of signature ``s``, vectorized."""
    mem_ptr, num_pos, num_neg, edge_ptr, inc_ptr, inc_edge, k0, k1 = plan
    j = np.arange(start, start + size, dtype=np.uint64)
    c0, c1 = j & M32, j >> S32
    key0, key1 = k0[s], k1[s]
    n_edge = edge_ptr[s + 1] - edge_ptr[s]
    ev = np.empty((size, n_edge))
    for e in range(0, n_edge, 2):
        u0, u1 = uniform_pair(c0, c1, e // 2, 0, key0, key1)
        ev[:, e] = u0
        if e + 1 < n_edge:
            ev[:, e + 1] = u1
    m0, m1 = mem_ptr[s], mem_ptr[s + 1]
    total = np.zeros(size)
    for i in range(m0, m1):
        p = num_pos[i]
        shared = inc_ptr[i + 1] - inc_ptr[i]
        vals = np.empty((size, p))
        vals[:, :shared] = ev[:, inc_edge[inc_ptr[i] : inc_ptr[i + 1]]]
        stream = 1 + 2 * (i - m0)
        for t in range(shared, p, 2):
            u0, u1 = uniform_pair(c0, c1, (t - shared) // 2, stream, key0, key1)
            vals[:, t] = u0
            if t + 1 < p:
                vals[:, t + 1] = u1
        vals.sort(axis=1)
        total += _member_ap_np(vals, num_neg[i], c0, c1, stream + 1, key0, key1)
    return total / (m1 - m0)


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
        k0[s], k1[s] = stream_key(seed, sig)
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
    for r, s in enumerate(range(lo, hi)):
        for c, size in chunks(null_size, _CHUNK):
            out[r, c : c + size] = _joint_null_np(plan, s, start + c, size)
    return out


def joint_pvalues(
    scores: np.ndarray,
    sig_ix: np.ndarray,
    signatures,
    null_size: int,
    seed: int | None,
    progress_bar: bool = False,
) -> np.ndarray:
    """P-value of each group's mAP against the joint null of its signature.

    Group ``g`` has signature ``signatures[sig_ix[g]]``, a canonical
    ``(members, edges)`` pair (:func:`canonical_signature`) of its members'
    ``(num_pos, num_neg)`` and its positive pairs between local member
    indices. ``p_g = (1 + #{null >= thr_g}) / (null_size + 1)`` with ``thr_g``
    from :func:`~copairs.nulls.tie_thresholds`. Nulls are generated in chunks
    of samples and not stored.
    """
    thr = tie_thresholds(scores)
    sig_ix = np.asarray(sig_ix, dtype=np.int64)
    if len(sig_ix) != len(thr):
        raise ValueError(f"{len(thr)} scores but {len(sig_ix)} sig_ix")
    if len(sig_ix) and (sig_ix.min() < 0 or sig_ix.max() >= len(signatures)):
        raise ValueError(f"sig_ix must index the {len(signatures)} signatures")
    null_size = check_null_size(null_size)
    plan = joint_plan(signatures, resolve_seed(seed))
    order = np.argsort(sig_ix, kind="stable")
    ptr = np.searchsorted(sig_ix[order], np.arange(len(signatures) + 1))
    counts = np.zeros(len(thr), dtype=np.int64)
    sigs = range(len(signatures))
    for s in progress(sigs, len(sigs), progress_bar, "joint mAP null"):
        groups = order[ptr[s] : ptr[s + 1]]
        if not len(groups):
            continue
        for c, size in chunks(null_size, _CHUNK):
            null = np.sort(joint_nulls(plan, size, c, slice(s, s + 1))[0])
            counts[groups] += size - np.searchsorted(null, thr[groups], side="left")
    return (counts + 1) / (null_size + 1)


def group_signatures(groups, num_pos, num_neg, profile=None, pos_pairs=None):
    """Canonical signature of each group, or None where it cannot be inferred.

    ``groups`` lists each group's rows of the AP table and ``num_pos``,
    ``num_neg`` its rows' pair counts. With ``pos_pairs`` (indexing the
    profiles ``profile`` of the AP rows), a group's edges are its positive
    pairs inside it (:func:`copairs.nulls.common.group_edges`). Without them,
    a group whose members each have ``len(group) - 1`` positives is taken to
    be a replicate group (all its pairs are edges), a group of one profile
    has no edges, and other groups get None.
    """
    if pos_pairs is None:
        edges_of = [
            [(i, j) for i in range(len(ix)) for j in range(i + 1, len(ix))]
            if len(ix) == 1 or (num_pos[ix] == len(ix) - 1).all()
            else None
            for ix in groups
        ]
    else:
        edges_of, _ = group_edges(groups, profile, num_pos, pos_pairs)
    return [
        None
        if edges is None
        else canonical_signature(zip(num_pos[ix], num_neg[ix]), edges)
        for ix, edges in zip(groups, edges_of)
    ]


def group_pvalues(
    scores: np.ndarray,
    groups,
    num_pos: np.ndarray,
    num_neg: np.ndarray,
    null_size: int,
    seed: int | None,
    profile: np.ndarray | None = None,
    pos_pairs: np.ndarray | None = None,
    progress_bar: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """P-value of each group's mAP ``scores`` under the joint null.

    Groups share the null of their signature (:func:`group_signatures`).
    Returns ``(p_values, fallback)``: ``fallback`` lists the groups without a
    signature, whose p-values are NaN.
    """
    signatures = group_signatures(groups, num_pos, num_neg, profile, pos_pairs)
    unique = sorted(set(sig for sig in signatures if sig is not None))
    index = {sig: k for k, sig in enumerate(unique)}
    sampled = [g for g, sig in enumerate(signatures) if sig is not None]
    p_values = np.full(len(groups), np.nan)
    p_values[sampled] = joint_pvalues(
        scores[sampled],
        [index[signatures[g]] for g in sampled],
        unique,
        null_size,
        seed,
        progress_bar=progress_bar,
    )
    return p_values, [g for g, sig in enumerate(signatures) if sig is None]
