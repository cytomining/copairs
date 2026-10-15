"""Exact Monte Carlo samples of the average-precision null distribution.

Under the null, the ``num_pos`` positives of a rank list of length ``total``
occupy a uniformly random subset of the ranks. Instead of shuffling a
``null_size x total`` matrix, each sample places the positives one at a time:
the number of negatives before the next positive is drawn by inverting its
survival function (Vitter 1984, "Faster methods for random sampling",
Algorithm A), which needs one uniform per positive and no memory beyond a few
scalars.

Sample ``j`` of configuration ``(num_pos, total)`` is a pure function of
``(seed, num_pos, total, j)`` through a Philox counter, and every backend
performs the same IEEE double operations in the same order, so the NumPy,
Numba and CUDA implementations return bitwise-identical nulls.
"""

import math

import numba
import numpy as np

from copairs.methods import BACKENDS
from copairs.nulls.philox import M32, S32, uniform53, philox4x32, config_key_arrays

# Use the guided gap search when the expected gap exceeds this many times the
# number of remaining positives (its cost per probe); results are identical.
GUIDED_RATIO = 32
# Algorithm A's own rounding bound, (2g + 2) eps, widens the undecidable band as
# gaps grow; from totals of ~1e8 the last probes all fall in it and replay
# O(g) products, so the plain loop is faster there. Checked first, the cap also
# bounds k: GUIDED_RATIO * k * (k + 1) stays far inside int64, and the guided
# condition implies k < 1449, so the k-term log1p sum errs by < 1e-11.
GUIDED_MAX_TOTAL = 2**26
# Bound on |log P(G > g) evaluated with log1p - log of Algorithm A's product|,
# excluding the product's own rounding, when every factor is <= 1 - 2**-10.
_LOG_MARGIN = 1e-9
_EPS = 2.0**-52
_FACTOR_LIMIT = 1.0 - 2.0**-10


def gap_loop(remaining, k, u):
    """Algorithm A: smallest gap g whose product P(G > g) is <= u (defines the stream)."""
    top = remaining - k
    quot = top / remaining
    gap = 0
    while quot > u:
        gap += 1
        top -= 1
        quot = quot * (top / (remaining - gap))
    return gap


def _make_gap_guided(gap_loop, jit=lambda f: f):
    """Build the guided search; ``jit`` compiles its helper (Numba) or not (Python)."""

    @jit
    def stops(remaining, k, g, u, log_u):
        """Whether Algorithm A's product for gap ``g`` is <= ``u``."""
        # The factor with t = k - 1 is the largest; near 1 the log is ill-conditioned.
        if (g + 1) / (remaining - k + 1) <= _FACTOR_LIMIT:
            acc = 0.0
            for t in range(k):
                acc += math.log1p(-((g + 1) / (remaining - t)))
            margin = _LOG_MARGIN + (2 * g + 2) * _EPS
            if acc < log_u - margin:
                return True
            if acc > log_u + margin:
                return False
        # Too close to call: replay the product exactly.
        top = remaining - k
        quot = top / remaining
        for gap in range(1, g + 1):
            top -= 1
            quot = quot * (top / (remaining - gap))
        return not quot > u

    def gap_guided(remaining, k, u):
        """:func:`gap_loop` by exponential and binary search over the closed form."""
        last = remaining - k  # the product is exactly 0 there
        if u == 0.0:
            return gap_loop(remaining, k, u)
        log_u = math.log(u)
        guess = (remaining - 0.5 * (k - 1)) * (1.0 - math.exp(log_u / k)) - 1.0
        g = min(max(int(guess), 0), last)
        if stops(remaining, k, g, u, log_u):
            hi, step = g, 1
            lo = hi - step
            while lo >= 0 and stops(remaining, k, lo, u, log_u):
                hi = lo
                step *= 2
                lo = hi - step
            lo = max(lo, -1)
        else:
            lo, step = g, 1
            hi = lo + step
            while hi < last and not stops(remaining, k, hi, u, log_u):
                lo = hi
                step *= 2
                hi = lo + step
            hi = min(hi, last)
        while hi - lo > 1:  # stops(lo) is false (or lo == -1), stops(hi) is true
            mid = (lo + hi) // 2
            if stops(remaining, k, mid, u, log_u):
                hi = mid
            else:
                lo = mid
        return hi

    return gap_guided


def _make_ap_sample(philox, uniform, gap_loop, gap_guided):
    """Build the scalar sampler around ``philox``, ``uniform`` and the gap searches."""

    def ap_sample(num_pos, total, j, k0, k1):
        """Average precision of sample ``j`` of configuration ``(num_pos, total)``."""
        c0 = np.uint64(j) & M32
        c1 = np.uint64(j) >> S32
        remaining = total
        k = num_pos
        rank = 0
        acc = 0.0
        i = 0
        u_next = 0.0
        while k > 0:
            if k == remaining:
                # Every remaining rank holds a positive.
                while k > 0:
                    rank += 1
                    i += 1
                    acc += i / rank
                    k -= 1
                break
            if i % 2 == 0:
                w0, w1, w2, w3 = philox(c0, c1, np.uint64(i // 2), np.uint64(0), k0, k1)
                u = uniform(w0, w1)
                u_next = uniform(w2, w3)
            else:
                u = u_next
            if k == 1:
                gap = int(math.floor(remaining * u))
                if gap > remaining - 1:
                    gap = remaining - 1
            elif remaining < GUIDED_MAX_TOTAL and remaining - k > GUIDED_RATIO * k * (
                k + 1
            ):
                gap = gap_guided(remaining, k, u)
            else:
                gap = gap_loop(remaining, k, u)
            rank += gap + 1
            i += 1
            acc += i / rank
            remaining -= gap + 1
            k -= 1
        return acc / num_pos

    return ap_sample


_gap_guided = _make_gap_guided(gap_loop)
_ap_sample = _make_ap_sample(philox4x32, uniform53, gap_loop, _gap_guided)


def _ap_null_numpy(num_pos, total, start, size, k0, k1):
    """Vectorized NumPy version of :func:`_ap_sample` over ``size`` samples (float64)."""
    j = np.arange(start, start + size, dtype=np.uint64)
    c0, c1 = j & M32, j >> S32
    k0, k1 = np.uint64(k0), np.uint64(k1)
    remaining = np.full(size, total, dtype=np.int64)
    rank = np.zeros(size, dtype=np.int64)
    acc = np.zeros(size, dtype=np.float64)
    live = np.arange(size)
    for i in range(num_pos):
        k = num_pos - i
        full = remaining[live] == k
        if full.any():
            ix = live[full]
            for step in range(k):
                rank[ix] += 1
                acc[ix] += (i + step + 1) / rank[ix]
            live = live[~full]
        if live.size == 0:
            break
        w = philox4x32(c0[live], c1[live], np.uint64(i // 2), np.uint64(0), k0, k1)
        u = uniform53(w[0], w[1]) if i % 2 == 0 else uniform53(w[2], w[3])
        rem = remaining[live]
        if k == 1:
            gap = np.minimum(np.floor(rem * u).astype(np.int64), rem - 1)
        else:
            top = rem - k
            quot = top / rem
            gap = np.zeros(live.size, dtype=np.int64)
            run = np.flatnonzero(quot > u)
            while run.size:
                gap[run] += 1
                top[run] -= 1
                quot[run] = quot[run] * (top[run] / (rem[run] - gap[run]))
                run = run[quot[run] > u[run]]
        rank[live] += gap + 1
        acc[live] += (i + 1) / rank[live]
        remaining[live] = rem - gap - 1
    return acc / num_pos


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
def _ap_nulls_numba(num_pos, total, k0, k1, start, out):
    n_conf, size = out.shape
    for flat in numba.prange(n_conf * size):
        c = flat // size
        t = flat - c * size
        out[c, t] = _ap_sample_nb(num_pos[c], total[c], start + t, k0[c], k1[c])


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first."""
    backends = ["numba", "numpy"]
    from copairs.nulls import cuda

    if cuda.is_available():
        backends.insert(0, "cuda")
    return backends


def resolve_backend(backend: str) -> str:
    """Return a concrete backend name for ``backend`` (``"auto"`` picks the fastest)."""
    available = available_backends()
    if backend == "auto":
        return available[0]
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend not in available:
        raise ValueError(f"backend {backend!r} is not available here ({available})")
    return backend


def _validate_confs(confs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confs = np.asarray(confs)
    if confs.ndim != 2 or confs.shape[1] != 2:
        raise ValueError(f"confs must have shape (n, 2), got {confs.shape}")
    num_pos = confs[:, 0].astype(np.int64)
    total = confs[:, 1].astype(np.int64)
    if (num_pos != confs[:, 0]).any() or (total != confs[:, 1]).any():
        raise ValueError("configurations must be integer counts")
    if (num_pos < 1).any() or (total < num_pos).any():
        raise ValueError("each configuration needs 1 <= num_pos <= total")
    return num_pos, total


def null_plan(confs: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    """Return validated ``(num_pos, total, k0, k1)`` arrays of ``confs`` for ``seed``.

    Callers sampling the same configurations chunk by chunk derive this once.
    """
    num_pos, total = _validate_confs(confs)
    k0, k1 = config_key_arrays(seed, num_pos, total)
    return num_pos, total, k0, k1


def ap_nulls(
    confs: np.ndarray,
    null_size: int,
    seed: int,
    start: int = 0,
    backend: str = "auto",
    dtype=np.float32,
):
    """Sample the AP null of every ``(num_pos, total)`` row of ``confs``.

    Parameters
    ----------
    confs : np.ndarray
        ``(n, 2)`` integer array of ``(num_pos, total)`` configurations.
    null_size : int
        Number of samples per configuration.
    seed : int
        Seed in ``[0, 2**64)``. Sample ``j`` of a configuration depends only on
        ``(seed, num_pos, total, j)``.
    start : int
        Index of the first sample, so a null can be generated in chunks.
    backend : str
        ``"auto"``, ``"cuda"``, ``"numba"`` or ``"numpy"``. All return identical
        values.
    dtype : np.float32 or np.float64
        Output precision. Samples are computed in float64 and rounded once.

    Returns
    -------
    np.ndarray
        ``(n, null_size)`` AP samples.
    """
    plan = null_plan(confs, seed)
    backend = resolve_backend(backend)
    out = _ap_nulls(plan, null_size, start, backend, dtype)
    return out.get() if backend == "cuda" else out


def _ap_nulls(plan, null_size, start, backend, dtype):
    """:func:`ap_nulls` from a :func:`null_plan`; CUDA results stay on the device."""
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    num_pos, total, k0, k1 = plan
    if backend == "cuda":
        from copairs.nulls import cuda

        return cuda.ap_nulls(num_pos, total, k0, k1, start, null_size, dtype)
    out = np.empty((len(num_pos), null_size), dtype=dtype)
    if backend == "numba":
        _ap_nulls_numba(num_pos, total, k0, k1, start, out)
    else:
        for c in range(len(num_pos)):
            out[c] = _ap_null_numpy(
                int(num_pos[c]), int(total[c]), start, null_size, k0[c], k1[c]
            )
    return out
