"""Exact Monte Carlo samples of the average-precision null distribution.

Under the null, the ``num_pos`` positives of a rank list of length ``total``
occupy a uniformly random subset of the ranks. Instead of shuffling a
``null_size x total`` matrix, each sample places the positives one at a time:
the number of negatives before the next positive is drawn by inverting its
survival function (Vitter 1984, "Faster methods for random sampling",
Algorithm A), which needs one uniform per positive and no memory beyond a few
scalars.

Sample ``j`` of configuration ``(num_pos, total)`` is a pure function of
``(seed, num_pos, total, j)`` through a Philox counter, and every gap is
Algorithm A's, the smallest whose float64 product is at most ``u``. Searches
that shortcut the product through ``log1p`` decide only outside an error
margin and replay the product inside it, so the gap does not depend on the
library that computes ``log1p``, ``log`` and ``exp`` (NumPy, glibc or CUDA,
each within a few ulp; the margin covers far more). The scalar functions here
are the reference; :mod:`copairs.nulls._numba` compiles the same source,
bitwise identical to the NumPy backend (tested), and :mod:`copairs.nulls.cuda`
ports it line by line, stepping long gaps in float32.
"""

import os
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from copairs.nulls.philox import M32, S32, uniform53, philox4x32, config_key_arrays
from copairs.nulls.backend import (
    resolve_backend,
    _import_cuda_backend,
    _import_numba_backend,
)

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
# excluding the product's own rounding, when every factor is <= 1 - 2**-10. It
# leaves room for log1p and log errors of thousands of ulp, so it holds for any
# libm (CUDA documents 1 ulp for log1p, log and exp in double precision).
_LOG_MARGIN = 1e-9
_EPS = 2.0**-52
_FACTOR_LIMIT = 1.0 - 2.0**-10
# The NumPy backend runs its gap loop to the largest gap among the samples, about
# ln(samples) times the mean gap (remaining - k) / (k + 1), and a bisection in
# about k * log2(remaining) array passes, so it bisects from a much smaller
# ratio. Its own k bound keeps the log1p sum error bound above.
NUMPY_GUIDED_RATIO = 2
_GUIDED_MAX_K = 1448
# Default bound on the NumPy backend's temporaries, in bytes.
DEFAULT_BUDGET = 2**30
# Upper bound on the NumPy backend's temporaries per sample, in bytes: its
# tracemalloc peak is at most 217 bytes per sample (gap bisection; 128 for one
# positive, 178 for the plain gap loop), output included.
_NUMPY_SAMPLE_BYTES = 256
# Smallest block of samples worth a NumPy worker thread of its own: with 4096,
# 4 configurations x 1e4 samples took 0.34 s on 16 threads, with 16384 0.29 s.
_MIN_BLOCK = 16384


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


def _gap_loop_numpy(remaining, k, u):
    """:func:`gap_loop` over arrays of ``remaining`` and ``u``."""
    top = remaining - k
    quot = top / remaining
    gap = np.zeros(len(u), dtype=np.int64)
    run = np.flatnonzero(quot > u)
    # Every sample still running has the same gap, so carry only theirs.
    top, quot, rem, u = top[run], quot[run], remaining[run], u[run]
    g = 0
    while run.size:
        g += 1
        top -= 1
        quot = quot * (top / (rem - g))
        more = quot > u
        if not more.all():
            gap[run[~more]] = g
            run, top, quot, rem, u = (a[more] for a in (run, top, quot, rem, u))
    return gap


def _product_numpy(remaining, k, g):
    """Algorithm A's product after gap ``g``, as :func:`gap_loop` computes it."""
    top = remaining - k
    quot = top / remaining
    run = np.flatnonzero(g >= 1)
    gap = 1
    while run.size:
        top[run] -= 1
        quot[run] = quot[run] * (top[run] / (remaining[run] - gap))
        gap += 1
        run = run[g[run] >= gap]
    return quot


def _stops_numpy(remaining, k, g, u, log_u):
    """Vectorized ``stops`` of :func:`_make_gap_guided` (product at ``g`` <= ``u``)."""
    # Exact in float64 (counts are < 2**53), so each term rounds as before.
    g1 = (g + 1).astype(np.float64)
    rem = remaining.astype(np.float64)
    sure = g1 / (rem - (k - 1)) <= _FACTOR_LIMIT
    acc = np.zeros(len(g))
    term = np.empty(len(g))
    with np.errstate(divide="ignore"):  # log1p(-1) at g = remaining - k, replayed
        for t in range(k):
            np.subtract(rem, t, out=term)
            np.divide(g1, term, out=term)
            np.negative(term, out=term)
            acc += np.log1p(term, out=term)
    margin = _LOG_MARGIN + (2 * g + 2) * _EPS
    below = acc < log_u - margin
    stop = sure & below
    # Too close to call: replay the product exactly.
    replay = np.flatnonzero(~sure | ~(below | (acc > log_u + margin)))
    if replay.size:
        product = _product_numpy(remaining[replay], k, g[replay])
        stop[replay] = ~(product > u[replay])
    return stop


def _gap_bisect_numpy(remaining, k, u):
    """:func:`gap_loop` by bisection over the closed form, for ``u > 0``.

    The product falls monotonically to exactly 0 at ``remaining - k``, so the
    smallest gap where it is ``<= u`` is unique, and each probe decides it
    exactly (see :func:`_make_gap_guided`).
    """
    gap = remaining - k  # output; the product is exactly 0 there
    lo = np.full(len(u), -1, dtype=np.int64)  # the product is > u at lo
    hi, log_u = gap.copy(), np.log(u)  # and <= u at hi
    run = np.arange(len(u))
    # Bisect only the open intervals, carrying their arrays along.
    while True:
        open_ = hi - lo > 1
        if not open_.all():
            gap[run[~open_]] = hi[~open_]
            run, lo, hi, remaining, u, log_u = (
                a[open_] for a in (run, lo, hi, remaining, u, log_u)
            )
        if not run.size:
            return gap
        mid = (lo + hi) // 2
        stop = _stops_numpy(remaining, k, mid, u, log_u)
        hi = np.where(stop, mid, hi)
        lo = np.where(stop, lo, mid)


def _ap_null_numpy(num_pos, total, start, size, k0, k1):
    """Vectorized NumPy version of :func:`_ap_sample` over ``size`` samples (float64).

    Gaps come from the plain loop or a bisection, by a cost rule of its own;
    both give Algorithm A's gap exactly, so samples equal the scalar ones.
    """
    j = np.arange(start, start + size, dtype=np.uint64)
    c0, c1 = j & M32, j >> S32
    k0, k1 = np.uint64(k0), np.uint64(k1)
    remaining = np.full(size, total, dtype=np.int64)
    rank = np.zeros(size, dtype=np.int64)
    acc = np.zeros(size, dtype=np.float64)
    for i in range(num_pos):
        k = num_pos - i
        # Positive i takes words 0-1 of Philox block i // 2, and i + 1 words 2-3.
        if i % 2 == 0:
            w = philox4x32(c0, c1, np.uint64(i // 2), np.uint64(0), k0, k1)
            u = uniform53(w[0], w[1])
        else:
            u = uniform53(w[2], w[3])
        # A sample whose remaining ranks all hold positives gets gaps of 0 here,
        # without the scalar sampler's early exit.
        if k == 1:
            gap = np.minimum(np.floor(remaining * u).astype(np.int64), remaining - 1)
        else:
            gap = np.empty(size, dtype=np.int64)
            # Checked first, the k bound also keeps k * (k + 1) inside int64.
            wide = (
                remaining - k > NUMPY_GUIDED_RATIO * k * (k + 1)
                if k <= _GUIDED_MAX_K
                else False
            )
            guided = (u > 0) & (remaining < GUIDED_MAX_TOTAL) & wide
            gap[guided] = _gap_bisect_numpy(remaining[guided], k, u[guided])
            gap[~guided] = _gap_loop_numpy(remaining[~guided], k, u[~guided])
        rank += gap + 1
        acc += (i + 1) / rank
        remaining -= gap + 1
    return acc / num_pos


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
        ``"auto"``, ``"cuda"``, ``"numba"`` or ``"numpy"``. ``"auto"`` picks
        Numba when installed, else NumPy, which return identical values.
        ``"cuda"`` runs only when asked for: it also steps long gaps in
        float32, which is faster on GPUs with few float64 units, and error
        bounds rather than identical arithmetic make its samples NumPy's
        (see :mod:`copairs.nulls.cuda`). It holds at most 1 GiB of samples
        on the GPU at a time.
    dtype : np.float32 or np.float64
        Output precision. Samples are computed in float64 and rounded once.

    Returns
    -------
    np.ndarray
        ``(n, null_size)`` AP samples.
    """
    plan = null_plan(confs, seed)
    backend = resolve_backend(backend)
    if backend == "cuda":
        # Sampled within DEFAULT_BUDGET of GPU memory at a time.
        return _import_cuda_backend().ap_nulls_to_host(
            plan, start, null_size, _check_dtype(dtype), DEFAULT_BUDGET
        )
    return _ap_nulls(plan, null_size, start, backend, dtype)


def _ap_nulls(
    plan,
    null_size,
    start,
    backend,
    dtype,
    budget_bytes=DEFAULT_BUDGET,
    max_workers=None,
):
    """:func:`ap_nulls` from a :func:`null_plan` and a resolved backend.

    The NumPy backend samples blocks of configurations' samples on a thread
    pool of at most ``max_workers`` threads (default :func:`_numpy_workers`),
    sized so the blocks in flight keep their temporaries within
    ``budget_bytes``; the output is allocated on top of that. The Numba
    backend runs on Numba's threads. Samples do not depend on the threads.
    The CUDA backend returns a CuPy array.
    """
    dtype = _check_dtype(dtype)
    if backend == "cuda":
        return _import_cuda_backend().ap_nulls(plan, start, null_size, dtype)
    num_pos, total, k0, k1 = plan
    out = np.empty((len(num_pos), null_size), dtype=dtype)
    if backend == "numba":
        _import_numba_backend().ap_nulls(num_pos, total, k0, k1, start, out)
        return out
    # Each worker holds one block, so their temporaries together fit the budget.
    slots = max(1, budget_bytes // _NUMPY_SAMPLE_BYTES)
    if max_workers is None:
        max_workers = _numpy_workers()
    workers = max(1, min(int(max_workers), slots // _MIN_BLOCK))
    # Split configurations into blocks when there are fewer than workers.
    per_conf = -(-workers // max(len(num_pos), 1))
    block = min(slots // workers, max(_MIN_BLOCK, -(-null_size // per_conf)))
    tasks = [(c, s) for c in range(len(num_pos)) for s in range(0, null_size, block)]

    def sample(task):
        c, s = task
        n = min(block, null_size - s)
        out[c, s : s + n] = _ap_null_numpy(
            int(num_pos[c]), int(total[c]), start + s, n, k0[c], k1[c]
        )

    if workers == 1 or len(tasks) < 2:
        for task in tasks:
            sample(task)
    else:
        with ThreadPoolExecutor(min(workers, len(tasks))) as ex:
            list(ex.map(sample, tasks))
    return out


def _check_dtype(dtype) -> np.dtype:
    """``dtype`` as a NumPy dtype; ValueError unless float32 or float64."""
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    return dtype


def _numpy_workers() -> int:
    """Threads of the NumPy backend: the CPUs this process may run on.

    That is the process's CPU affinity where the OS reports it (it follows
    ``taskset`` and Slurm allocations), else ``os.cpu_count()``.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1
