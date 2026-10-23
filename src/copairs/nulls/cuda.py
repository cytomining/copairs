"""CUDA kernels of the AP null sampler and p-value counts (optional, CuPy).

Imported only through :func:`copairs.nulls.backend._import_cuda_backend`
(``pip install 'copairs[gpu]'``).

The sampler kernel is a line-by-line port of
:func:`copairs.nulls.sampler._ap_sample`: the same Philox blocks and the same
double-precision operations in the same order, compiled without FMA
contraction, with two differences that error bounds, not identical
arithmetic, keep from changing the samples:

- The guided gap search calls CUDA's ``log1p``, ``log`` and ``exp``, not
  glibc's. As on the CPU, it decides a gap only outside the log margin of
  :mod:`copairs.nulls.sampler`, which covers either library, and replays the
  float64 product inside it.
- The plain gap loop steps in float32, which GPUs with few float64 units run
  up to 64 times faster. A step is taken in float32 only when the float32
  product clears ``u`` by its band, and otherwise the float64 product is
  replayed. The band covers the float32 product's worst-case rounding
  relative to Algorithm A's float64 product for every gap below
  :data:`F32_MAX_GAP` (checked exhaustively by tests/test_cuda.py), and longer
  gaps replay in float64.

So the gaps are Algorithm A's and the samples NumPy's. A CPU emulation of the
kernel (glibc's libm) confirms it bit for bit; a GPU run has not yet.

The counts compare float64 nulls against the same thresholds, and sum a
group's members in the same order, as the NumPy backend; they are exact.
"""

import numpy as np

from copairs import _cuda
from copairs.nulls import sampler
from copairs.nulls.common import split_segments, segment_batches

cp = _cuda.cp
host = _cuda.host

# Gaps the float32 loop steps before it replays the float64 product. Its band
# covers the worst-case rounding below this many steps; unlike the bound itself
# (about 1.06e7 steps), that holds without reference to the callers' arguments.
F32_MAX_GAP = 2**22

# Constants shared with the CPU sampler, so both pick the same gap search.
_DEFINES = (
    f"#define GUIDED_RATIO {sampler.GUIDED_RATIO}LL\n"
    f"#define GUIDED_MAX_TOTAL {sampler.GUIDED_MAX_TOTAL}LL\n"
    f"#define LOG_MARGIN {sampler._LOG_MARGIN!r}\n"
    f"#define EPS {sampler._EPS!r}\n"
    f"#define FACTOR_LIMIT {sampler._FACTOR_LIMIT!r}\n"
    f"#define F32_MAX_GAP {F32_MAX_GAP}LL\n"
)

_SAMPLER = r"""
__device__ __forceinline__ void philox4x32(unsigned int c[4], unsigned int k0, unsigned int k1) {
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        unsigned int hi0 = __umulhi(0xD2511F53u, c[0]), lo0 = 0xD2511F53u * c[0];
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c[2]), lo1 = 0xCD9E8D57u * c[2];
        unsigned int n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
    }
}

__device__ __forceinline__ double uniform53(unsigned int a, unsigned int b) {
    unsigned long long x = ((unsigned long long)(a >> 5) << 26) | (b >> 6);
    return (double)x * (1.0 / 9007199254740992.0);
}

// Gap searches; mirror copairs.nulls.sampler.gap_loop and gap_guided.

// Algorithm A's float64 product after g steps, as gap_loop builds it.
__device__ double gap_product(long long remaining, long long k, long long g) {
    long long top = remaining - k;
    double quot = (double)top / (double)remaining;
    for (long long gap = 1; gap <= g; ++gap) {
        top -= 1;
        quot = quot * ((double)top / (double)(remaining - gap));
    }
    return quot;
}

// gap_loop resumed at step gap, whose product is quot.
__device__ long long gap_loop_from(long long remaining, long long k, double u, long long gap,
                                   double quot) {
    long long top = remaining - k - gap;
    while (quot > u) {
        gap += 1;
        top -= 1;
        quot = quot * ((double)top / (double)(remaining - gap));
    }
    return gap;
}

__device__ long long gap_loop(long long remaining, long long k, double u) {
    return gap_loop_from(remaining, k, u, 0, gap_product(remaining, k, 0));
}

// gap_loop run mostly in float32 (full rate on GPUs with few float64 units). A step
// is decided in float32 only when the float32 product clears u by its band;
// otherwise the float64 product is replayed and the loop finishes in float64, so
// the gap equals gap_loop's. Below 2^24 ranks the counts are exact in float32, and
// u >= 1e-30 keeps every product a normal float, so after g steps the float32
// product is within (1 +- 2^-24)^(2g + 1) of the exact one and the float64 product
// within (1 +- 2^-53)^(2g + 1). The band, with uf's and the thresholds' roundings,
// covers both for g < F32_MAX_GAP (tests/test_cuda.py checks every g).
__device__ long long gap_loop_f32(long long remaining, long long k, double u) {
    if (remaining >= (1LL << 24) || u < 1e-30) return gap_loop(remaining, k, u);
    float uf = (float)u;
    long long top = remaining - k, gap = 0;
    float quot = (float)top / (float)remaining;
    while (gap < F32_MAX_GAP) {
        float band = (float)(2 * gap + 4) * 1.2e-7f;
        if (quot > uf * (1.0f + band)) {
            gap += 1;
            top -= 1;
            quot = quot * ((float)top / (float)(remaining - gap));
            continue;
        }
        if (quot < uf * (1.0f - band)) return gap;
        break;
    }
    // Too close to call: replay the float64 product to this gap and finish in float64.
    return gap_loop_from(remaining, k, u, gap, gap_product(remaining, k, gap));
}

__device__ bool stops(long long remaining, long long k, long long g, double u, double log_u) {
    if ((double)(g + 1) / (double)(remaining - k + 1) <= FACTOR_LIMIT) {
        double acc = 0.0;
        for (long long t = 0; t < k; ++t) acc += log1p(-((double)(g + 1) / (double)(remaining - t)));
        double margin = LOG_MARGIN + (double)(2 * g + 2) * EPS;
        if (acc < log_u - margin) return true;
        if (acc > log_u + margin) return false;
    }
    return !(gap_product(remaining, k, g) > u);
}

__device__ long long gap_guided(long long remaining, long long k, double u) {
    long long last = remaining - k;
    if (u == 0.0) return gap_loop(remaining, k, u);
    double log_u = log(u);
    double guess = ((double)remaining - 0.5 * (double)(k - 1)) * (1.0 - exp(log_u / (double)k)) - 1.0;
    long long g = (long long)guess;
    g = g < 0 ? 0 : (g > last ? last : g);
    long long lo, hi, step = 1;
    if (stops(remaining, k, g, u, log_u)) {
        hi = g;
        lo = hi - step;
        while (lo >= 0 && stops(remaining, k, lo, u, log_u)) { hi = lo; step *= 2; lo = hi - step; }
        if (lo < -1) lo = -1;
    } else {
        lo = g;
        hi = lo + step;
        while (hi < last && !stops(remaining, k, hi, u, log_u)) { lo = hi; step *= 2; hi = lo + step; }
        if (hi > last) hi = last;
    }
    while (hi - lo > 1) {
        long long mid = (lo + hi) / 2;
        if (stops(remaining, k, mid, u, log_u)) hi = mid; else lo = mid;
    }
    return hi;
}

// Average precision of sample j; mirrors copairs.nulls.sampler._ap_sample.
__device__ double ap_sample(long long num_pos, long long total, unsigned long long j,
                            unsigned int k0, unsigned int k1) {
    long long remaining = total, k = num_pos, rank = 0, i = 0;
    double acc = 0.0, u, u_next = 0.0;
    while (k > 0) {
        if (k == remaining) {
            while (k > 0) { rank += 1; i += 1; acc += (double)i / (double)rank; k -= 1; }
            break;
        }
        if (i % 2 == 0) {
            unsigned int c[4] = {(unsigned int)(j & 0xFFFFFFFFull), (unsigned int)(j >> 32),
                                 (unsigned int)(i / 2), 0u};
            philox4x32(c, k0, k1);
            u = uniform53(c[0], c[1]);
            u_next = uniform53(c[2], c[3]);
        } else {
            u = u_next;
        }
        long long gap;
        if (k == 1) {
            gap = (long long)floor((double)remaining * u);
            if (gap > remaining - 1) gap = remaining - 1;
        } else if (remaining < GUIDED_MAX_TOTAL && remaining - k > GUIDED_RATIO * k * (k + 1)) {
            gap = gap_guided(remaining, k, u);
        } else {
            gap = gap_loop_f32(remaining, k, u);
        }
        rank += gap + 1;
        i += 1;
        acc += (double)i / (double)rank;
        remaining -= gap + 1;
        k -= 1;
    }
    return acc / (double)num_pos;
}

template <typename T>
__device__ void ap_nulls(const long long* num_pos, const long long* total,
                         const unsigned int* k0, const unsigned int* k1,
                         long long start, long long size, long long n_conf, T* out) {
    long long n = n_conf * size;
    for (long long flat = blockIdx.x * (long long)blockDim.x + threadIdx.x; flat < n;
         flat += (long long)gridDim.x * blockDim.x) {
        long long c = flat / size;
        long long t = flat - c * size;
        out[flat] = (T)ap_sample(num_pos[c], total[c], (unsigned long long)(start + t),
                                 k0[c], k1[c]);
    }
}

extern "C" __global__ void ap_nulls_f32(const long long* num_pos, const long long* total,
                                        const unsigned int* k0, const unsigned int* k1,
                                        long long start, long long size, long long n_conf,
                                        float* out) {
    ap_nulls<float>(num_pos, total, k0, k1, start, size, n_conf, out);
}

extern "C" __global__ void ap_nulls_f64(const long long* num_pos, const long long* total,
                                        const unsigned int* k0, const unsigned int* k1,
                                        long long start, long long size, long long n_conf,
                                        double* out) {
    ap_nulls<double>(num_pos, total, k0, k1, start, size, n_conf, out);
}
"""

_AP_NULLS = _DEFINES + _SAMPLER


def plan_on_device(plan):
    """Return a :func:`copairs.nulls.sampler.null_plan` copied to the GPU."""
    num_pos, total, k0, k1 = plan
    return (
        cp.asarray(num_pos, dtype=cp.int64),
        cp.asarray(total, dtype=cp.int64),
        cp.asarray(k0, dtype=cp.uint32),
        cp.asarray(k1, dtype=cp.uint32),
    )


def ap_nulls(plan, start: int, size: int, dtype):
    """``(n_conf, size)`` CuPy array of samples ``start..start + size - 1``.

    ``plan`` is a :func:`copairs.nulls.sampler.null_plan`, on the host or from
    :func:`plan_on_device`.
    """
    num_pos, total, k0, k1 = plan_on_device(plan)
    n_conf = len(num_pos)
    out = cp.empty((n_conf, size), dtype=dtype)
    if out.size == 0:
        return out
    args = (
        num_pos,
        total,
        k0,
        k1,
        np.int64(start),
        np.int64(size),
        np.int64(n_conf),
        out,
    )
    name = "ap_nulls_f64" if out.dtype == np.float64 else "ap_nulls_f32"
    kernel = _cuda.kernel(_AP_NULLS, name)
    kernel(_cuda.grid(out.size), (_cuda.BLOCK,), args)
    return out


def ap_nulls_to_host(plan, start: int, size: int, dtype, budget_bytes: int):
    """:func:`ap_nulls` as a NumPy array, sampled on the GPU chunk by chunk.

    Each chunk holds at most ``budget_bytes`` of samples on the GPU (at least
    one sample): whole configurations' rows while one fits, else pieces of
    one row. The chunk is copied to the host before the next is sampled.
    """
    dtype = np.dtype(dtype)
    plan = plan_on_device(plan)
    n_conf = len(plan[0])
    out = np.empty((n_conf, size), dtype=dtype)
    cols = max(1, min(size, budget_bytes // dtype.itemsize))
    rows = max(1, budget_bytes // (dtype.itemsize * cols)) if cols == size else 1
    for r in range(0, n_conf, rows):
        part = tuple(a[r : r + rows] for a in plan)
        for s in range(0, size, cols):
            n = min(cols, size - s)
            ap_nulls(part, start + s, n, dtype).get(out=out[r : r + rows, s : s + n])
    return out


_GROUP_GE = r"""
// counts[g] += #{t : sum_e conf_cnt[e] * null[conf_ix[e], t] / n_group[g] >= thr[g]}
extern "C" __global__ void group_ge(const double* null, long long size, long long n_groups,
                                    const long long* ptr, const long long* conf_ix,
                                    const long long* conf_cnt, const long long* n_group,
                                    const double* thr, unsigned long long* counts) {
    for (long long g = blockIdx.y; g < n_groups; g += gridDim.y) {
        long long lo = ptr[g], hi = ptr[g + 1];
        unsigned long long hits = 0;
        for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < size;
             t += (long long)gridDim.x * blockDim.x) {
            double acc = 0.0;
            for (long long e = lo; e < hi; ++e)
                acc += (double)conf_cnt[e] * null[conf_ix[e] * size + t];
            if (acc / (double)n_group[g] >= thr[g]) hits += 1;
        }
        for (int off = 16; off > 0; off >>= 1) hits += __shfl_down_sync(0xffffffffu, hits, off);
        if ((threadIdx.x & 31) == 0 && hits) atomicAdd(counts + g, hits);
    }
}
"""


def group_ge(null, ptr, conf_ix, conf_cnt, n_group, thr, counts):
    """GPU counterpart of :func:`copairs.nulls.pvalues._group_ge_host`.

    ``null`` is a ``(n_conf, size)`` chunk on the GPU; the groups (``ptr``,
    ``conf_ix``, ``conf_cnt``, ``n_group`` and ``thr``) and their int64
    ``counts`` are on the host, as there. The groups go to the GPU with each
    chunk: 8 bytes per group and member.
    """
    null = cp.ascontiguousarray(null, dtype=cp.float64)
    size, n_groups = null.shape[1], len(ptr) - 1
    groups = (cp.asarray(a, dtype=cp.int64) for a in (ptr, conf_ix, conf_cnt, n_group))
    thr = cp.asarray(thr, dtype=cp.float64)
    hits = cp.zeros(n_groups, dtype=cp.uint64)
    _cuda.kernel(_GROUP_GE, "group_ge")(
        _cuda.grid(size, blocks_per_sm=8, rows=n_groups),
        (_cuda.BLOCK,),
        (null, np.int64(size), np.int64(n_groups), *groups, thr, hits),
    )
    counts += host(hits).astype(np.int64)


_COUNT_KERNELS = r"""
// Bin ptr[s] + s + q counts the samples of null row rows[s] with exactly q thresholds
// of segment s <= the sample (thresholds sorted, NaN last; ptr[0] == 0).
extern "C" __global__ void exceed_hist(const double* null, long long size, long long n_seg,
                                       const long long* rows, const long long* ptr,
                                       const double* thr, unsigned long long* hist) {
    for (long long flat = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         flat < n_seg * size; flat += (long long)gridDim.x * blockDim.x) {
        long long s = flat / size;
        long long lo = ptr[s], hi = ptr[s + 1];
        if (hi == lo) continue;
        double x = null[rows[s] * size + (flat - s * size)];
        atomicAdd(hist + (upper_bound(thr, lo, hi, x) + s), 1ull);
    }
}

// counts[p] += #{samples >= thr[p]}: the suffix sums of each segment's bins.
extern "C" __global__ void suffix_add(const long long* ptr, long long n_seg,
                                      const unsigned long long* hist, long long* counts) {
    for (long long s = blockIdx.x * (long long)blockDim.x + threadIdx.x; s < n_seg;
         s += (long long)gridDim.x * blockDim.x) {
        long long lo = ptr[s], hi = ptr[s + 1], off = lo + s;
        unsigned long long run = 0;
        for (long long q = hi - lo - 1; q >= 0; --q) {
            run += hist[off + q + 1];
            counts[lo + q] += (long long)run;
        }
    }
}
"""
_COUNT_GE = _cuda.PRELUDE + _COUNT_KERNELS

# GPU bytes of count_ge per bin of a batch: the bin, and the threshold and count
# of its score (a segment's extra bin pays for its row and offset), plus the
# offsets' end.
_BIN_BYTES = 24
_BATCH_BYTES = 8


def count_ge(null, thr, ptr, counts, budget_bytes: int):
    """GPU counterpart of :func:`copairs.nulls.pvalues._count_ge_host`.

    ``null`` is a ``(n_conf, size)`` chunk on the GPU; ``thr`` (the sorted
    float64 thresholds), ``ptr`` (each configuration's segment of ``thr``) and
    ``counts`` (their int64 counts) are on the host. The thresholds go to the
    GPU in batches whose thresholds, counts and histogram bins take at most
    ``budget_bytes``, a configuration's thresholds split where they alone
    would exceed it, as in the Numba backend. Each piece counts its own
    thresholds exactly, so the counts do not depend on the batches.
    """
    n_conf, size = null.shape
    null = cp.ascontiguousarray(null, dtype=cp.float64)
    max_bins = max(2, (budget_bytes - _BATCH_BYTES) // _BIN_BYTES)
    rows, ptr = split_segments(
        np.arange(n_conf, dtype=np.int64), np.asarray(ptr, dtype=np.int64), max_bins - 1
    )
    for first, end in segment_batches(ptr, max_bins):
        lo, hi = int(ptr[first]), int(ptr[end])
        if hi == lo:
            continue
        seg = cp.asarray(ptr[first : end + 1] - lo)
        n_seg = np.int64(end - first)
        hist = cp.zeros(hi - lo + int(n_seg), dtype=cp.uint64)
        part = cp.zeros(hi - lo, dtype=cp.int64)
        args = (null, np.int64(size), n_seg, cp.asarray(rows[first:end]), seg)
        _cuda.kernel(_COUNT_GE, "exceed_hist")(
            _cuda.grid(int(n_seg) * size),
            (_cuda.BLOCK,),
            (*args, cp.asarray(thr[lo:hi]), hist),
        )
        _cuda.kernel(_COUNT_GE, "suffix_add")(
            _cuda.grid(int(n_seg)), (_cuda.BLOCK,), (seg, n_seg, hist, part)
        )
        counts[lo:hi] += host(part)
