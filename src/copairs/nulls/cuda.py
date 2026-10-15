"""CUDA (CuPy) backend for the AP null sampler.

The kernel is a line-by-line port of :func:`copairs.nulls.sampler._ap_sample`:
the same Philox blocks, the same double-precision operations in the same order,
and FMA contraction disabled, so it returns the same float32 values as the
NumPy and Numba backends.
"""

import os
import warnings
import functools

import numpy as np

from copairs.nulls import sampler

try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only without cupy
    cp = None

# Constants shared with the CPU sampler, so both pick the same gap search.
_DEFINES = (
    f"#define GUIDED_RATIO {sampler.GUIDED_RATIO}LL\n"
    f"#define GUIDED_MAX_TOTAL {sampler.GUIDED_MAX_TOTAL}LL\n"
    f"#define LOG_MARGIN {sampler._LOG_MARGIN!r}\n"
    f"#define EPS {sampler._EPS!r}\n"
    f"#define FACTOR_LIMIT {sampler._FACTOR_LIMIT!r}\n"
)

_PHILOX = (
    _DEFINES
    + r"""
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
__device__ long long gap_loop(long long remaining, long long k, double u) {
    long long top = remaining - k, gap = 0;
    double quot = (double)top / (double)remaining;
    while (quot > u) {
        gap += 1;
        top -= 1;
        quot = quot * ((double)top / (double)(remaining - gap));
    }
    return gap;
}

// gap_loop run mostly in float32 (full rate on GPUs with few float64 units). A step
// is decided in float32 only when the float32 product clears u by more than its
// error bound relative to the float64 product; otherwise the float64 product is
// replayed and the loop finishes in float64, so the gap equals gap_loop's.
__device__ long long gap_loop_f32(long long remaining, long long k, double u) {
    if (remaining >= (1LL << 24) || u < 1e-30) return gap_loop(remaining, k, u);
    float uf = (float)u;
    long long top = remaining - k, gap = 0;
    float quot = (float)top / (float)remaining;
    while (true) {
        // |float32 product / float64 product - 1| <= (2 gap + 1) 2^-24 (1 + 2^-23), plus
        // the roundings of uf and of the thresholds: covered twice over.
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
    long long t = remaining - k;
    double q = (double)t / (double)remaining;
    for (long long g = 1; g <= gap; ++g) {
        t -= 1;
        q = q * ((double)t / (double)(remaining - g));
    }
    while (q > u) {
        gap += 1;
        t -= 1;
        q = q * ((double)t / (double)(remaining - gap));
    }
    return gap;
}

__device__ bool stops(long long remaining, long long k, long long g, double u, double log_u) {
    if ((double)(g + 1) / (double)(remaining - k + 1) <= FACTOR_LIMIT) {
        double acc = 0.0;
        for (long long t = 0; t < k; ++t) acc += log1p(-((double)(g + 1) / (double)(remaining - t)));
        double margin = LOG_MARGIN + (double)(2 * g + 2) * EPS;
        if (acc < log_u - margin) return true;
        if (acc > log_u + margin) return false;
    }
    long long top = remaining - k;
    double quot = (double)top / (double)remaining;
    for (long long gap = 1; gap <= g; ++gap) {
        top -= 1;
        quot = quot * ((double)top / (double)(remaining - gap));
    }
    return !(quot > u);
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
"""
)

_AP_NULLS = (
    _PHILOX
    + r"""
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
)

_OPTIONS = ("--fmad=false", "-std=c++14")
_BLOCK = 256


_PROBE = 'extern "C" __global__ void probe(int* x) { if (x) *x = 1; }'


def is_available() -> bool:
    """Whether CuPy sees a CUDA device and can compile kernels for it.

    Cached per process: a child forked after its parent initialised CUDA
    cannot use it, so the child probes again and falls back to the CPU.
    """
    return _probe(os.getpid())


@functools.cache
def _probe(pid: int) -> bool:
    if cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return False
    except (RuntimeError, ImportError, OSError):
        return False  # no driver, or CUDA libraries that fail to load
    try:
        # A visible GPU is not enough: without NVRTC (e.g. CuPy without its
        # [ctk] extra or a CUDA Toolkit) every kernel fails to compile.
        cp.RawKernel(_PROBE, "probe", options=_OPTIONS).compile()
    except (RuntimeError, ImportError, OSError, cp.cuda.compiler.CompileException) as e:
        warnings.warn(
            f"a CUDA device is visible but copairs' kernels fail to compile "
            f"({type(e).__name__}: {e}); the CUDA backend is unavailable",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


@functools.cache
def _compiled(source: str, device_id: int):
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=source, options=_OPTIONS)


def kernel(source: str, name: str):
    """Kernel ``name`` of CUDA ``source``, compiled once per device."""
    return _compiled(source, cp.cuda.Device().id).get_function(name)


def grid(n: int, blocks_per_sm: int = 32) -> int:
    """1-D grid size for ``n`` threads of ``_BLOCK``, capped per multiprocessor."""
    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    return int(max(1, min((n + _BLOCK - 1) // _BLOCK, sms * blocks_per_sm)))


def ap_nulls(num_pos, total, k0, k1, start: int, size: int, dtype=np.float32):
    """``(n_conf, size)`` CuPy array of AP null samples in ``dtype``."""
    n_conf = len(num_pos)
    out = cp.empty((n_conf, size), dtype=dtype)
    if out.size == 0:
        return out
    args = (
        cp.asarray(num_pos, dtype=cp.int64),
        cp.asarray(total, dtype=cp.int64),
        cp.asarray(k0, dtype=cp.uint32),
        cp.asarray(k1, dtype=cp.uint32),
        np.int64(start),
        np.int64(size),
        np.int64(n_conf),
        out,
    )
    name = "ap_nulls_f64" if out.dtype == np.float64 else "ap_nulls_f32"
    kernel(_AP_NULLS, name)((grid(out.size),), (_BLOCK,), args)
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


class GroupCounter:
    """Accumulate group-null exceedance counts over chunks of AP null samples."""

    def __init__(self, ptr, conf_ix, conf_cnt, n_group, thr):
        self.n_groups = len(ptr) - 1
        self.args = tuple(cp.asarray(a) for a in (ptr, conf_ix, conf_cnt, n_group, thr))
        self._counts = cp.zeros(self.n_groups, dtype=cp.uint64)

    def __call__(self, null):
        """Add the exceedances of one ``(n_conf, size)`` float64 chunk of nulls."""
        size = null.shape[1]
        gx = grid(size, blocks_per_sm=8)
        gy = int(min(self.n_groups, 65535))
        kernel(_GROUP_GE, "group_ge")(
            (gx, gy),
            (_BLOCK,),
            (null, np.int64(size), np.int64(self.n_groups), *self.args, self._counts),
        )

    def counts(self) -> np.ndarray:
        """Total exceedance count of each group."""
        return self._counts.get().astype(np.int64)


_COUNT_GE = r"""
// Bin (ptr[c] - ptr[0] + c) + q counts the samples of configuration c with exactly q
// thresholds of its segment <= the sample (thresholds sorted, NaN last).
extern "C" __global__ void exceed_hist(const double* null, long long size, long long n_conf,
                                       const long long* ptr, const double* thr,
                                       unsigned long long* hist) {
    for (long long flat = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         flat < n_conf * size; flat += (long long)gridDim.x * blockDim.x) {
        long long c = flat / size;
        long long lo = ptr[c], hi = ptr[c + 1];
        if (hi == lo) continue;
        double x = null[flat];
        long long a = lo, b = hi;
        while (a < b) {
            long long mid = (a + b) >> 1;
            double v = thr[mid];
            if (!isnan(v) && v <= x) a = mid + 1; else b = mid;
        }
        atomicAdd(hist + (a - ptr[0] + c), 1ull);
    }
}

// counts[p] += #{samples >= thr[p]}: the suffix sums of each configuration's bins.
extern "C" __global__ void suffix_add(const long long* ptr, long long n_conf,
                                      const unsigned long long* hist, long long* counts) {
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x; c < n_conf;
         c += (long long)gridDim.x * blockDim.x) {
        long long lo = ptr[c], hi = ptr[c + 1], off = lo - ptr[0] + c;
        unsigned long long run = 0;
        for (long long q = hi - lo - 1; q >= 0; --q) {
            run += hist[off + q + 1];
            counts[lo + q] += (long long)run;
        }
    }
}
"""


def count_ge(null, thr, ptr, counts):
    """GPU counterpart of ``pvalues._count_ge_host`` on device arrays."""
    n_conf, size = null.shape
    ptr = np.asarray(ptr, dtype=np.int64)
    hist = cp.zeros(int(ptr[-1] - ptr[0]) + n_conf, dtype=cp.uint64)
    ptr = cp.asarray(ptr)
    null = cp.ascontiguousarray(null, dtype=cp.float64)
    args = (null, np.int64(size), np.int64(n_conf), ptr, thr, hist)
    kernel(_COUNT_GE, "exceed_hist")((grid(null.size),), (_BLOCK,), args)
    kernel(_COUNT_GE, "suffix_add")(
        (grid(n_conf),), (_BLOCK,), (ptr, np.int64(n_conf), hist, counts)
    )
