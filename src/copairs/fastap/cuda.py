"""CUDA kernels behind the AP stage's ``backend="cuda"`` (optional, CuPy).

Imported only through ``copairs._backend.backend_module("cuda")`` (``pip
install 'copairs[gpu]'``). It offers the interface of :mod:`copairs.fastap`
that the AP functions use:

- :func:`similarity_fn` and :func:`pair_similarity` compute pair similarities
  on the GPU, optionally keeping them there;
- :func:`ap_from_pairs` computes AP by counting on the GPU, streaming the
  negative pairs in chunks of :data:`PAIR_CHUNK`;
- multilabel APs, the multilabel matcher and normalized AP run on Numba, on the
  CPU (:mod:`copairs.fastap.multilabel`, :func:`expected_ap_array`).

Given the same similarities, APs equal the Numba backend's bit for bit: the
counts are exact and the AP sums run in the same order. The kernels'
similarities sum in another order than Numba's, so they match to float32
rounding and near-tied pairs can rank differently.
"""

import numpy as np

from copairs import _cuda
from copairs.fastap import draws, ranking, multilabel, _bind_similarity
from copairs.fastap.ranking import rank_keys, index_pairs, sortable_keys
from copairs.fastap.similarity import FAST_METRICS, MINKOWSKI_METRICS, _unit_rows
from copairs.fastap.normalization import expected_ap_array

__all__ = [
    "MAX_DRAW_QUERIES",
    "PAIR_CHUNK",
    "ap_from_pairs",
    "expected_ap_array",
    "multilabel",
    "pair_similarity",
    "similarity_fn",
]

cp = _cuda.cp
# Pairs, and negatives' keys, copied to the GPU at a time.
PAIR_CHUNK = 1 << 24

_SOURCE = r"""
// NumPy ordering: NaN sorts after every number and ties with NaN.
__device__ __forceinline__ bool le(float v, float key) {
    return isnan(key) || (!isnan(v) && v <= key);
}

template <typename T>
__device__ void dot_pairs(const T* x, long long d, const unsigned int* pairs, long long n,
                          float* out) {
    int lane = threadIdx.x & 31;
    long long warp = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    long long n_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long p = warp; p < n; p += n_warps) {
        const T* a = x + (long long)pairs[2 * p] * d;
        const T* b = x + (long long)pairs[2 * p + 1] * d;
        double acc = 0.0;
        for (long long t = lane; t < d; t += 32) acc += (double)a[t] * (double)b[t];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[p] = (float)acc;
    }
}

// max that propagates NaN, like np.max (fmax drops it).
__device__ __forceinline__ double nan_max(double a, double b) {
    return (isnan(a) || isnan(b)) ? (a + b) : fmax(a, b);
}

// 1 / (1 + distance); kind 0 = euclidean, 1 = manhattan, 2 = chebyshev.
template <typename T>
__device__ void minkowski_pairs(const T* x, long long d, const unsigned int* pairs, long long n,
                                int kind, float* out) {
    int lane = threadIdx.x & 31;
    long long warp = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    long long n_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long p = warp; p < n; p += n_warps) {
        const T* a = x + (long long)pairs[2 * p] * d;
        const T* b = x + (long long)pairs[2 * p + 1] * d;
        double acc = 0.0;
        for (long long t = lane; t < d; t += 32) {
            double diff = fabs((double)a[t] - (double)b[t]);
            acc = kind == 0 ? acc + diff * diff : kind == 1 ? acc + diff : nan_max(acc, diff);
        }
        for (int off = 16; off > 0; off >>= 1) {
            double other = __shfl_down_sync(0xffffffffu, acc, off);
            acc = kind == 2 ? nan_max(acc, other) : acc + other;
        }
        if (lane == 0) out[p] = (float)(1.0 / (1.0 + (kind == 0 ? sqrt(acc) : acc)));
    }
}

extern "C" __global__ void dot_pairs_f32(const float* x, long long d, const unsigned int* pairs,
                                         long long n, float* out) {
    dot_pairs<float>(x, d, pairs, n, out);
}
extern "C" __global__ void dot_pairs_f64(const double* x, long long d, const unsigned int* pairs,
                                         long long n, float* out) {
    dot_pairs<double>(x, d, pairs, n, out);
}
extern "C" __global__ void minkowski_pairs_f32(const float* x, long long d,
                                               const unsigned int* pairs, long long n, int kind,
                                               float* out) {
    minkowski_pairs<float>(x, d, pairs, n, kind, out);
}
extern "C" __global__ void minkowski_pairs_f64(const double* x, long long d,
                                               const unsigned int* pairs, long long n, int kind,
                                               float* out) {
    minkowski_pairs<double>(x, d, pairs, n, kind, out);
}

// For both endpoints i of each negative pair, count it in bin ptr[i] + i + q, where
// q = #{positive keys of i <= key}.
extern "C" __global__ void negative_hist(const unsigned int* neg_pairs, const float* keys,
                                         long long n_neg, const long long* ptr,
                                         const float* vals, unsigned long long* hist,
                                         unsigned long long* n_negative) {
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < 2 * n_neg;
         e += (long long)gridDim.x * blockDim.x) {
        long long i = neg_pairs[e];
        float key = keys[e >> 1];
        long long lo = ptr[i], hi = ptr[i + 1];
        while (lo < hi) {
            long long mid = (lo + hi) >> 1;
            if (le(vals[mid], key)) lo = mid + 1; else hi = mid;
        }
        atomicAdd(hist + lo + i, 1ull);
        atomicAdd(n_negative + i, 1ull);
    }
}

extern "C" __global__ void ap_from_hist(const long long* ptr, const unsigned long long* hist,
                                        long long n, double* ap) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        long long num_pos = ptr[i + 1] - ptr[i], base = ptr[i] + i;
        if (num_pos == 0) { ap[i] = nan(""); continue; }
        unsigned long long before = 0;
        double acc = 0.0;
        for (long long t = 0; t < num_pos; ++t) {
            before += hist[base + t];
            acc += (double)(t + 1) / (double)(t + 1 + before);
        }
        ap[i] = acc / (double)num_pos;
    }
}
"""


def _kernel(name: str):
    return _cuda.kernel(_SOURCE, name)


def _device_pairs(pairs):
    """Contiguous uint32 ``(n, 2)`` pairs on the GPU (indices are < 2**32)."""
    pairs = cp.asarray(pairs).reshape(-1, 2)
    return cp.ascontiguousarray(pairs.astype(cp.uint32, copy=False))


def pair_similarity(feats: np.ndarray, metric, on_device: bool = False):
    """GPU counterpart of :func:`copairs.fastap.pair_similarity`.

    The kernel computes in float64 like the Numba one, from the same unit
    rows, and returns float32 similarities: a NumPy array, or with
    ``on_device`` a CuPy array. Pairs are checked on the host and copied to
    the GPU :data:`PAIR_CHUNK` at a time.
    """
    if not isinstance(metric, str) or metric not in FAST_METRICS:
        return None
    feats = np.asarray(feats)
    if feats.dtype not in (np.float32, np.float64):
        feats = feats.astype(np.float64)
    if metric in MINKOWSKI_METRICS:
        x, name = feats, "minkowski_pairs"
        extra = (np.int32(MINKOWSKI_METRICS.index(metric)),)
    else:
        x, name, extra = _unit_rows(feats, metric == "correlation"), "dot_pairs", ()
    x = cp.asarray(np.ascontiguousarray(x))
    name += "_f64" if x.dtype == cp.float64 else "_f32"
    d = np.int64(x.shape[1])

    def similarity(pairs):
        pairs = index_pairs(pairs, len(x))
        out = cp.empty(len(pairs), dtype=cp.float32)
        for start in range(0, len(pairs), PAIR_CHUNK):
            chunk = _device_pairs(pairs[start : start + PAIR_CHUNK])
            n = len(chunk)
            args = (x, d, chunk, np.int64(n), *extra, out[start : start + n])
            _kernel(name)((_cuda.grid(32 * n),), (_cuda.BLOCK,), args)
        if metric == "abs_cosine":
            cp.abs(out, out=out)
        return out if on_device else out.get()

    return similarity


def similarity_fn(feats, distance, generic, on_device: bool = False):
    """:func:`copairs.fastap.similarity_fn` with the GPU kernels.

    With ``on_device``, the kernels' similarities stay on the GPU for
    :func:`ap_from_pairs`; distances without a kernel use ``generic``.
    """
    return _bind_similarity(feats, pair_similarity(feats, distance, on_device), generic)


def _pair_sims(sims, pairs):
    """:func:`copairs.fastap.ranking.pair_sims` for NumPy or CuPy similarities."""
    if not _cuda.is_device_array(sims):
        return ranking.pair_sims(sims, pairs)
    if sims.shape != (len(pairs),):
        raise ValueError(
            f"need one similarity per pair: {len(pairs)} pairs, sims of shape "
            f"{sims.shape}"
        )
    return sims


def ap_from_pairs(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims,
    neg_sims,
    budget_bytes: int = 2**30,
    n: int | None = None,
):
    """:func:`copairs.fastap.ap_from_pairs` on the GPU; returns NumPy arrays.

    Similarities may be NumPy or CuPy arrays. Beyond them the GPU holds the
    positive pairs, their sorted keys and the histograms; negative pairs and
    their keys are copied :data:`PAIR_CHUNK` at a time, and the histograms
    add up across chunks, so chunking does not change the result. ``n`` of
    ``2**32`` or more, whose indices the GPU sort cannot pack, runs on Numba,
    the only place ``budget_bytes`` applies.
    """
    n = None if n is None else ranking.profile_count(n)
    pos_pairs, neg_pairs = index_pairs(pos_pairs, n), index_pairs(neg_pairs, n)
    if n is None:
        n = 1 + max(int(p.max()) if len(p) else -1 for p in (pos_pairs, neg_pairs))
    pos_sims = _pair_sims(pos_sims, pos_pairs)
    neg_sims = _pair_sims(neg_sims, neg_pairs)
    if n >= 2**32:
        return ranking.ap_from_pairs(
            pos_pairs,
            neg_pairs,
            _cuda.host(pos_sims),
            _cuda.host(neg_sims),
            budget_bytes,
            n,
        )
    profile = _device_pairs(pos_pairs).ravel()
    keys = cp.repeat(rank_keys(pos_sims, cp), 2)
    order = cp.argsort((profile.astype(cp.uint64) << 32) | sortable_keys(keys, cp))
    vals = cp.ascontiguousarray(keys[order])
    ptr = cp.searchsorted(
        profile[order].astype(cp.int64), cp.arange(n + 1, dtype=cp.int64)
    )
    del profile, keys, order
    hist = cp.zeros(int(ptr[-1]) + n, dtype=cp.uint64)
    n_neg = cp.zeros(n, dtype=cp.uint64)
    for start in range(0, len(neg_pairs), PAIR_CHUNK):
        pairs = _device_pairs(neg_pairs[start : start + PAIR_CHUNK])
        keys = rank_keys(neg_sims[start : start + PAIR_CHUNK], cp)
        args = (pairs, keys, np.int64(len(pairs)), ptr, vals, hist, n_neg)
        _kernel("negative_hist")((_cuda.grid(2 * len(pairs)),), (_cuda.BLOCK,), args)
    ap = cp.empty(n, dtype=cp.float64)
    _kernel("ap_from_hist")(
        (_cuda.grid(n),), (_cuda.BLOCK,), (ptr, hist, np.int64(n), ap)
    )
    ap, num_pos = ap.get(), cp.diff(ptr).get()
    total = num_pos + n_neg.get().astype(np.int64)
    paired_ix = np.flatnonzero(total)
    null_confs = np.stack([num_pos[paired_ix], total[paired_ix]], axis=1)
    return paired_ix, ap[paired_ix], null_confs.astype(np.uint32)


# draw_ap holds k float32 keys and k uint32 bins in shared memory, of which a
# kernel gets 48 KiB without opting in. Larger draws run on Numba.
MAX_DRAW_QUERIES = 48 * 1024 // 8

_DRAW_SOURCE = r"""
// NumPy order of float32 keys: NaN after every number, and ties with NaN.
__device__ __forceinline__ bool key_lt(float a, float b) {
    return !isnan(a) && (isnan(b) || a < b);
}
__device__ __forceinline__ bool key_le(float v, float key) {
    return isnan(key) || (!isnan(v) && v <= key);
}

// AP of each (draw, query) row of sims (n_rows, k + m), ranked by the float32 key
// 1 - similarity with a positive ahead of a negative of the same key, as
// copairs.fastap.draws._row_ap: the first k columns are the draw's queries (the
// row's own column is skipped), the rest its references.
extern "C" __global__ void draw_ap(const float* sims, int k, int m, long long n_rows,
                                   double* ap) {
    extern __shared__ float smem[];
    float* pos = smem;                                   // k - 1 keys, ascending
    unsigned int* hist = (unsigned int*)(smem + k);      // k bins
    for (long long row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const float* s = sims + row * (long long)(k + m);
        int q = (int)(row % k);
        for (int j = threadIdx.x; j < k; j += blockDim.x) hist[j] = 0u;
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            if (j == q) continue;
            float v = 1.0f - s[j];
            int r = 0;  // rank of key v among the positives', ties by column
            for (int i = 0; i < k; ++i) {
                if (i == q || i == j) continue;
                float w = 1.0f - s[i];
                r += key_lt(w, v) || (!key_lt(v, w) && i < j);
            }
            pos[r] = v;
        }
        __syncthreads();
        for (int e = threadIdx.x; e < m; e += blockDim.x) {
            float key = 1.0f - s[k + e];
            int lo = 0, hi = k - 1;  // positives ranked ahead of this negative
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (key_le(pos[mid], key)) lo = mid + 1; else hi = mid;
            }
            atomicAdd(hist + lo, 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long before = 0;
            double acc = 0.0;
            for (int t = 0; t < k - 1; ++t) {
                before += hist[t];
                acc += (double)(t + 1) / (double)(t + 1 + before);
            }
            ap[row] = acc / (double)(k - 1);
        }
        __syncthreads();
    }
}
"""


def draw_aps(x: np.ndarray, idx: np.ndarray, k: int, budget_bytes: int) -> np.ndarray:
    """``backend="cuda"`` of :func:`copairs.fastap.draw_average_precisions`.

    Takes its checked float32 unit rows ``x`` and ``(n_draws, k + m)`` row
    indices ``idx`` (queries first), for ``k <= MAX_DRAW_QUERIES``. Each
    chunk of draws' similarities comes from one batched matrix product on
    the GPU, within ``budget_bytes`` of GPU memory.
    """
    x = cp.asarray(x)
    n_draws, width = idx.shape
    m = width - k
    out = cp.empty((n_draws, k), dtype=cp.float64)
    kernel = _cuda.kernel(_DRAW_SOURCE, "draw_ap")
    threads = 128 if k + m >= 128 else 32
    step = draws._chunk(n_draws, k, m, x.shape[1], budget_bytes)
    for start in range(0, n_draws, step):
        rows = x[cp.asarray(idx[start : start + step])]  # (b, k + m, d)
        sims = cp.ascontiguousarray(cp.matmul(rows[:, :k], rows.transpose(0, 2, 1)))
        n_rows = sims.shape[0] * k
        args = (
            sims,
            np.int32(k),
            np.int32(m),
            np.int64(n_rows),
            out[start : start + step],
        )
        kernel((int(min(n_rows, 65535 * 4)),), (threads,), args, shared_mem=8 * k)
        del rows, sims  # free this chunk before gathering the next one
    return out.get()
