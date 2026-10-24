"""CUDA kernels behind the AP stage's ``backend="cuda"`` (optional, CuPy).

Imported only through ``copairs._backend.backend_module("cuda")`` (``pip
install 'copairs[gpu]'``). It offers the interface of :mod:`copairs.fastap`
that the AP functions use:

- :func:`pair_similarity` computes pair similarities on the GPU, and
  :func:`similarity_fn` leaves them there as :class:`DeviceSims`, which
  :func:`ap_from_pairs` computes chunk by chunk and ``np.asarray`` copies to
  the host;
- :func:`ap_from_pairs` computes AP by counting on the GPU, streaming the
  negative pairs, their similarities and keys within ``budget_bytes``;
- the AP stage's steps without GPU kernels (multilabel matching and APs,
  normalized AP) run on Numba under ``"cuda"`` (``copairs._backend.kernels``).

Given the same similarities, APs equal the Numba backend's bit for bit: the
counts are exact and the AP sums run in the same order. The kernels'
similarities sum in another order than Numba's, so they match to float32
rounding and near-tied pairs can rank differently.
"""

import numpy as np

from copairs import _cuda
from copairs.fastap import draws, ranking, _bind_similarity
from copairs.fastap.ranking import rank_keys, index_pairs, sortable_keys
from copairs.fastap.similarity import kernel_rows

__all__ = [
    "PAIR_CHUNK",
    "DeviceSims",
    "ap_from_pairs",
    "pair_similarity",
    "similarity_fn",
]

cp = _cuda.cp
# Pairs whose similarities the GPU computes at a time for the host.
PAIR_CHUNK = 1 << 24
# GPU bytes per negative pair that ap_from_pairs streams: the uint32 pair, its
# float32 similarity and its float32 key.
_NEGATIVE_BYTES = 16

_PAIR_KERNELS = r"""
// Compiled with --fmad=false (copairs._cuda.OPTIONS): the counting kernels round as
// the CPU does. The similarity sums, which match the CPU's only to rounding anyway,
// ask for fused multiply-adds with fma().
// max that propagates NaN, like np.max (fmax drops it).
__device__ __forceinline__ double nan_max(double a, double b) {
    return (isnan(a) || isnan(b)) ? (a + b) : fmax(a, b);
}

// Similarity of each pair, one warp per pair: kind -1 = dot product (of unit rows),
// else 1 / (1 + distance) with kind 0 = euclidean, 1 = manhattan, 2 = chebyshev.
template <typename T>
__device__ void pair_sims(const T* x, long long d, const unsigned int* pairs, long long n,
                          int kind, float* out) {
    int lane = threadIdx.x & 31;
    long long warp = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    long long n_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long p = warp; p < n; p += n_warps) {
        const T* a = x + (long long)pairs[2 * p] * d;
        const T* b = x + (long long)pairs[2 * p + 1] * d;
        double acc = 0.0;
        for (long long t = lane; t < d; t += 32) {
            if (kind < 0) {
                acc = fma((double)a[t], (double)b[t], acc);
                continue;
            }
            double diff = fabs((double)a[t] - (double)b[t]);
            acc = kind == 0 ? fma(diff, diff, acc) : kind == 1 ? acc + diff : nan_max(acc, diff);
        }
        for (int off = 16; off > 0; off >>= 1) {
            double other = __shfl_down_sync(0xffffffffu, acc, off);
            acc = kind == 2 ? nan_max(acc, other) : acc + other;
        }
        if (lane == 0)
            out[p] = (float)(kind < 0 ? acc : 1.0 / (1.0 + (kind == 0 ? sqrt(acc) : acc)));
    }
}

extern "C" __global__ void pair_sims_f32(const float* x, long long d, const unsigned int* pairs,
                                         long long n, int kind, float* out) {
    pair_sims<float>(x, d, pairs, n, kind, out);
}
extern "C" __global__ void pair_sims_f64(const double* x, long long d, const unsigned int* pairs,
                                         long long n, int kind, float* out) {
    pair_sims<double>(x, d, pairs, n, kind, out);
}

// For both endpoints i of each negative pair, count it in bin ptr[i] + i + q, where
// q = #{positive keys of i <= key}.
extern "C" __global__ void negative_hist(const unsigned int* neg_pairs, const float* keys,
                                         long long n_neg, const long long* ptr,
                                         const float* vals, unsigned long long* hist) {
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < 2 * n_neg;
         e += (long long)gridDim.x * blockDim.x) {
        long long i = neg_pairs[e];
        float key = keys[e >> 1];
        atomicAdd(hist + upper_bound(vals, ptr[i], ptr[i + 1], key) + i, 1ull);
    }
}

// AP of each profile i from its bins, and its number of negatives: all its bins.
extern "C" __global__ void ap_from_hist(const long long* ptr, const unsigned long long* hist,
                                        long long n, double* ap, unsigned long long* n_neg) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        long long num_pos = ptr[i + 1] - ptr[i], base = ptr[i] + i;
        unsigned long long before;
        double value = ap_from_counts(hist + base, num_pos, &before);
        ap[i] = num_pos == 0 ? nan("") : value;
        n_neg[i] = before + hist[base + num_pos];
    }
}
"""


def _kernel(name: str):
    """Kernel ``name`` of the module's one source (:data:`_SOURCE`)."""
    return _cuda.kernel(_SOURCE, name)


def _device_pairs(pairs):
    """Contiguous uint32 ``(n, 2)`` pairs on the GPU, for indices below ``2**32``.

    Cast on the host, so only 8 bytes per pair are copied and held.
    """
    return cp.asarray(np.ascontiguousarray(pairs, dtype=np.uint32).reshape(-1, 2))


class DeviceSims:
    """Similarities of ``pairs``, computed on the GPU only when they are used.

    :func:`ap_from_pairs` computes them chunk by chunk with its own pairs,
    so all of them are never held at once, and :meth:`get` (or
    ``np.asarray``, as the multilabel APs on Numba do) returns them all as
    a NumPy array. ``source`` is the pairs object they were made for.
    """

    def __init__(self, sims_of, pairs: np.ndarray, source):
        self._sims_of = sims_of
        self.pairs = pairs
        self.source = source

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.pairs)

    def of(self, device_pairs):
        """Float32 similarities of ``device_pairs`` (uint32, on the GPU), on the GPU."""
        return self._sims_of(device_pairs)

    def get(self) -> np.ndarray:
        """All similarities as a NumPy array, computed :data:`PAIR_CHUNK` at a time."""
        out = np.empty(len(self.pairs), dtype=np.float32)
        for start in range(0, len(out), PAIR_CHUNK):
            chunk = self.pairs[start : start + PAIR_CHUNK]
            self.of(_device_pairs(chunk)).get(out=out[start : start + len(chunk)])
        return out

    def __array__(self, dtype=None, copy=None):
        """:meth:`get`, for NumPy; a host array is always a new one."""
        if copy is False:
            raise ValueError("similarities on the GPU have no host array to share")
        return self.get().astype(dtype or np.float32, copy=False)


def pair_similarity(feats: np.ndarray, metric, on_device: bool = False):
    """GPU counterpart of :func:`copairs.fastap.pair_similarity`.

    The kernel computes in float64 like the Numba one, from the same unit
    rows, and returns float32 similarities: a NumPy array, or with
    ``on_device`` a :class:`DeviceSims`. Pairs are checked on the host and
    copied to the GPU in chunks.
    """
    # The kernels index rows with uint32 pairs.
    rows = None if len(feats) >= 2**32 else kernel_rows(feats, metric)
    if rows is None:
        return None
    x, kind = cp.asarray(rows[0]), np.int32(rows[1])
    name = "pair_sims_f64" if x.dtype == cp.float64 else "pair_sims_f32"
    d = np.int64(x.shape[1])

    def sims_of(device_pairs):
        n = len(device_pairs)
        out = cp.empty(n, dtype=cp.float32)
        if n:
            args = (x, d, device_pairs, np.int64(n), kind, out)
            _kernel(name)(_cuda.grid(32 * n), (_cuda.BLOCK,), args)
        if metric == "abs_cosine":
            cp.abs(out, out=out)
        return out

    def similarity(pairs):
        sims = DeviceSims(sims_of, index_pairs(pairs, len(x)), pairs)
        return sims if on_device else sims.get()

    return similarity


def similarity_fn(feats, distance, generic):
    """:func:`copairs.fastap.similarity_fn` with the GPU kernels.

    The kernels' similarities are :class:`DeviceSims`, left for
    :func:`ap_from_pairs` to compute; distances without a kernel use
    ``generic``.
    """
    kernel = pair_similarity(feats, distance, on_device=True)
    return _bind_similarity(feats, kernel, generic)


def _pair_sims(sims, pairs, given):
    """:func:`copairs.fastap.ranking.pair_sims` for NumPy, CuPy or device sims.

    :class:`DeviceSims` must belong to these pairs: ``given``, the pairs
    object passed in, or equal pairs.
    """
    if isinstance(sims, DeviceSims):
        if sims.source is not given and not np.array_equal(sims.pairs, pairs):
            raise ValueError("these similarities were computed for other pairs")
        return sims
    return ranking.pair_sims(sims, pairs, cp if _cuda.is_device_array(sims) else np)


def _chunk_sims(sims, start: int, device_pairs):
    """Similarities of ``device_pairs``, pairs ``start:`` of ``sims``, on the GPU."""
    if isinstance(sims, DeviceSims):
        return sims.of(device_pairs)
    return cp.asarray(sims[start : start + len(device_pairs)])


def ap_from_pairs(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    pos_sims,
    neg_sims,
    budget_bytes: int = 2**30,
    n: int | None = None,
):
    """:func:`copairs.fastap.ap_from_pairs` on the GPU; returns NumPy arrays.

    Similarities may be NumPy or CuPy arrays, or :class:`DeviceSims`, which
    are computed here. The GPU holds the positive pairs, their sorted keys
    and the histograms, and streams the negative pairs with their
    similarities and keys, at most ``budget_bytes`` of them at a time. The
    histograms add up across chunks, so chunking does not change the result.
    ``n`` of ``2**32`` or more, whose indices the GPU sort cannot pack, runs
    on Numba, where ``budget_bytes`` bounds its histograms.
    """
    given = pos_pairs, neg_pairs
    pos_pairs, neg_pairs, n = ranking.checked_pairs(pos_pairs, neg_pairs, n)
    pos_sims = _pair_sims(pos_sims, pos_pairs, given[0])
    neg_sims = _pair_sims(neg_sims, neg_pairs, given[1])
    if n >= 2**32:
        return ranking.ap_from_pairs(
            pos_pairs,
            neg_pairs,
            _cuda.host(pos_sims),  # DeviceSims too, through __array__
            _cuda.host(neg_sims),
            budget_bytes,
            n,
        )
    pairs = _device_pairs(pos_pairs)
    keys = cp.repeat(rank_keys(_chunk_sims(pos_sims, 0, pairs), cp), 2)
    profile = pairs.ravel()
    order = cp.argsort((profile.astype(cp.uint64) << 32) | sortable_keys(keys, cp))
    vals = cp.ascontiguousarray(keys[order])
    ptr = cp.searchsorted(
        profile[order].astype(cp.int64), cp.arange(n + 1, dtype=cp.int64)
    )
    del pairs, profile, keys, order
    hist = cp.zeros(int(ptr[-1]) + n, dtype=cp.uint64)
    step = max(1, budget_bytes // _NEGATIVE_BYTES)
    for start in range(0, len(neg_pairs), step):
        pairs = _device_pairs(neg_pairs[start : start + step])
        keys = rank_keys(_chunk_sims(neg_sims, start, pairs), cp)
        args = (pairs, keys, np.int64(len(pairs)), ptr, vals, hist)
        _kernel("negative_hist")(_cuda.grid(2 * len(pairs)), (_cuda.BLOCK,), args)
        del pairs, keys  # free this chunk before the next one
    ap, n_neg = cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.uint64)
    _kernel("ap_from_hist")(
        _cuda.grid(n), (_cuda.BLOCK,), (ptr, hist, np.int64(n), ap, n_neg)
    )
    num_pos = cp.diff(ptr).get()
    return ranking.paired_results(ap.get(), num_pos, n_neg.get().astype(np.int64))


# draw_ap holds k float32 keys and k uint32 bins in shared memory, of which a
# kernel gets 48 KiB without opting in. Larger draws run on Numba.
_MAX_DRAW_QUERIES = 48 * 1024 // 8

_DRAW_KERNEL = r"""
// NumPy order of float32 keys: NaN after every number, and ties with NaN.
__device__ __forceinline__ bool key_lt(float a, float b) {
    return !isnan(a) && (isnan(b) || a < b);
}

// AP of each (draw, query) row of sims (n_rows, k + m), ranked by the float32 key
// 1 - similarity with a positive ahead of a negative of the same key, as
// copairs.fastap.draws._row_ap: the first k columns are the draw's queries (the
// row's own column is skipped), the rest its references.
extern "C" __global__ void draw_ap(const float* sims, int k, int m, long long n_rows,
                                   double* ap) {
    extern __shared__ float smem[];
    float* pos = smem;                                   // k - 1 keys, sorted ascending
    unsigned int* hist = (unsigned int*)(smem + k);      // k bins
    int n = k - 1, size = 1;
    while (size < n) size <<= 1;
    for (long long row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const float* s = sims + row * (long long)(k + m);
        int q = (int)(row % k);
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            hist[j] = 0u;
            if (j != q) pos[j - (j > q)] = 1.0f - s[j];
        }
        __syncthreads();
        // Bitonic sort of the size-padded keys whose compare-exchanges all put the
        // larger key at the higher slot: slots from n on act as keys above every
        // other (NaN included), so exchanges that reach them are skipped. Tied keys
        // compare alike everywhere below, so their order does not matter.
        for (int width = 2; width <= size; width <<= 1) {
            for (int stride = width >> 1; stride > 0; stride >>= 1) {
                for (int t = threadIdx.x; t < size >> 1; t += blockDim.x) {
                    int a, b;
                    if (stride == width >> 1) {  // compare mirrored slots of a block
                        a = (t / stride) * width + t % stride;
                        b = (t / stride) * width + width - 1 - t % stride;
                    } else {
                        a = (t / stride) * 2 * stride + t % stride;
                        b = a + stride;
                    }
                    if (b < n && key_lt(pos[b], pos[a])) {
                        float v = pos[a];
                        pos[a] = pos[b];
                        pos[b] = v;
                    }
                }
                __syncthreads();
            }
        }
        for (int e = threadIdx.x; e < m; e += blockDim.x) {
            // bin: the positives ranked ahead of this negative
            atomicAdd(hist + upper_bound(pos, 0, k - 1, 1.0f - s[k + e]), 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long before;
            ap[row] = ap_from_counts(hist, k - 1, &before);
        }
        __syncthreads();
    }
}
"""
# All the module's kernels, compiled as one module per device.
_SOURCE = _cuda.PRELUDE + _PAIR_KERNELS + _DRAW_KERNEL


def draw_aps(x: np.ndarray, idx: np.ndarray, k: int, budget_bytes: int) -> np.ndarray:
    """``backend="cuda"`` of :func:`copairs.fastap.draw_average_precisions`.

    Takes the arguments of :func:`copairs.fastap.draws.draw_aps`, which
    scores draws of more than ``_MAX_DRAW_QUERIES`` queries on Numba. Each
    chunk of draws' similarities comes from one batched matrix product on
    the GPU, and its APs are copied to the host before the next. The rows
    stay on the GPU when they take at most half of ``budget_bytes``, and the
    chunks get the rest; otherwise each chunk's rows are gathered on the
    host and copied.
    """
    if k > _MAX_DRAW_QUERIES:
        return draws.draw_aps(x, idx, k, budget_bytes)
    n_draws, width = idx.shape
    m = width - k
    resident = 2 * x.nbytes <= budget_bytes
    rows_of = cp.asarray(x) if resident else x
    budget_bytes -= x.nbytes if resident else 0
    out = np.empty((n_draws, k), dtype=np.float64)
    kernel = _kernel("draw_ap")
    threads = 128 if k + m >= 128 else 32
    step = draws._chunk(n_draws, k, m, x.shape[1], budget_bytes)
    for start in range(0, n_draws, step):
        chunk = idx[start : start + step]
        rows = rows_of[cp.asarray(chunk)] if resident else cp.asarray(x[chunk])
        sims = cp.ascontiguousarray(cp.matmul(rows[:, :k], rows.transpose(0, 2, 1)))
        del rows
        n_rows = len(chunk) * k
        ap = cp.empty((len(chunk), k), dtype=cp.float64)
        args = (sims, np.int32(k), np.int32(m), np.int64(n_rows), ap)
        blocks = _cuda.grid(n_rows * threads, block=threads)  # a block per row
        kernel(blocks, (threads,), args, shared_mem=8 * k)
        ap.get(out=out[start : start + len(chunk)])
        del sims, ap  # free this chunk before gathering the next one
    return out
