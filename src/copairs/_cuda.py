"""CuPy runtime shared by copairs' CUDA kernels: probe, compilation, launch sizing.

Imports CuPy at the top, so copairs loads this module only through
:func:`copairs._backend.import_backend`, when the ``"cuda"`` backend is
requested or :func:`copairs.nulls.available_backends` checks for a GPU
(``pip install 'copairs[gpu]'``). It does not depend on :mod:`copairs.fastap` or
:mod:`copairs.nulls`, whose CUDA kernels both use it.
"""

import os
import warnings
import functools
from typing import NamedTuple

import cupy as cp
import numpy as np

# Without FMA contraction, kernels round every operation as the CPU code they
# mirror does; a kernel that may fuse a multiply-add asks for it with fma().
OPTIONS = ("--fmad=false", "-std=c++14")
BLOCK = 256

_PROBE = 'extern "C" __global__ void probe(int* x) { if (x) *x = 1; }'

# Device functions shared by copairs' kernel sources, which start with it.
PRELUDE = r"""
// First index in sorted vals[lo:hi] after key (NumPy order, NaN last); mirrors
// copairs.fastap.ranking.upper_bound.
template <typename T, typename I>
__device__ __forceinline__ I upper_bound(const T* vals, I lo, I hi, T key) {
    bool key_nan = isnan(key);
    while (lo < hi) {
        I mid = (lo + hi) >> 1;
        T v = vals[mid];
        if (key_nan || (!isnan(v) && v <= key)) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// AP given hist[t] negatives ranked between positives t - 1 and t, in the order of
// copairs.fastap.ranking.ap_from_counts; *before gets the negatives ranked ahead of
// the last positive.
template <typename C>
__device__ __forceinline__ double ap_from_counts(const C* hist, long long num_pos,
                                                 unsigned long long* before) {
    unsigned long long n = 0;
    double acc = 0.0;
    for (long long t = 0; t < num_pos; ++t) {
        n += hist[t];
        acc += (double)(t + 1) / (double)(t + 1 + n);
    }
    *before = n;
    return acc / (double)num_pos;
}
"""
_INSTALL = "pip install 'copairs[gpu]'"


def usable() -> bool:
    """Whether CuPy sees a CUDA device and can compile kernels for it.

    Probed once per process. A child forked after its parent's probe found a
    device (initialising CUDA) cannot use CUDA, so it reports no usable GPU
    without calling CUDA, where a kernel load would fail as a compile failure.
    A visible GPU whose kernels fail to compile is a misconfiguration: the
    first call warns, at the caller of
    :func:`copairs.nulls.available_backends`, unless :func:`require` has
    raised for it already.
    """
    pid = os.getpid()
    probe = _probe(pid)
    if probe.broken and pid not in _reported:
        _reported.add(pid)
        warnings.warn(
            f"a CUDA device is visible, but {probe.reason}; the CUDA backend "
            "is unavailable",
            RuntimeWarning,
            stacklevel=4,  # usable < _cuda_usable < available_backends < caller
        )
    return probe.reason is None


def require() -> None:
    """Raise RuntimeError, naming the extra, unless :func:`usable`."""
    pid = os.getpid()
    reason = _probe(pid).reason
    if reason is not None:
        _reported.add(pid)
        raise RuntimeError(
            f"backend='cuda' needs a CUDA GPU that CuPy can compile kernels "
            f"for, and found none ({reason}). Install the GPU extra with "
            f"`{_INSTALL}` on a machine with a CUDA 12 driver, plus NVRTC (a "
            "CUDA Toolkit, or `pip install 'cupy-cuda12x[ctk]'`)."
        )


class _Probe(NamedTuple):
    """Outcome of the CUDA probe in one process.

    ``reason`` says why the backend is unusable (None when it is usable),
    ``broken`` whether a visible GPU's kernels fail to compile.
    """

    reason: str | None = None
    broken: bool = False


# The process whose probe found a CUDA device, which initialises CUDA there.
_initialised_in: int | None = None
# Processes told that their visible GPU's kernels fail to compile.
_reported: set[int] = set()


@functools.cache
def _probe(pid: int) -> _Probe:
    """Whether the CUDA backend is usable in process ``pid``, and why not."""
    global _initialised_in
    if _initialised_in not in (None, pid):
        # Forked after CUDA was initialised: CUDA cannot run here.
        return _Probe(f"forked from process {_initialised_in}, which initialised CUDA")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return _Probe("no CUDA device")
    except (RuntimeError, ImportError, OSError) as exc:
        # No driver, an insufficient one, or CUDA libraries that fail to load:
        # no GPU, so backend="auto" uses the CPU without a warning.
        return _Probe(f"{type(exc).__name__}: {exc}")
    _initialised_in = pid
    try:
        # A visible GPU is not enough: without NVRTC (e.g. CuPy without its
        # [ctk] extra or a CUDA Toolkit) every kernel fails to compile.
        cp.RawKernel(_PROBE, "probe", options=OPTIONS).compile()
    except (
        RuntimeError,
        ImportError,
        OSError,
        cp.cuda.compiler.CompileException,
    ) as exc:
        return _Probe(f"kernels fail to compile ({type(exc).__name__}: {exc})", True)
    return _Probe()


@functools.cache
def _compiled(source: str, device_id: int):
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=source, options=OPTIONS)


@functools.cache
def _function(source: str, name: str, device_id: int):
    with cp.cuda.Device(device_id):
        return _compiled(source, device_id).get_function(name)


def kernel(source: str, name: str):
    """Kernel ``name`` of CUDA ``source``, compiled and fetched once per device."""
    return _function(source, name, cp.cuda.Device().id)


@functools.cache
def _multiprocessors(device_id: int) -> int:
    # One attribute: Device().attributes queries all of them.
    runtime = cp.cuda.runtime
    return runtime.deviceGetAttribute(runtime.cudaDevAttrMultiProcessorCount, device_id)


def grid(
    n: int, block: int = BLOCK, blocks_per_sm: int = 32, rows: int | None = None
) -> tuple[int, ...]:
    """Grid for ``n`` threads in blocks of ``block``, capped per multiprocessor.

    The kernels loop over the rest. With ``rows``, the grid also has a y
    dimension of ``rows`` blocks, capped at CUDA's 65535.
    """
    sms = _multiprocessors(cp.cuda.Device().id)
    x = int(max(1, min((n + block - 1) // block, sms * blocks_per_sm)))
    return (x,) if rows is None else (x, int(max(1, min(rows, 65535))))


def is_device_array(x) -> bool:
    """Whether ``x`` is a CuPy array (on the GPU)."""
    return isinstance(x, cp.ndarray)


def host(x):
    """``x`` as a NumPy array, copied from the GPU if it is a CuPy array."""
    return x.get() if is_device_array(x) else np.asarray(x)
