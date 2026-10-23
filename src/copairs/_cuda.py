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

import cupy as cp
import numpy as np

# Without FMA contraction, kernels round every operation as the CPU code they
# mirror does; a kernel that may fuse a multiply-add asks for it with fma().
OPTIONS = ("--fmad=false", "-std=c++14")
BLOCK = 256

_PROBE = 'extern "C" __global__ void probe(int* x) { if (x) *x = 1; }'
_INSTALL = "pip install 'copairs[gpu]'"


def usable() -> bool:
    """Whether CuPy sees a CUDA device and can compile kernels for it.

    Probed once per process: a child forked after its parent initialised CUDA
    cannot use it, so the child probes again and finds no usable GPU.
    """
    return _probe(os.getpid()) is None


def require() -> None:
    """Raise RuntimeError, naming the extra, unless :func:`usable`."""
    reason = _probe(os.getpid())
    if reason is not None:
        raise RuntimeError(
            f"backend='cuda' needs a CUDA GPU that CuPy can compile kernels "
            f"for, and found none ({reason}). Install the GPU extra with "
            f"`{_INSTALL}` on a machine with a CUDA 12 driver, plus NVRTC (a "
            "CUDA Toolkit, or `pip install 'cupy-cuda12x[ctk]'`)."
        )


@functools.cache
def _probe(pid: int) -> str | None:
    """None when the CUDA backend is usable in process ``pid``, else why not."""
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return "no CUDA device"
    except (RuntimeError, ImportError, OSError) as exc:
        # No driver, an insufficient one, or CUDA libraries that fail to load:
        # no GPU, so backend="auto" uses the CPU without a warning.
        return f"{type(exc).__name__}: {exc}"
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
        reason = f"kernels fail to compile: {type(exc).__name__}: {exc}"
        warnings.warn(
            f"a CUDA device is visible but copairs' kernels fail to compile "
            f"({type(exc).__name__}: {exc}); the CUDA backend is unavailable",
            RuntimeWarning,
            stacklevel=2,
        )
        return reason
    return None


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


def grid(n: int, blocks_per_sm: int = 32) -> int:
    """1-D grid for ``n`` threads in blocks of :data:`BLOCK`, capped per multiprocessor."""
    sms = _multiprocessors(cp.cuda.Device().id)
    return int(max(1, min((n + BLOCK - 1) // BLOCK, sms * blocks_per_sm)))


def is_device_array(x) -> bool:
    """Whether ``x`` is a CuPy array (on the GPU)."""
    return isinstance(x, cp.ndarray)


def host(x):
    """``x`` as a NumPy array, copied from the GPU if it is a CuPy array."""
    return x.get() if is_device_array(x) else np.asarray(x)
