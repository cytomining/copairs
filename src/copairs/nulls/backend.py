"""Method and backend choices of the AP null sampler and p-values.

Numba (``pip install 'copairs[numba]'``) and CuPy (``pip install
'copairs[gpu]'``) are optional. Nothing here imports them at module level:
:func:`_import_numba_backend` and :func:`_import_cuda_backend` load the
kernels the first time their backend is used, ``"auto"`` checks for Numba,
and :func:`available_backends` for both.
"""

import warnings
import importlib.util
from functools import cache

from copairs.methods import check_backend
from copairs._backend import import_backend


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first.

    CUDA is listed when CuPy imports and compiles kernels for a visible GPU,
    Numba when the Numba backend imports. An installed Numba or CuPy that
    fails to import is left out with a warning naming the error, once, as is
    a GPU whose kernels fail to compile; a machine without a GPU or driver
    just has no CUDA backend. The check runs once per process. ``"auto"``
    picks among the CPU backends only (see :func:`resolve_backend`).
    """
    backends = _cpu_backends()
    if _installed("cupy") and _cuda_usable():
        backends.insert(0, "cuda")
    return backends


def _cpu_backends() -> list[str]:
    """``["numba", "numpy"]`` when the Numba backend imports, else ``["numpy"]``."""
    if _installed("numba") and _numba_imports():
        return ["numba", "numpy"]
    return ["numpy"]


@cache
def _installed(package: str) -> bool:
    """Whether ``package`` is installed, looked up once per process."""
    return importlib.util.find_spec(package) is not None


@cache
def _numba_imports() -> bool:
    """Whether the installed Numba backend imports; warn once if it does not."""
    try:
        _import_numba_backend()
    except ImportError as exc:
        warnings.warn(
            f"Numba is installed but the Numba backend failed to import ({exc!r}); "
            "backend='auto' uses NumPy, which gives identical results.",
            RuntimeWarning,
            stacklevel=4,
        )
        return False
    return True


@cache
def _cupy_imports() -> bool:
    """Whether the installed CuPy imports; warn once if it does not."""
    try:
        import_backend("copairs._cuda", "cuda")
    except ImportError as exc:
        warnings.warn(
            f"CuPy is installed but failed to import ({exc!r}); backend='auto' "
            "does not use the GPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


def _cuda_usable() -> bool:
    """Whether the CUDA backend can run here (see :func:`copairs._cuda.usable`)."""
    return _cupy_imports() and import_backend("copairs._cuda", "cuda").usable()


def resolve_backend(backend: str, gpu: bool = False) -> str:
    """Return a concrete backend name for ``backend``.

    ``"auto"`` picks ``"numba"`` when the Numba backend imports and
    ``"numpy"`` otherwise, including when an installed Numba fails to import.
    That choice cannot change results: for every null (AP, copied, joint and
    permutation) both backends return bitwise-identical samples, draws and
    counts (tested). It is made once per process. A Numba that imports but
    cannot compile a kernel raises from the first call that needs it, under
    ``"auto"`` too. ``"auto"`` never picks ``"cuda"``, which runs only when
    asked for: its samples equal NumPy's by error bounds rather than by
    identical arithmetic (see :mod:`copairs.nulls.cuda`), and it has not been
    tested bit for bit on a GPU. Only the AP nulls and their p-values have
    GPU kernels; their functions pass ``gpu=True``, and elsewhere (the
    joint and permutation nulls) ``"cuda"`` raises ValueError instead of
    running on the CPU. ``"numba"`` and ``"cuda"`` raise an ImportError
    naming the extra when their package is missing, and ``"cuda"`` a
    RuntimeError when no GPU is usable.
    """
    check_backend(backend)
    if backend == "cuda" and not gpu:
        raise ValueError(
            "backend='cuda' runs only the AP nulls and their p-values "
            "(null='copied' in the mAP functions); this null runs on "
            "'numba' or 'numpy'"
        )
    if backend == "auto":
        return _cpu_backends()[0]
    if backend == "numba":
        _import_numba_backend()
    elif backend == "cuda":
        _import_cuda_backend()
    return backend


def _import_numba_backend():
    """Return :mod:`copairs.nulls._numba`, or raise an ImportError naming the extra."""
    return import_backend("copairs.nulls._numba", "numba")


def _import_cuda_backend():
    """Return :mod:`copairs.nulls.cuda`; raise, naming the extra, without a GPU.

    ImportError without CuPy, RuntimeError when no GPU is usable.
    """
    import_backend("copairs._cuda", "cuda").require()
    return import_backend("copairs.nulls.cuda", "cuda")
