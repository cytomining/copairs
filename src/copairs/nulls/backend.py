"""Method and backend choices of the AP null sampler and p-values.

Numba (``pip install 'copairs[numba]'``) and CuPy (``pip install
'copairs[gpu]'``) are optional. Nothing here imports them at module level:
:func:`_import_numba_backend` and :func:`_import_cuda_backend` load the
kernels the first time their backend is used, or ``"auto"`` checks for it.
"""

import warnings
import importlib.util
from functools import cache

from copairs._backend import import_backend

METHODS = ("fast", "legacy")
BACKENDS = ("auto", "cuda", "numba", "numpy")


def _check_backend(backend: str) -> None:
    """Raise ValueError unless ``backend`` names a known backend."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def check_method(method: str, backend: str = "auto") -> None:
    """Raise ValueError unless ``method`` and ``backend`` name known choices.

    The backend name is checked under ``method="legacy"`` too, which ignores
    it, so a misspelled backend fails whichever method is used.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    _check_backend(backend)


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first.

    CUDA is listed when CuPy imports and compiles kernels for a visible GPU,
    Numba when the Numba backend imports. An installed Numba or CuPy that
    fails to import is left out with a warning naming the error, once, as is
    a GPU whose kernels fail to compile; a machine without a GPU or driver
    just has no CUDA backend. The check runs once per process.
    """
    backends = ["numpy"]
    if importlib.util.find_spec("numba") is not None and _numba_imports():
        backends.insert(0, "numba")
    if importlib.util.find_spec("cupy") is not None and _cuda_usable():
        backends.insert(0, "cuda")
    return backends


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
            stacklevel=3,
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


def resolve_backend(backend: str) -> str:
    """Return a concrete backend name for ``backend``.

    ``"auto"`` picks the first of :func:`available_backends`: ``"cuda"``
    when CuPy can compile kernels for a visible GPU, else ``"numba"`` when
    the Numba backend imports, else ``"numpy"``. That choice cannot change
    results: under ``"auto"`` every backend returns bitwise-identical null
    samples and counts (tested), CUDA running its float64 gap search.
    ``"numba"`` and ``"cuda"`` raise an ImportError naming the extra when
    their package is missing, and ``"cuda"`` a RuntimeError when no GPU is
    usable.
    """
    _check_backend(backend)
    if backend == "auto":
        return available_backends()[0]
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
