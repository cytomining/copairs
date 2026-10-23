"""Method and backend choices of the AP null sampler and p-values.

Numba is optional (``pip install 'copairs[numba]'``). Nothing here imports it
at module level: :func:`_import_numba_backend` loads the compiled kernels the
first time the ``"numba"`` backend is used.
"""

import warnings
import importlib.util
from functools import cache

METHODS = ("fast", "legacy")
BACKENDS = ("auto", "numba", "numpy")


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

    Numba is listed when the Numba backend imports. An installed Numba that
    fails to import is left out, with a warning naming the error, once.
    """
    if importlib.util.find_spec("numba") is None or not _numba_imports():
        return ["numpy"]
    return ["numba", "numpy"]


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


def resolve_backend(backend: str) -> str:
    """Return a concrete backend name for ``backend``.

    ``"auto"`` picks ``"numba"`` when the Numba backend imports and
    ``"numpy"`` otherwise, including when an installed Numba fails to import.
    That choice cannot change results: both backends return bitwise-identical
    null samples and counts (tested). ``"numba"`` raises if it cannot import.
    """
    _check_backend(backend)
    if backend == "auto":
        return available_backends()[0]
    if backend == "numba":
        _import_numba_backend()
    return backend


def _import_numba_backend():
    """Return :mod:`copairs.nulls._numba`, or raise an ImportError naming the extra."""
    try:
        from copairs.nulls import _numba
    except ModuleNotFoundError as exc:
        if exc.name not in ("numba", "llvmlite"):
            raise
        raise ImportError(
            "backend='numba' requires the optional Numba dependency. "
            "Install it with `pip install 'copairs[numba]'`."
        ) from exc
    return _numba
