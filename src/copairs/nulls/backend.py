"""Method and backend choices of the AP null sampler and p-values.

Numba is optional (``pip install 'copairs[numba]'``). Nothing here imports it
at module level: :func:`_import_numba_backend` loads the compiled kernels the
first time the ``"numba"`` backend is used.
"""

import warnings
import importlib.util
from functools import cache

from copairs.methods import check_backend


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first.

    Numba is listed when the Numba backend imports. An installed Numba that
    fails to import is left out, with a warning naming the error, once.
    """
    return list(_available())


@cache
def _available() -> tuple[str, ...]:
    """:func:`available_backends`, found once per process."""
    if importlib.util.find_spec("numba") is None or not _numba_imports():
        return ("numpy",)
    return ("numba", "numpy")


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


def resolve_backend(backend: str) -> str:
    """Return a concrete backend name for ``backend``.

    ``"auto"`` picks ``"numba"`` when the Numba backend imports and
    ``"numpy"`` otherwise, including when an installed Numba fails to import.
    That choice cannot change results: for every null (AP, copied, joint and
    permutation) both backends return bitwise-identical samples, draws and
    counts (tested). It is made once per process. A Numba that imports but
    cannot compile a kernel raises from the first call that needs it, under
    ``"auto"`` too. ``"numba"`` raises if it cannot import.
    """
    check_backend(backend)
    if backend == "auto":
        return _available()[0]
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
