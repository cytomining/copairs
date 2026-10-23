"""Resolve the ``backend=`` argument of the AP functions; import optional backends.

A module of its own, free of SciPy and Numba, so that copairs.matching can
import it at the top while ``import copairs`` stays light.
"""

import importlib
from types import ModuleType

# Optional backends: the packages they import, by display name, and their extra.
_OPTIONAL = {"numba": ({"numba": "Numba"}, "numba")}


def import_backend(module: str, backend: str) -> ModuleType:
    """Import ``module``, which needs the optional dependencies of ``backend``.

    Only a missing dependency itself becomes the ImportError that names the
    extra to install. Other import failures, including a dependency's own
    missing modules, are raised unchanged.
    """
    packages, extra = _OPTIONAL[backend]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in packages:
            raise
        raise ImportError(
            f"backend={backend!r} requires the optional {packages[exc.name]} "
            f"dependency. Install it with `pip install 'copairs[{extra}]'`."
        ) from exc


def backend_module(backend: str):
    """Return the kernels for ``backend``: None for ``"numpy"``.

    For ``"numba"``, imports the optional Numba kernels, :mod:`copairs.fastap`,
    on first use (see :func:`import_backend`).
    """
    if backend == "numpy":
        return None
    if backend != "numba":
        raise ValueError("backend must be either 'numpy' or 'numba'.")
    return import_backend("copairs.fastap", backend)
