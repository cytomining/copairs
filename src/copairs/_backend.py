"""Resolve the ``backend=`` argument of the AP functions; import optional backends.

A module of its own, free of SciPy, Numba and CuPy, so that copairs.matching
can import it at the top while ``import copairs`` stays light.
"""

import importlib
from types import ModuleType

# Optional backends: the packages they import, by display name, and their extra.
# llvmlite is Numba's compiler, which the extras install with it. The CUDA AP
# stage also runs Numba kernels, so the gpu extra includes Numba.
_NUMBA = {"numba": "Numba", "llvmlite": "Numba"}
_OPTIONAL = {
    "numba": (_NUMBA, "numba"),
    "cuda": ({"cupy": "CuPy", **_NUMBA}, "gpu"),
}


def import_backend(module: str, backend: str) -> ModuleType:
    """Import ``module``, which needs the optional dependencies of ``backend``.

    Only a missing dependency itself, or Numba's llvmlite, becomes the
    ImportError that names the extra to install. Other import failures,
    including other modules a dependency fails to import, are raised
    unchanged.
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


def backend_module(backend: str, gpu: bool = True):
    """Return the kernels for ``backend``: None for ``"numpy"``.

    For ``"numba"``, imports the optional Numba kernels, :mod:`copairs.fastap`,
    on first use (see :func:`import_backend`). For ``"cuda"``,
    :mod:`copairs.fastap.cuda`, after checking for a usable GPU: a
    RuntimeError naming the extra without one. With ``gpu=False``, for
    functions that run only on the CPU, ``"cuda"`` is not a backend.
    """
    if backend == "numpy":
        return None
    if backend == "numba":
        return import_backend("copairs.fastap", backend)
    if backend != "cuda" or not gpu:
        names = "'numpy', 'numba' or 'cuda'" if gpu else "either 'numpy' or 'numba'"
        raise ValueError(f"backend must be {names}.")
    import_backend("copairs._cuda", backend).require()
    return import_backend("copairs.fastap.cuda", backend)


def cpu_backend(backend: str) -> str:
    """Return the backend of the steps that ``backend`` runs on the CPU.

    ``"cuda"`` runs them on Numba, as ``"numba"`` does.
    """
    return "numba" if backend == "cuda" else backend
