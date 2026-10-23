"""Import copairs' optional backends and find which are usable.

The backend names are in :mod:`copairs.methods`. A module of its own, free of
SciPy, Numba and CuPy, so that copairs.matching can import it at the top
while ``import copairs`` stays light.
"""

import warnings
import importlib
import importlib.util
from types import ModuleType
from functools import cache

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


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first.

    CUDA is listed when CuPy imports and compiles kernels for a visible GPU,
    Numba when the Numba backend imports. An installed Numba or CuPy that
    fails to import is left out with a warning naming the error, once, as is
    a GPU whose kernels fail to compile; a machine without a GPU or driver
    just has no CUDA backend. The check runs once per process. ``"auto"``
    picks among the CPU backends only (see
    :func:`copairs.nulls.resolve_backend`).
    """
    backends = cpu_backends()
    if _installed("cupy") and _cuda_usable():
        backends.insert(0, "cuda")
    return backends


def cpu_backends() -> list[str]:
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
        import_backend("copairs.nulls._numba", "numba")
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
