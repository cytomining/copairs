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


def import_kernels(module: str, backend: str) -> ModuleType:
    """Import ``module``, kernels of ``backend``, as :func:`import_backend` does.

    For ``"cuda"``, first checks for a usable GPU: a RuntimeError naming the
    extra without one (:func:`copairs._cuda.require`).
    """
    if backend == "cuda":
        import_backend("copairs._cuda", backend).require()
    return import_backend(module, backend)


# The AP stage's kernels of each optional backend.
_AP_KERNELS = {"numba": "copairs.fastap", "cuda": "copairs.fastap.cuda"}


def backend_module(backend: str, gpu: bool = True):
    """Return the kernels for ``backend``: None for ``"numpy"``.

    For ``"numba"``, imports the optional Numba kernels, :mod:`copairs.fastap`,
    on first use; for ``"cuda"``, :mod:`copairs.fastap.cuda`, after checking
    for a usable GPU (see :func:`import_kernels`). With ``gpu=False``, for
    functions that run only on the CPU, ``"cuda"`` is not a backend.
    """
    if backend == "numpy":
        return None
    if backend not in _AP_KERNELS or (backend == "cuda" and not gpu):
        names = "'numpy', 'numba' or 'cuda'" if gpu else "either 'numpy' or 'numba'"
        raise ValueError(f"backend must be {names}.")
    return import_kernels(_AP_KERNELS[backend], backend)


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
    if _installed("numba") and _imports("numba"):
        return ["numba", "numpy"]
    return ["numpy"]


@cache
def _installed(package: str) -> bool:
    """Whether ``package`` is installed, looked up once per process."""
    return importlib.util.find_spec(package) is not None


# Per optional backend: the module whose import shows that its installed
# packages work, what failed to import, and what that leaves.
_PROBES = {
    "numba": (
        "copairs.nulls._numba",
        "Numba is installed but the Numba backend",
        "backend='auto' uses NumPy, which gives identical results",
    ),
    "cuda": (
        "copairs._cuda",
        "CuPy is installed but",
        "the CUDA backend is unavailable",
    ),
}


@cache
def _imports(backend: str) -> bool:
    """Whether ``backend``'s installed packages import; warn once if they do not.

    The warning points at the caller of :func:`available_backends`.
    """
    module, subject, outcome = _PROBES[backend]
    try:
        import_backend(module, backend)
    except ImportError as exc:
        warnings.warn(
            f"{subject} failed to import ({exc!r}); {outcome}.",
            RuntimeWarning,
            stacklevel=4,  # _imports < cpu_backends or _cuda_usable < available_backends
        )
        return False
    return True


def _cuda_usable() -> bool:
    """Whether the CUDA backend can run here (see :func:`copairs._cuda.usable`)."""
    return _imports("cuda") and import_backend("copairs._cuda", "cuda").usable()
