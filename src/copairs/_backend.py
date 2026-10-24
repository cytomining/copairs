"""Import copairs' optional backends and find which are usable.

The backend names are in :mod:`copairs.methods`. A module of its own, free of
SciPy, Numba and CuPy, so that copairs.matching can import it at the top
while ``import copairs`` stays light.
"""

import os
import sys
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


# Each step's kernel modules on the optional backends. Under "cuda", a step
# without GPU kernels runs on Numba, which the gpu extra installs.
_KERNELS = {
    # The AP stage: similarities, APs by counting, batched draws.
    "ap": {"numba": "copairs.fastap", "cuda": "copairs.fastap.cuda"},
    # The AP stage's CPU steps: multilabel matching and APs, normalized AP.
    "ap_cpu": {"numba": "copairs.fastap"},
    # The AP nulls and their p-values (the copied mAP null).
    "ap_nulls": {"numba": "copairs.nulls._numba", "cuda": "copairs.nulls.cuda"},
    # The joint and permutation mAP nulls.
    "map_nulls": {"numba": "copairs.nulls._numba"},
}
# The CUDA backend's GPU check (copairs._cuda.usable, require).
_CUDA = "copairs._cuda"


def step_backend(step: str, backend: str) -> str:
    """Return the backend that ``step`` (a key of ``_KERNELS``) runs on for ``backend``.

    ``backend`` where the step has its kernels, and NumPy always; otherwise
    ``"numba"``: under ``"cuda"``, a step without GPU kernels runs on the
    CPU. Raises ValueError unless ``backend`` is numpy, numba or cuda.
    """
    if backend not in ("numpy", "numba", "cuda"):
        raise ValueError("backend must be 'numpy', 'numba' or 'cuda'.")
    if backend == "numpy" or backend in _KERNELS[step]:
        return backend
    return "numba"


def kernels(step: str, backend: str) -> ModuleType | None:
    """Return ``step``'s kernels for ``backend``, or None where it runs on NumPy.

    Imports the module :func:`step_backend` picks on first use. GPU kernels
    first check for a usable GPU: a RuntimeError naming the extra without
    one (:func:`copairs._cuda.require`). A missing package names the extra
    of ``backend``, so ``"cuda"`` names the gpu extra for its Numba steps too.
    """
    runs_on = step_backend(step, backend)
    if runs_on == "numpy":
        return None
    if runs_on == "cuda":
        import_backend(_CUDA, "cuda").require()
    return import_backend(_KERNELS[step][runs_on], backend)


def backend_module(backend: str) -> ModuleType | None:
    """Return the AP stage's kernels for ``backend``: None for ``"numpy"``.

    :mod:`copairs.fastap` for ``"numba"``, :mod:`copairs.fastap.cuda` for
    ``"cuda"`` (see :func:`kernels`).
    """
    return kernels("ap", backend)


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
        _KERNELS["map_nulls"]["numba"],
        "Numba is installed but the Numba backend",
        "backend='auto' uses NumPy, which gives identical results",
    ),
    "cuda": (
        _CUDA,
        "CuPy is installed but",
        "the CUDA backend is unavailable",
    ),
}


@cache
def _imports(backend: str) -> bool:
    """Whether ``backend``'s installed packages import; warn once if they do not.

    The warning points at the call into copairs that led here.
    """
    module, subject, outcome = _PROBES[backend]
    try:
        import_backend(module, backend)
    except ImportError as exc:
        warnings.warn(
            f"{subject} failed to import ({exc!r}); {outcome}.",
            RuntimeWarning,
            stacklevel=_caller_level(),
        )
        return False
    return True


_PACKAGE = os.path.dirname(__file__) + os.sep


def _caller_level() -> int:
    """Return the ``stacklevel`` that points the caller's warning out of copairs.

    It names the first frame outside the package. The call paths into a
    backend lookup differ in depth (available_backends, or "auto" deep in a
    null), so the level is counted, not fixed.
    """
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    return level


def _cuda_usable() -> bool:
    """Whether the CUDA backend can run here (see :func:`copairs._cuda.usable`)."""
    return _imports("cuda") and import_backend(_CUDA, "cuda").usable()
