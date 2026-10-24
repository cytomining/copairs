"""Method and backend choices of the AP null sampler and p-values.

Numba (``pip install 'copairs[numba]'``) and CuPy (``pip install
'copairs[gpu]'``) are optional. Nothing here imports them at module level:
:func:`_import_numba_backend` and :func:`_import_cuda_backend` load the
kernels the first time their backend is used, and ``"auto"`` checks for
Numba (:func:`copairs._backend.available_backends` lists both).
"""

from copairs.methods import check_backend
from copairs._backend import kernels, cpu_backends, step_backend


def resolve_backend(backend: str, step: str = "map_nulls") -> str:
    """Return the concrete backend that ``step`` of the nulls runs on.

    ``"auto"`` picks ``"numba"`` when the Numba backend imports and
    ``"numpy"`` otherwise, including when an installed Numba fails to import.
    That choice cannot change results: for every null (AP, copied, joint and
    permutation) both backends return bitwise-identical samples, draws and
    counts (tested). It is made once per process. A Numba that imports but
    cannot compile a kernel raises from the first call that needs it, under
    ``"auto"`` too. ``"auto"`` never picks ``"cuda"``, which runs only when
    asked for: its samples equal NumPy's by error bounds rather than by
    identical arithmetic (see :mod:`copairs.nulls.cuda`), and it has not been
    tested bit for bit on a GPU. Only the AP nulls and their p-values
    (``step="ap_nulls"``) have GPU kernels; under ``"cuda"`` the joint and
    permutation nulls (``"map_nulls"``, the default) run on Numba, as the AP
    stage's CPU steps do. ``"numba"`` and ``"cuda"`` raise an ImportError
    naming the extra when their package is missing, and GPU kernels a
    RuntimeError when no GPU is usable.
    """
    check_backend(backend)
    if backend == "auto":
        return cpu_backends()[0]
    runs_on = step_backend(step, backend)
    if runs_on == "cuda":
        _import_cuda_backend()
    elif runs_on == "numba":
        kernels(step, backend)  # a missing package names the extra of backend
    return runs_on


def _import_numba_backend():
    """Return :mod:`copairs.nulls._numba`, or raise an ImportError naming the extra."""
    return kernels("map_nulls", "numba")


def _import_cuda_backend():
    """Return :mod:`copairs.nulls.cuda`; raise, naming the extra, without a GPU.

    ImportError without CuPy, RuntimeError when no GPU is usable.
    """
    return kernels("ap_nulls", "cuda")
