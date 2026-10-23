"""Method and backend choices of the AP null sampler and p-values.

Numba (``pip install 'copairs[numba]'``) and CuPy (``pip install
'copairs[gpu]'``) are optional. Nothing here imports them at module level:
:func:`_import_numba_backend` and :func:`_import_cuda_backend` load the
kernels the first time their backend is used, and ``"auto"`` checks for
Numba (:func:`copairs._backend.available_backends` lists both).
"""

from copairs.methods import check_backend
from copairs._backend import cpu_backends, import_kernels


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
        return cpu_backends()[0]
    if backend == "numba":
        _import_numba_backend()
    elif backend == "cuda":
        _import_cuda_backend()
    return backend


def _import_numba_backend():
    """Return :mod:`copairs.nulls._numba`, or raise an ImportError naming the extra."""
    return import_kernels("copairs.nulls._numba", "numba")


def _import_cuda_backend():
    """Return :mod:`copairs.nulls.cuda`; raise, naming the extra, without a GPU.

    ImportError without CuPy, RuntimeError when no GPU is usable.
    """
    return import_kernels("copairs.nulls.cuda", "cuda")
