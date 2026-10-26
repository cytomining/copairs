"""Resolve the ``backend=`` argument of the AP functions.

A module of its own, free of SciPy and Numba, so that copairs.matching can
import it at the top while ``import copairs`` stays light.
"""


def backend_module(backend: str):
    """Return the kernels for ``backend``: None for ``"numpy"``.

    For ``"numba"``, imports the optional Numba kernels, :mod:`copairs.fastap`,
    on first use. Keeps transitive import failures: only a missing ``numba``
    itself becomes the ImportError that names the extra.
    """
    if backend == "numpy":
        return None
    if backend != "numba":
        raise ValueError("backend must be either 'numpy' or 'numba'.")
    try:
        from copairs import fastap
    except ModuleNotFoundError as exc:
        if exc.name != "numba":
            raise
        raise ImportError(
            "backend='numba' requires the optional Numba dependency. "
            "Install it with `pip install 'copairs[numba]'`."
        ) from exc
    return fastap
