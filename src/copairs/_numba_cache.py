"""Numba's on-disk cache keyed on copairs' sources, and the launch lock.

``numba.njit(cache=True)`` checks a cached kernel only against the file that
defines it. Jitted functions it calls from other files are compiled into the
same cache entry, so after an edit to one of those files Numba keeps loading
the old machine code. :func:`cached_njit` puts a digest of every source file
of copairs into the name Numba gives its cache entry, so an edit to any of
them leads to a new entry, compiled from the current source, whichever files
the kernel compiles.

Every parallel kernel of copairs runs under one lock (:func:`launch`).

Numba is imported only when a kernel is built, so importing this module does
not need the optional dependency.
"""

import types
import hashlib
import threading
from pathlib import Path
from functools import cache

# Held while a parallel kernel runs (see launch).
_LAUNCH = threading.Lock()


def launch(kernel, *args):
    """Run the parallel ``kernel`` on Numba's threads, one launch at a time.

    Callers may launch from several threads: users' threads, and the
    permutation null's pool of groups next to them. Numba's workqueue
    threading layer, its fallback without TBB or OpenMP, aborts the process
    on concurrent launches, and the OpenMP layer would start a team of
    threads per launching thread. Returns what ``kernel`` returns.
    """
    with _LAUNCH:
        return kernel(*args)


@cache
def source_digest() -> str:
    """Hex digest of the paths and contents of copairs' ``.py`` files."""
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode() + b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def cached_njit(**options):
    """``numba.njit(cache=True, **options)``, cached per content of copairs.

    For kernels that compile jitted functions of other files. Numba names a
    cache entry after the function's module and ``__qualname__``, so the
    kernel compiles a copy of the function whose qualname ends in
    :func:`source_digest`; the function itself is left unchanged. Entries
    for older sources are never read again.
    """
    import numba

    suffix = source_digest()

    def jit(func):
        copy = types.FunctionType(
            func.__code__,
            func.__globals__,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        copy.__kwdefaults__ = func.__kwdefaults__
        copy.__doc__ = func.__doc__
        copy.__qualname__ = f"{func.__qualname__}_{suffix}"
        return numba.njit(cache=True, **options)(copy)

    return jit
