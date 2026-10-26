"""Numba's on-disk cache keyed on every source file a kernel compiles.

``numba.njit(cache=True)`` checks a cached kernel only against the file that
defines it. Jitted functions it calls from other files are compiled into the
same cache entry, so after an edit to one of those files Numba keeps loading
the old machine code. :func:`cached_njit` puts a digest of the source files a
kernel depends on into the name Numba gives its cache entry, so an edit to any
of them leads to a new entry, compiled from the current source.

Numba is imported only when a kernel is built, so importing this module does
not need the optional dependency.
"""

import types
import hashlib
from pathlib import Path
from functools import cache


@cache
def source_digest(paths: tuple[str, ...]) -> str:
    """Hex digest of the contents of the files at ``paths``."""
    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def cached_njit(sources: tuple[str, ...], **options):
    """``numba.njit(cache=True, **options)``, cached per content of ``sources``.

    ``sources`` are the paths of the defining file and of every file whose
    jitted functions the kernel calls. Numba names a cache entry after the
    function's module and ``__qualname__``, so the kernel compiles a copy of
    the function whose qualname ends in :func:`source_digest` of ``sources``;
    the function itself is left unchanged. Entries for older sources are
    never read again.
    """
    import numba

    suffix = source_digest(tuple(sources))

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
