"""The implementation choices shared by copairs' public functions."""

METHODS = ("fast", "legacy")
BACKENDS = ("auto", "cuda", "numba", "numpy")
NULLS = ("permutation", "joint", "copied")


def check_backend(backend: str) -> None:
    """Raise ValueError unless ``backend`` names a known backend."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def check_method(method: str, backend: str = "auto") -> None:
    """Raise ValueError unless ``method`` and ``backend`` name known choices.

    The backend name is checked under ``method="legacy"`` too, which ignores
    it, so a misspelled backend fails whichever method is used.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    check_backend(backend)


def check_null(method: str, null: str | None) -> str:
    """Return the mAP null to use: ``null``, or the method's default.

    The fast method defaults to the ``"permutation"`` null; ``method="legacy"``
    reproduces copairs <= 0.5.5, whose only null is ``"copied"``.
    """
    if null is None:
        return "permutation" if method == "fast" else "copied"
    if null not in NULLS:
        raise ValueError(f"unknown null {null!r}; expected one of {NULLS}")
    if method == "legacy" and null != "copied":
        raise ValueError(f"method='legacy' supports only null='copied', got {null!r}")
    return null
