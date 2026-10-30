"""The implementation choices shared by copairs' public functions."""

NULLS = ("permutation", "joint", "copied")


def check_null(null: str) -> str:
    """Return ``null``; raise ValueError unless it names a known mAP null."""
    if null not in NULLS:
        raise ValueError(f"unknown null {null!r}; expected one of {NULLS}")
    return null
