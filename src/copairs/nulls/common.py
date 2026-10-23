"""Helpers shared by the mAP nulls: tie thresholds, seeds and null sizes.

A null value counts against an observed score when ``null >= score - tol``
(see :func:`tie_thresholds`). The tolerance absorbs rounding between the two
computations, so a null rank list identical to the observed one is counted as
a tie, as in the ``>=`` definition of a permutation p-value.
"""

import numpy as np

# Far above float64 summation-order differences of an AP (~k * 2**-52). Distinct
# AP values can lie closer than this in larger configurations (e.g. 9e-13 apart
# for 2 positives among 5000), so a null value just below a score may count as
# a tie; each such value has null probability ~1/C(total, num_pos).
TIE_TOL = 1e-10
# Rounding, in units of eps * |score|, allowed for scores below float64
# precision: float32 APs are within 0.48 of the exact value (measured for k up
# to 20000 positives), and a float32 mean adds at most 0.5.
TIE_ULPS = 8


def tie_thresholds(scores) -> np.ndarray:
    """float64 ``scores - tol``; null values at or above a threshold count.

    ``tol`` is :data:`TIE_TOL` for float64 scores. Lower-precision float scores,
    such as float32 APs, carry their own rounding, so ``tol`` also covers
    ``TIE_ULPS * eps * |score|`` of their dtype.
    """
    scores = np.asarray(scores)
    dtype = scores.dtype if scores.dtype.kind == "f" else np.dtype(np.float64)
    scores = scores.astype(np.float64)
    eps = float(np.finfo(dtype).eps)
    return scores - np.maximum(TIE_TOL, TIE_ULPS * eps * np.abs(scores))


def resolve_seed(seed: int | None) -> int:
    """Return ``seed`` as an int, or fresh OS entropy when it is ``None``.

    The range is checked where the seed is used, by
    :func:`copairs.nulls.philox.config_key_arrays`.
    """
    if seed is None:
        return int(np.random.SeedSequence().entropy % 2**64)
    return int(seed)


def check_null_size(null_size: int) -> int:
    """Return ``null_size`` as an int; raise unless it is a non-negative integer."""
    if int(null_size) != null_size or null_size < 0:
        raise ValueError(f"null_size must be a non-negative integer, got {null_size}")
    return int(null_size)


def chunk_size(null_size: int, rows: int, budget_bytes: int) -> int:
    """Return the samples per chunk that fit ``rows`` float64 nulls in the budget."""
    return max(1, min(null_size, budget_bytes // (8 * max(rows, 1))))


def chunks(null_size: int, size: int):
    """``(start, size)`` sample chunks, generated lazily."""
    return ((s, min(size, null_size - s)) for s in range(0, null_size, size))


def progress(items, total: int, enabled: bool, desc: str):
    """Wrap ``items`` in a progress bar when ``enabled`` and there are 2+ items."""
    if not enabled or total < 2:
        return items
    from tqdm.autonotebook import tqdm

    return tqdm(items, total=total, desc=desc, leave=False)
