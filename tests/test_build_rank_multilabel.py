"""Test the concatenation of ranges."""

import numpy as np

from copairs.compute import concat_ranges


def naive_concat_ranges(start: np.ndarray, end: np.ndarray):
    """Concatenate ranges into a mask."""
    mask = []
    for s, e in zip(start, end):
        mask.extend(range(s, e))
    return np.asarray(mask, dtype=np.int32)


def test_concat_ranges():
    """Test the concatenation of ranges."""
    rng = np.random.default_rng()
    num_range = 5, 10
    start_range = 2, 10
    size_range = 3, 5
    for _ in range(50):
        num = rng.integers(*num_range)
        start = rng.integers(*start_range, size=num)
        end = start + rng.integers(*size_range, size=num)
        assert np.array_equal(
            concat_ranges(start, end), naive_concat_ranges(start, end)
        )
