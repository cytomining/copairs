"""Tests for the counter-based random stream and the shared null helpers."""

import numpy as np
import pytest

from copairs.nulls import common, philox


@pytest.mark.parametrize(
    "ctr,key,expected",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF,) * 2,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_known_answers(ctr, key, expected):
    """Philox4x32-10 matches the Random123 known-answer vectors."""
    words = philox.philox4x32(*map(np.uint64, ctr), *map(np.uint64, key))
    assert [int(w) for w in words] == list(expected)


def test_uniform53_range_and_resolution():
    """Uniforms lie in [0, 1) and carry 53 bits: all-ones words map just below 1."""
    u = np.uint64
    assert philox.uniform53(u(0), u(0)) == 0.0
    top = philox.uniform53(u(0xFFFFFFFF), u(0xFFFFFFFF))
    assert top < 1.0 and top == 1.0 - 2.0**-53


def scalar_config_key(seed, num_pos, total):
    """Philox key of one configuration, one block at a time."""
    u = np.uint64
    w = philox.philox4x32(
        u(num_pos),
        u(total),
        u(seed & 0xFFFFFFFF),
        u(seed >> 32),
        philox.SALT0,
        philox.SALT1,
    )
    return int(w[0]), int(w[1])


@pytest.mark.parametrize("seed", [0, 3, 2**33 + 7, 2**64 - 1])
def test_config_key_arrays_match_scalar(seed):
    """Vectorized configuration keys equal the scalar derivation."""
    rng = np.random.default_rng(seed % 1000)
    num_pos = rng.integers(1, 2**32 - 1, 500)
    total = rng.integers(1, 2**32 - 1, 500)
    k0, k1 = philox.config_key_arrays(seed, num_pos, total)
    expected = [scalar_config_key(seed, int(p), int(t)) for p, t in zip(num_pos, total)]
    np.testing.assert_array_equal(
        np.stack([k0, k1], axis=1), np.array(expected, dtype=np.uint64)
    )


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_seed_out_of_range(seed):
    """Seeds outside [0, 2**64) raise."""
    with pytest.raises(ValueError, match="seed must be"):
        philox.config_key_arrays(seed, np.array([1]), np.array([2]))


def test_config_keys_reject_wide_configurations():
    """Configurations that do not fit a 32-bit counter word raise."""
    with pytest.raises(ValueError, match="< 2\\*\\*32"):
        philox.config_key_arrays(0, np.array([1]), np.array([2**32]))


def test_resolve_seed():
    """An int seed passes through; None draws fresh entropy in [0, 2**64)."""
    assert common.resolve_seed(7) == 7
    seeds = {common.resolve_seed(None) for _ in range(4)}
    assert len(seeds) == 4 and all(0 <= s < 2**64 for s in seeds)


@pytest.mark.parametrize("null_size", [-1, -100, 2.5])
def test_check_null_size_rejects_invalid(null_size):
    """Negative or fractional null sizes raise instead of giving invalid p-values."""
    with pytest.raises(ValueError, match="null_size"):
        common.check_null_size(null_size)


def test_check_null_size_accepts_integral_floats():
    """An integral float null size is accepted and returned as an int."""
    assert common.check_null_size(1000.0) == 1000
    assert isinstance(common.check_null_size(1000.0), int)


def test_chunks_cover_null_lazily():
    """Chunks tile [0, null_size) and are generated lazily."""
    assert list(common.chunks(10, 4)) == [(0, 4), (4, 4), (8, 2)]
    assert list(common.chunks(0, 4)) == []
    assert next(common.chunks(10**12, 1)) == (0, 1)  # a generator: no 1e12 list
    assert common.chunk_size(100, 10, 8 * 10 * 7) == 7
    assert common.chunk_size(100, 10, 1) == 1
    assert common.chunk_size(5, 1, 2**30) == 5


def test_tie_thresholds():
    """float64 scores lose TIE_TOL; float32 scores lose their own rounding too."""
    s64 = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(common.tie_thresholds(s64), s64 - common.TIE_TOL)
    s32 = np.array([0.5, 1.0], dtype=np.float32)
    tol32 = common.TIE_ULPS * float(np.finfo(np.float32).eps) * s32.astype(np.float64)
    np.testing.assert_array_equal(
        common.tie_thresholds(s32), s32.astype(np.float64) - tol32
    )
    ints = common.tie_thresholds(np.array([1, 2]))
    assert ints.dtype == np.float64
    np.testing.assert_array_equal(ints, np.array([1.0, 2.0]) - common.TIE_TOL)
