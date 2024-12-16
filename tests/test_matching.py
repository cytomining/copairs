"""Test functions for Matcher."""

from string import ascii_letters

import numpy as np
import pandas as pd
import pytest

from copairs import Matcher
from tests.helpers import create_dframe, simulate_plates, simulate_random_dframe

SEED = 0


def run_stress_sample_null(dframe, num_pairs):
    """Assert every generated null pair does not match any column."""
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    for _ in range(num_pairs):
        id1, id2 = matcher.sample_null_pair(dframe.columns)
        row1 = dframe.loc[id1]
        row2 = dframe.loc[id2]
        assert (row1 != row2).all()


def test_null_sample_large():
    """Test Matcher guarantees elements with different values."""
    dframe = create_dframe(32, 10000)
    run_stress_sample_null(dframe, 5000)


def test_null_sample_small():
    """Test Sample with small set."""
    dframe = create_dframe(3, 10)
    run_stress_sample_null(dframe, 100)


def test_null_sample_nan_vals():
    """Test NaN values are ignored."""
    dframe = create_dframe(4, 15)
    rng = np.random.default_rng(SEED)
    nan_mask = rng.random(dframe.shape) < 0.5
    dframe[nan_mask] = np.nan
    run_stress_sample_null(dframe, 1000)


def get_naive_pairs(dframe: pd.DataFrame, sameby, diffby):
    """Compute valid pairs using cross product from pandas."""
    cross = dframe.reset_index().merge(
        dframe.reset_index(), how="cross", suffixes=("_x", "_y")
    )
    index = True
    for col in sameby:
        index = (cross[f"{col}_x"] == cross[f"{col}_y"]) & index
    for col in diffby:
        index = (cross[f"{col}_x"] != cross[f"{col}_y"]) & index

    pairs = cross.loc[index, ["index_x", "index_y"]]
    # remove rows that pair themselves
    pairs = pairs[pairs["index_x"] != pairs["index_y"]]
    pairs = pairs.sort_values(["index_x", "index_y"]).reset_index(drop=True)
    return pairs


def check_naive(dframe, matcher: Matcher, sameby, diffby):
    """Check Matcher and naive generate same pairs."""
    gt_pairs = get_naive_pairs(dframe, sameby, diffby)
    vals = matcher.get_all_pairs(sameby, diffby)
    vals = sum(vals.values(), [])
    vals = pd.DataFrame(vals, columns=["index_x", "index_y"])
    vals = vals.sort_values(["index_x", "index_y"]).reset_index(drop=True)
    vals = set(vals.apply(frozenset, axis=1))
    gt_pairs = set(gt_pairs.apply(frozenset, axis=1))
    assert gt_pairs == vals


def check_simulated_data(length, vocab_size, sameby, diffby, rng):
    """Test sample of valid pairs from a simulated dataset."""
    dframe = simulate_random_dframe(length, vocab_size, sameby, diffby, rng)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    check_naive(dframe, matcher, sameby, diffby)


def test_stress_simulated_data():
    """Run multiple tests using simulated data."""
    rng = np.random.default_rng(SEED)
    num_cols_range = [2, 6]
    vocab_size_range = [5, 10]
    length_range = [100, 500]
    for _ in range(100):
        num_cols = rng.integers(*num_cols_range)
        length = rng.integers(*length_range)
        cols = ascii_letters[:num_cols]
        sizes = rng.integers(*vocab_size_range, size=num_cols)
        vocab_size = dict(zip(cols, sizes))
        ndiffby = np.clip(rng.integers(num_cols), 1, num_cols - 2)
        diffby = list(cols[:ndiffby])
        sameby = list(cols[ndiffby:])
        check_simulated_data(length, vocab_size, sameby, diffby, rng)


def test_empty_sameby():
    """Test query without sameby."""
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    check_naive(dframe, matcher, sameby=[], diffby=["w", "c"])
    check_naive(dframe, matcher, sameby=[], diffby=["w"])


def test_empty_diffby():
    """Test query without diffby."""
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    matcher.get_all_pairs(["c"], [])
    check_naive(dframe, matcher, sameby=["c"], diffby=[])
    check_naive(dframe, matcher, sameby=["w", "c"], diffby=[])


def test_raise_distjoint():
    """Test check for disjoint sameby and diffby."""
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    with pytest.raises(ValueError, match="must be disjoint lists"):
        matcher.get_all_pairs("c", ["w", "c"])


def test_raise_no_params():
    """Test check for at least one of sameby and diffby."""
    dframe = create_dframe(3, 10)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    with pytest.raises(ValueError, match="at least one should be provided"):
        matcher.get_all_pairs([], [])


def assert_sameby_diffby(dframe: pd.DataFrame, pairs_dict: dict, sameby, diffby):
    """Assert the pairs are valid."""
    for _, pairs in pairs_dict.items():
        for id1, id2 in pairs:
            for col in sameby:
                assert dframe.loc[id1, col] == dframe.loc[id2, col]
            for col in diffby:
                assert dframe.loc[id1, col] != dframe.loc[id2, col]


def test_simulate_plates_mult_sameby_large():
    """Test matcher successfully complete analysis of a large dataset."""
    dframe = simulate_plates(n_compounds=15000, n_replicates=20, plate_size=384)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    sameby = ["c", "w"]
    diffby = ["p"]
    pairs_dict = matcher.get_all_pairs(sameby, diffby)
    assert_sameby_diffby(dframe, pairs_dict, sameby, diffby)
