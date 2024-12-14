"""Test matching with `any` conditions using simulated data."""

from string import ascii_letters

import numpy as np
import pandas as pd

from copairs import Matcher
from tests.helpers import simulate_random_dframe

SEED = 42


def get_naive_pairs(dframe: pd.DataFrame, sameby, diffby):
    """Compute valid pairs using cross product from pandas."""
    cross = dframe.reset_index().merge(
        dframe.reset_index(), how="cross", suffixes=("_x", "_y")
    )
    index_all = True
    for col in sameby["all"]:
        index_all = (cross[f"{col}_x"] == cross[f"{col}_y"]) & index_all
    for col in diffby["all"]:
        index_all = (cross[f"{col}_x"] != cross[f"{col}_y"]) & index_all

    index_same_by_any = not sameby["any"]
    for col in sameby["any"]:
        index_same_by_any = (cross[f"{col}_x"] == cross[f"{col}_y"]) | index_same_by_any

    index_diff_by_any = not diffby["any"]
    for col in diffby["any"]:
        index_diff_by_any = (cross[f"{col}_x"] != cross[f"{col}_y"]) | index_diff_by_any
    index_any = index_same_by_any & index_diff_by_any

    index = index_all & index_any
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
    sameby_cols = sameby["all"] + sameby["any"]
    diffby_cols = diffby["all"] + diffby["any"]
    dframe = simulate_random_dframe(length, vocab_size, sameby_cols, diffby_cols, rng)
    matcher = Matcher(dframe, dframe.columns, seed=SEED)
    check_naive(dframe, matcher, sameby, diffby)


def test_stress_simulated_data_any_all():
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
        sameby = {"all": [], "any": list(cols[ndiffby:])}
        diffby = {"all": list(cols[:ndiffby]), "any": []}
        check_simulated_data(length, vocab_size, sameby, diffby, rng)


def test_stress_simulated_data_all_all():
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
        diffby = {"all": list(cols[:ndiffby]), "any": []}
        sameby = {"all": list(cols[ndiffby:]), "any": []}
        check_simulated_data(length, vocab_size, sameby, diffby, rng)


def test_stress_simulated_data_all_any():
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
        ndiffby = np.clip(rng.integers(num_cols), 2, num_cols - 2)
        sameby = {"all": list(cols[:ndiffby]), "any": []}
        diffby = {"all": [], "any": list(cols[ndiffby:])}
        check_simulated_data(length, vocab_size, sameby, diffby, rng)


def test_stress_simulated_data_any_any():
    """Run multiple tests using simulated data."""
    rng = np.random.default_rng(SEED)
    num_cols_range = [4, 6]
    vocab_size_range = [5, 10]
    length_range = [100, 500]
    for _ in range(100):
        num_cols = rng.integers(*num_cols_range)
        length = rng.integers(*length_range)
        cols = ascii_letters[:num_cols]
        sizes = rng.integers(*vocab_size_range, size=num_cols)
        vocab_size = dict(zip(cols, sizes))
        ndiffby = np.clip(rng.integers(num_cols), 2, num_cols - 2)
        diffby = {"all": [], "any": list(cols[:ndiffby])}
        sameby = {"all": [], "any": list(cols[ndiffby:])}
        check_simulated_data(length, vocab_size, sameby, diffby, rng)
