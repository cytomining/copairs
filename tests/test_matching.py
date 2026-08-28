"""Test functions for Matcher."""

from string import ascii_letters

import numpy as np
import duckdb
import pandas as pd
import pytest

from copairs import matching
from tests.helpers import create_dframe, simulate_plates, simulate_random_dframe
from copairs.matching import _validate, find_pairs

SEED = 0


def test_duckdb_threads_defaults_to_affinity_with_cap(monkeypatch):
    """The default respects process affinity and does not exceed eight workers."""
    monkeypatch.delenv("COPAIRS_DUCKDB_THREADS", raising=False)
    monkeypatch.setattr(matching.os, "sched_getaffinity", lambda _: set(range(32)))
    assert matching._duckdb_threads() == 8

    monkeypatch.setattr(matching.os, "sched_getaffinity", lambda _: {1, 3, 5})
    assert matching._duckdb_threads() == 3


def test_duckdb_threads_falls_back_to_cpu_count(monkeypatch):
    """Platforms without affinity support use the bounded logical CPU count."""
    monkeypatch.delenv("COPAIRS_DUCKDB_THREADS", raising=False)
    monkeypatch.delattr(matching.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(matching.os, "cpu_count", lambda: 16)
    assert matching._duckdb_threads() == 8


def test_duckdb_threads_handles_unavailable_cpu_count(monkeypatch):
    """Failed affinity lookup and an unknown CPU count retain one worker."""
    monkeypatch.delenv("COPAIRS_DUCKDB_THREADS", raising=False)

    def unavailable_affinity(_pid):
        raise OSError

    monkeypatch.setattr(matching.os, "sched_getaffinity", unavailable_affinity)
    monkeypatch.setattr(matching.os, "cpu_count", lambda: None)
    assert matching._duckdb_threads() == 1


@pytest.mark.parametrize("configured", ["", "0", "-1", "1.5", "invalid"])
def test_duckdb_threads_rejects_invalid_env(monkeypatch, configured):
    """The environment override must contain a positive integer."""
    monkeypatch.setenv("COPAIRS_DUCKDB_THREADS", configured)
    with pytest.raises(ValueError, match="must be a positive integer"):
        matching._duckdb_threads()


def test_duckdb_threads_accepts_positive_env_override(monkeypatch):
    """An explicit positive worker count overrides the default cap."""
    monkeypatch.setenv("COPAIRS_DUCKDB_THREADS", "12")
    assert matching._duckdb_threads() == 12


def _sorted_pairs(pairs):
    """Return pair rows in a deterministic order for set comparisons."""
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


@pytest.mark.parametrize(
    ("dframe", "expected"),
    [
        (
            pd.DataFrame({"same": [0, 0, 1, 1], "different": [0, 1, 0, 1]}),
            np.array([[0, 1], [2, 3]], dtype=np.uint32),
        ),
        (
            pd.DataFrame({"same": [0, 0], "different": [1, 1]}),
            np.empty((0, 2), dtype=np.uint32),
        ),
    ],
)
def test_find_pairs_pandas_relation_pair_set_parity(dframe, expected):
    """Pandas and DuckDB relation inputs retain identical pair sets."""
    pandas_result = find_pairs(dframe, ["same"], ["different"])
    with duckdb.connect(":memory:") as connection:
        relation = connection.from_df(dframe.reset_index())
        relation_result = find_pairs(relation, ["same"], ["different"])

    np.testing.assert_array_equal(_sorted_pairs(pandas_result), expected)
    np.testing.assert_array_equal(_sorted_pairs(relation_result), expected)


def test_find_pairs_relation_uses_origin_connection_without_materializing(monkeypatch):
    """Relations keep their connection settings and are not converted to pandas."""
    monkeypatch.setenv("COPAIRS_DUCKDB_THREADS", "invalid")

    def fail_on_materialization(_relation):
        raise AssertionError("DuckDB relation was materialized as a DataFrame")

    monkeypatch.setattr(duckdb.DuckDBPyRelation, "df", fail_on_materialization)
    with duckdb.connect(":memory:", config={"threads": "1"}) as connection:
        relation = connection.sql(
            "SELECT * FROM (VALUES (0, 0, 0), (1, 0, 1), (2, 1, 0)) "
            "AS rows(index, same, different)"
        )
        result = find_pairs(relation, ["same"], ["different"])

    np.testing.assert_array_equal(result, np.array([[0, 1]], dtype=np.uint32))


def test_find_pairs_pandas_pair_set_is_identical_across_thread_counts(monkeypatch):
    """Configured thread counts do not change pair identity or orientation."""
    dframe = pd.DataFrame(
        {
            "same": np.arange(48) % 5,
            "different": np.arange(48) % 7,
        }
    )
    connect = duckdb.connect
    connection_configs = []

    def recording_connect(*args, **kwargs):
        connection_configs.append(kwargs.get("config"))
        return connect(*args, **kwargs)

    monkeypatch.setattr(matching.duckdb, "connect", recording_connect)
    results = []
    for threads in (1, 4, 8):
        monkeypatch.setenv("COPAIRS_DUCKDB_THREADS", str(threads))
        results.append(find_pairs(dframe, ["same"], ["different"]))

    assert connection_configs == [{"threads": "1"}, {"threads": "4"}, {"threads": "8"}]
    np.testing.assert_array_equal(_sorted_pairs(results[1]), _sorted_pairs(results[0]))
    np.testing.assert_array_equal(_sorted_pairs(results[2]), _sorted_pairs(results[0]))


def run_stress_sample_null(dframe, num_pairs):
    """Assert every generated null pair does not match any column."""
    null_pair = find_pairs(dframe, dframe.columns, [], rev=True)
    randints = np.random.randint(len(null_pair), size=num_pairs)
    for i in randints:
        id1, id2 = null_pair[i]
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


def check_naive(dframe, sameby, diffby):
    """Check Matcher and naive generate same pairs."""
    gt_pairs = get_naive_pairs(dframe, sameby, diffby)
    vals = find_pairs(dframe, sameby, diffby)
    vals = pd.DataFrame(vals, columns=["index_x", "index_y"])
    vals = vals.sort_values(["index_x", "index_y"]).reset_index(drop=True)
    vals = set(vals.apply(frozenset, axis=1))
    gt_pairs = set(gt_pairs.apply(frozenset, axis=1))
    assert gt_pairs == vals


def check_simulated_data(length, vocab_size, sameby, diffby, rng):
    """Test sample of valid pairs from a simulated dataset."""
    dframe = simulate_random_dframe(length, vocab_size, sameby, diffby, rng)
    check_naive(dframe, sameby, diffby)


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
    check_naive(dframe, sameby=[], diffby=["w", "c"])
    check_naive(dframe, sameby=[], diffby=["w"])


def test_empty_diffby():
    """Test query without diffby."""
    dframe = create_dframe(3, 10)
    check_naive(dframe, sameby=["c"], diffby=[])
    check_naive(dframe, sameby=["w", "c"], diffby=[])


def test_raise_distjoint():
    """Test check for disjoint sameby and diffby."""
    dframe = create_dframe(3, 10)
    with pytest.raises(ValueError, match="must be disjoint lists"):
        find_pairs(dframe, "c", ["w", "c"])


def test_raise_no_params():
    """Test check for at least one of sameby and diffby."""
    dframe = create_dframe(3, 10)
    with pytest.raises(ValueError, match="at least one should be provided"):
        find_pairs(dframe, [], [])


def test_validate_string_inputs():
    """_validate should convert string inputs to tuples."""
    with pytest.deprecated_call():
        sameby, diffby = _validate("c", "p")
    assert sameby == ("c",)
    assert diffby == ("p",)


def assert_sameby_diffby(dframe: pd.DataFrame, pairs: dict, sameby, diffby):
    """Assert the pairs are valid."""
    for id1, id2 in pairs:
        for col in sameby:
            assert dframe.loc[id1, col] == dframe.loc[id2, col]
        for col in diffby:
            assert dframe.loc[id1, col] != dframe.loc[id2, col]


def test_simulate_plates_mult_sameby_large():
    """Test matcher successfully complete analysis of a large dataset."""
    dframe = simulate_plates(n_compounds=15000, n_replicates=20, plate_size=384)
    sameby = ["c", "w"]
    diffby = ["p"]
    pairs_dict = find_pairs(dframe, sameby, diffby)
    assert_sameby_diffby(dframe, pairs_dict, sameby, diffby)
