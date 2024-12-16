"""Tests data filtering by query."""

import numpy as np
import pytest

from copairs.map.filter import evaluate_and_filter
from tests.helpers import simulate_random_dframe

SEED = 0


@pytest.fixture
def mock_dataframe():
    """Create a mock dataframe."""
    length = 10
    vocab_size = {"p": 3, "w": 3, "l": 10}
    pos_sameby = ["l"]
    pos_diffby = ["p"]
    rng = np.random.default_rng(SEED)
    df = simulate_random_dframe(length, vocab_size, pos_sameby, pos_diffby, rng)
    df.drop_duplicates(subset=pos_sameby, inplace=True)
    return df


def test_correct(mock_dataframe):
    """Test correct query."""
    df, parsed_cols = evaluate_and_filter(mock_dataframe, ["p == 'p1'", "w > 'w2'"])
    assert not df.empty
    assert "p" in parsed_cols and "w" in parsed_cols
    assert all(df["w"].str.extract(r"(\d+)")[0].astype(int) > 2)


def test_invalid_query(mock_dataframe):
    """Test invalid query."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_and_filter(mock_dataframe, ['l == "lHello"'])
    assert "Invalid combined query expression" in str(excinfo.value)
    assert "No data matched the query" in str(excinfo.value)


def test_empty_result(mock_dataframe):
    """Test empty result."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_and_filter(mock_dataframe, ['p == "p1"', 'p == "p2"'])
    assert "Duplicate queries for column" in str(excinfo.value)


def test_empty_result_from_valid_query(mock_dataframe):
    """Test empty result from valid query."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_and_filter(mock_dataframe, ['p == "p4"'])
    assert "No data matched the query" in str(excinfo.value)
