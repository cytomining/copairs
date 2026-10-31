"""Regressions for count normalization and early argument validation."""

import numpy as np
import pandas as pd
import pytest

from copairs.map import (
    get_map_pvalue,
    mean_average_precision_hierarchical,
)
from tests.test_review_adjustments import profiles


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
@pytest.mark.parametrize("dtype", ["Int64", "UInt32", "int64", "uint32", "float64"])
def test_numeric_count_dtypes(dtype, null, tmp_path):
    """All models normalize valid integer-valued count columns consistently."""
    ap, _ = profiles()
    converted = ap.astype({"n_pos_pairs": dtype, "n_total_pairs": dtype})
    kwargs = dict(null=null, progress_bar=False, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        get_map_pvalue(converted, ["g"], 20, **kwargs),
        get_map_pvalue(ap, ["g"], 20, **kwargs),
    )


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
def test_arrow_count_dtypes(null, tmp_path):
    """Arrow-backed integral columns follow the same validation and sampling path."""
    pytest.importorskip("pyarrow")
    ap, _ = profiles()
    converted = ap.astype(
        {"n_pos_pairs": "int64[pyarrow]", "n_total_pairs": "uint32[pyarrow]"}
    )
    kwargs = dict(null=null, progress_bar=False, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        get_map_pvalue(converted, ["g"], 20, **kwargs),
        get_map_pvalue(ap, ["g"], 20, **kwargs),
    )


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_workers_validated_before_sampling(workers):
    """Shared validation rejects invalid workers even for scores-only joint calls."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="max_workers"):
        get_map_pvalue(
            ap, ["g"], 0, null="joint", max_workers=workers, progress_bar=False
        )


@pytest.mark.parametrize("keys", [[], ["g", "g"], ["absent"], [None]])
def test_group_columns_validated_before_sampling(keys):
    """Empty, duplicate, missing and invalid column names get useful errors."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="sameby|column"):
        get_map_pvalue(ap, keys, 0, progress_bar=False)


@pytest.mark.parametrize("keys", [[], ["g", "g"], [None]])
def test_hierarchy_columns_validated_before_sampling(keys):
    """Scores-only cannot bypass invalid hierarchy specifications."""
    ap, _ = profiles()
    with pytest.raises(ValueError, match="hierarchical_by|column"):
        mean_average_precision_hierarchical(
            ap, ["batch", "g"], 0, 0.05, hierarchical_by=keys, progress_bar=False
        )


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "UInt64"])
def test_count_validation_preserves_large_integers(dtype):
    """Validation does not round integer counts through float64."""
    from copairs.map.map import _normalize_pair_counts

    counts = pd.DataFrame(
        {"n_pos_pairs": [2**53 + 1], "n_total_pairs": [2**63 - 1]}
    ).astype(dtype)
    np.testing.assert_array_equal(
        _normalize_pair_counts(counts), [[2**53 + 1, 2**63 - 1]]
    )


@pytest.mark.parametrize(
    "value,dtype",
    [
        (np.nan, "float64"),
        (np.inf, "float64"),
        (1.5, "float64"),
        (-1, "int64"),
        (2**63, "uint64"),
        (2**63, "float64"),
        (True, "bool"),
        (pd.NA, "Int64"),
    ],
)
def test_invalid_counts_raise_useful_errors(value, dtype):
    """Missing, fractional, nonfinite and out-of-range counts never reach a sampler."""
    from copairs.map.map import _normalize_pair_counts

    counts = pd.DataFrame(
        {"n_pos_pairs": [1], "n_total_pairs": pd.Series([value], dtype=dtype)}
    )
    with pytest.raises(ValueError, match="pair counts"):
        _normalize_pair_counts(counts)


@pytest.mark.parametrize("null", ["joint", "permutation", "copied"])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_excluded_group_counts_are_not_cast(null, tmp_path):
    """Excluded missing-key rows can contain missing counts without contaminating other groups."""
    ap, _ = profiles()
    expected = get_map_pvalue(
        ap[ap.g == "b"], ["g"], 20, null=null, progress_bar=False, cache_dir=tmp_path
    )
    ap["n_total_pairs"] = ap.n_total_pairs.astype(float)
    ap.loc[ap.g == "a", ["g", "n_total_pairs"]] = [None, np.nan]
    got = get_map_pvalue(
        ap, ["g"], 20, null=null, progress_bar=False, cache_dir=tmp_path
    )
    assert got.p_value.tolist() == expected.p_value.tolist()
