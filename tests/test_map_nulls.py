"""Tests for the choice of mAP null in copairs' mAP functions."""

import numpy as np
import pandas as pd
import pytest

from copairs import compute
from copairs.map import get_map_pvalue, mean_average_precision


def ap_frame(rows):
    """Build an ``ap_scores`` frame from (group, ap, n_pos_pairs, n_total_pairs) rows."""
    frame = pd.DataFrame(
        rows, columns=["g", "average_precision", "n_pos_pairs", "n_total_pairs"]
    )
    frame["normalized_average_precision"] = 0.0
    return frame


def test_copied_null_averages_configuration_nulls(tmp_path):
    """null="copied" averages each member's configuration null, as copairs 0.5.5."""
    rows = [("a", 0.4, 2, 30)] * 3 + [("b", 0.3, 4, 40), ("b", 0.5, 5, 41)]
    frame = ap_frame(rows)
    got = get_map_pvalue(
        frame, ["g"], 500, 0, progress_bar=False, cache_dir=tmp_path, null="copied"
    )
    confs, rev_ix = np.unique(
        frame[["n_pos_pairs", "n_total_pairs"]].values, axis=0, return_inverse=True
    )
    null_dists = compute.get_null_dists(
        confs, 500, seed=0, cache_dir=tmp_path / "ref", progress_bar=False
    )
    for g, p in zip(got["g"], got["p_value"]):
        rows = np.flatnonzero(frame["g"] == g)
        null = null_dists[rev_ix.ravel()[rows]].mean(axis=0)
        score = frame["average_precision"].iloc[rows].mean()
        assert p == ((null > score).sum() + 1) / 501


def test_unknown_null():
    """Unknown nulls, and pos_pairs outside the joint null, raise."""
    frame = ap_frame([("a", 1.0, 1, 100)] * 2)
    kwargs = dict(null_size=10, threshold=0.05, seed=0, progress_bar=False)
    with pytest.raises(ValueError, match="unknown null"):
        mean_average_precision(frame, ["g"], null="exact", **kwargs)
    with pytest.raises(ValueError, match="joint null only"):
        mean_average_precision(
            frame, ["g"], null="copied", pos_pairs=np.array([[0, 1]]), **kwargs
        )
